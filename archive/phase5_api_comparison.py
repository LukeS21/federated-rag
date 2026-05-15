#!/usr/bin/env python3
"""Phase 6 API vs Local Comparison — true 1:1 survey‑graph comparison.

Runs the **exact same survey graph** (``build_survey_graph``) through both
local Ollama and DeepSeek API, to produce an apples‑to‑apples comparison of
per‑theme anchoring scores, claim counts, latency, and decomposition quality.

Model tiering (consistent with local Ollama split):
  Cloud (DeepSeek):
    Light tasks (per‑theme, extraction, decomposition) → deepseek‑chat
    Heavy tasks (cross‑theme, critic, arbiter, gap analysis) → deepseek‑v4‑pro
  Local (Ollama):
    Light → gemma4:e4b (OLLAMA_SMALL_MODEL)
    Heavy → qwen3.6:35b (OLLAMA_LARGE_MODEL)
    Gap  → gemma4:e4b (GAP_ANALYSIS_MODEL) — but gap uses deepseek‑v4‑pro for cloud

Cloud results persist to disk so local can be re‑run any number of times
without re‑paying API credits.

Usage:
    python phase5_api_comparison.py --run cloud          # DeepSeek (~$1, ~5 min)
    python phase5_api_comparison.py --run local          # Ollama (~5‑8 min, reads saved cloud)
    python phase5_api_comparison.py --run both           # Both sequentially (~10‑13 min)
    python -m pytest phase5_api_comparison.py -v         # Regression guard
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchoring.evidence_check import decompose_claims, compute_anchoring_score
from src.unicode_map import scrub_unicode
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.networkx_json_storage import NetworkXJSONStorage
from src.graph.graph_builder import build_survey_graph
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.pre_summarizer import PreSummarizer
from src.ingestion.pre_extractor import PreExtractor
from src.citation_manager.citekey_utils import (
    resolve_cite_key,
    parse_paper_metadata,
    try_zotero_add,
)
from src.ingestion.pdf_parser import (
    compute_content_hash,
    extract_title_from_chunks,
    check_content_duplicate,
    save_content_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_comparison")

# ── ANSI ───────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("projects/default")
COMPARISON_DIR = PROJECT_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
BM25_CORPUS_PATH = str(PROJECT_DIR / "bm25_corpus.json")
GRAPH_PATH = str(PROJECT_DIR / "project_graph.json")
QUERY_CACHE_DIR = PROJECT_DIR / "query_cache"

NUM_CTX = 16384
LLM_TIMEOUT = 900
os.environ.setdefault("LLM_TIMEOUT", str(LLM_TIMEOUT))
CLIENT_KWARGS = {"timeout": LLM_TIMEOUT}

# Most recent survey query (loaded from cache)
_DEFAULT_QUERY = ""


def _get_default_query() -> str:
    """Read the most recent query from the decomposition cache."""
    global _DEFAULT_QUERY
    if _DEFAULT_QUERY:
        return _DEFAULT_QUERY

    if not QUERY_CACHE_DIR.exists():
        return "How do T cells and macrophages coordinate bone healing around titanium implants?"

    latest_ts = 0
    for f in sorted(QUERY_CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("type") == "decomposition":
                ts = data.get("_cached_at", 0)
                if ts > latest_ts:
                    latest_ts = ts
                    _DEFAULT_QUERY = data.get("query", "").strip()
        except Exception:
            continue

    return _DEFAULT_QUERY if _DEFAULT_QUERY and len(_DEFAULT_QUERY) > 20 else (
        "How do T cells and macrophages coordinate bone healing around titanium "
        "implants? How does obesity impact this?"
    )


# ── Infrastructure setup (shared across runs) ──────────────────────────────

_infra: Dict[str, Any] | None = None


def _get_infrastructure() -> Dict[str, Any]:
    """Lazy-init retriever, graph storage, chroma (once)."""
    global _infra
    if _infra is not None:
        return _infra

    logger.info("Setting up comparison infrastructure...")

    chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)

    # Hybrid retrieval in anchoring
    from src.anchoring.evidence_check import set_anchoring_chroma
    set_anchoring_chroma(chroma)

    # Load BM25 from saved corpus
    bm25_path = Path(BM25_CORPUS_PATH)
    if bm25_path.exists():
        try:
            corpus = json.loads(bm25_path.read_text(encoding="utf-8"))
            if isinstance(corpus, list) and corpus:
                bm25.add_documents([str(x) for x in corpus])
        except Exception:
            pass
    else:
        # Build from ChromaDB
        try:
            data = chroma.collection.get(include=["documents"])
            docs = (data or {}).get("documents") or []
            if docs:
                bm25.add_documents([str(d) for d in docs])
        except Exception:
            pass

    graph_storage = NetworkXJSONStorage(GRAPH_PATH)

    _infra = {
        "chroma": chroma,
        "bm25": bm25,
        "retriever": retriever,
        "graph_storage": graph_storage,
    }
    return _infra


def _clear_query_cache() -> int:
    """Remove all query cache entries to force fresh LLM calls. Returns count."""
    if not QUERY_CACHE_DIR.exists():
        return 0
    removed = 0
    for p in sorted(QUERY_CACHE_DIR.glob("*.json")):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# ── Run survey graph for a given provider ─────────────────────────────────

def _run_survey_for_provider(
    query: str,
    provider: str,
) -> Dict[str, Any]:
    """Run the full Survey Mode pipeline with *provider*.

    Temporarily switches ``LLM_PROVIDER`` and model env vars, clears the
    query cache (to avoid cross‑provider cache pollution), runs the graph,
    then restores the original config.

    Returns:
        dict with ``elapsed_s``, ``per_theme``, ``cross_theme``, ``gap_analysis``,
        ``n_themes``, ``n_claims_per_theme``, ``anchoring_scores``.
    """
    # Save original env
    original = {
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", ""),
        "OLLAMA_SMALL_MODEL": os.environ.get("OLLAMA_SMALL_MODEL", ""),
        "OLLAMA_LARGE_MODEL": os.environ.get("OLLAMA_LARGE_MODEL", ""),
        "GAP_ANALYSIS_MODEL": os.environ.get("GAP_ANALYSIS_MODEL", ""),
    }

    try:
        os.environ["LLM_PROVIDER"] = provider

        # ── Per‑provider model config for comparison ──
        if provider == "deepseek":
            os.environ["OLLAMA_SMALL_MODEL"] = "deepseek-chat"
            os.environ["OLLAMA_LARGE_MODEL"] = "deepseek-v4-pro"
            os.environ["GAP_ANALYSIS_MODEL"] = "deepseek-v4-pro"
        else:
            # Local: use the current .env config (preserve user settings)
            # but ensure GAP_ANALYSIS_MODEL is consistent with the heavy tier
            # for a fair comparison against deepseek-v4-pro
            os.environ["GAP_ANALYSIS_MODEL"] = os.getenv("GAP_ANALYSIS_MODEL",
                                              os.getenv("OLLAMA_LARGE_MODEL", "qwen3.6:35b"))

        # Force fresh LLM calls (clear query cache to avoid cross-provider pollution)
        cleared = _clear_query_cache()
        if cleared:
            logger.info("  Cleared %d query cache entries for %s run", cleared, provider)

        infra = _get_infrastructure()
        retriever = infra["retriever"]
        graph_storage = infra["graph_storage"]

        # Build the survey graph
        app = build_survey_graph(retriever, graph_storage)

        initial_state = {
            "user_query": query,
            "query_scope": "public",
            "mode": "survey",
            "num_ctx": NUM_CTX,
            "client_kwargs": CLIENT_KWARGS,
        }

        config = {"configurable": {"thread_id": f"cmp-{provider}-{int(time.time())}"}}
        t0 = time.time()
        final_state = app.invoke(initial_state, config)
        elapsed = time.time() - t0

        # Auto-approve (skip human-in-the-loop)
        try:
            app.update_state(config, {"human_approved": True})
        except Exception:
            pass

    finally:
        # Restore original env
        for k, v in original.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    # ── Extract metrics ──────────────────────────────────────────────────
    per_theme_raw = final_state.get("per_theme_syntheses", {})
    per_theme: Dict[str, Any] = {}
    for theme_name, ts in per_theme_raw.items():
        if isinstance(ts, dict):
            syn = ts.get("synthesis", "")
            score = ts.get("anchoring_score", 0)
        else:
            syn = str(ts)
            score = 0
        per_theme[theme_name] = {
            "synthesis_len": len(syn or ""),
            "anchoring_score": round(float(score) if score else 0, 4),
            "n_claims": len(decompose_claims(scrub_unicode(syn or ""))),
        }

    cross_text = final_state.get("cross_theme_synthesis", "") or ""
    gap_text = final_state.get("gap_analysis", "") or ""
    themes_list = final_state.get("decomposed_themes", [])

    return {
        "provider": provider,
        "query": query,
        "elapsed_s": round(elapsed, 1),
        "n_themes": len(per_theme),
        "n_decomposed_themes": len(themes_list),
        "models": {
            "per_theme": os.getenv("OLLAMA_SMALL_MODEL", "unknown"),
            "cross_theme": os.getenv("OLLAMA_LARGE_MODEL", "unknown"),
            "gap": os.getenv("GAP_ANALYSIS_MODEL", "unknown"),
        },
        "per_theme": per_theme,
        "cross_theme": {
            "text_len": len(cross_text),
            "n_claims": len(decompose_claims(scrub_unicode(cross_text))),
        },
        "gap_analysis": {
            "text_len": len(gap_text),
        },
        "anchoring_scores": {
            name: ts["anchoring_score"] for name, ts in per_theme.items()
        },
    }


# ── Persistence ───────────────────────────────────────────────────────────

def _save_run(results: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("  Saved %s results to %s", results["provider"], path)


def _load_run(provider: str) -> Optional[Dict[str, Any]]:
    path = COMPARISON_DIR / f"{provider}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── Comparison logic ──────────────────────────────────────────────────────

def _compare(cloud: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    """Compute side‑by‑side metrics."""
    # Find common themes (match by name if possible)
    cloud_scores = cloud.get("anchoring_scores", {})
    local_scores = local.get("anchoring_scores", {})

    cloud_avg = sum(cloud_scores.values()) / max(len(cloud_scores), 1)
    local_avg = sum(local_scores.values()) / max(len(local_scores), 1)

    cloud_claims = sum(ts.get("n_claims", 0) for ts in cloud.get("per_theme", {}).values())
    local_claims = sum(ts.get("n_claims", 0) for ts in local.get("per_theme", {}).values())

    cloud_models = cloud.get("models", {})
    local_models = local.get("models", {})

    comparison = {
        "query": cloud.get("query", "")[:120],
        "cloud": {
            "provider": "deepseek",
            "model_tiers": (
                f"per-theme: {cloud_models.get('per_theme', '?')} | "
                f"cross-theme: {cloud_models.get('cross_theme', '?')} | "
                f"gap: {cloud_models.get('gap', '?')}"
            ),
            "n_themes": cloud.get("n_themes", 0),
            "n_decomposed_themes": cloud.get("n_decomposed_themes", 0),
            "avg_anchor_score": round(cloud_avg, 4),
            "anchoring_scores": cloud_scores,
            "total_claims": cloud_claims,
            "cross_claims": cloud.get("cross_theme", {}).get("n_claims", 0),
            "elapsed_s": cloud.get("elapsed_s", 0),
        },
        "local": {
            "provider": "ollama",
            "model_tiers": (
                f"per-theme: {local_models.get('per_theme', '?')} | "
                f"cross-theme: {local_models.get('cross_theme', '?')} | "
                f"gap: {local_models.get('gap', '?')}"
            ),
            "n_themes": local.get("n_themes", 0),
            "n_decomposed_themes": local.get("n_decomposed_themes", 0),
            "avg_anchor_score": round(local_avg, 4),
            "anchoring_scores": local_scores,
            "total_claims": local_claims,
            "cross_claims": local.get("cross_theme", {}).get("n_claims", 0),
            "elapsed_s": local.get("elapsed_s", 0),
        },
        "deltas": {
            "themes_diff": cloud.get("n_themes", 0) - local.get("n_themes", 0),
            "anchor_score_diff": round(cloud_avg - local_avg, 4),
            "claims_diff": cloud_claims - local_claims,
        },
    }
    return comparison


def _print_comparison(comparison: Dict[str, Any]) -> None:
    """Pretty‑print a single comparison."""
    cloud = comparison["cloud"]
    local = comparison["local"]
    deltas = comparison["deltas"]

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  API vs LOCAL — TRUE 1:1 SURVEY GRAPH COMPARISON{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"\n  {CYAN}Query{RESET}: {comparison['query'][:100]}")
    print(f"\n  {CYAN}Model tiering{RESET}:")
    print(f"    Cloud:  {cloud['model_tiers']}")
    print(f"    Local:  {local['model_tiers']}")
    print()

    # Header
    print(f"  {'Metric':<32} {'Cloud (DeepSeek)':<18} {'Local (Ollama)':<18} {'Delta':<14}")
    print(f"  {'-' * 32} {'-' * 18} {'-' * 18} {'-' * 14}")

    print(f"  {'Themes (synthesized)':<32} {cloud['n_themes']:<18} {local['n_themes']:<18} "
          f"{deltas['themes_diff']:>+d}")
    print(f"  {'Themes (decomposed)':<32} {cloud['n_decomposed_themes']:<18} "
          f"{local['n_decomposed_themes']:<18} "
          f"{(cloud['n_decomposed_themes'] - local['n_decomposed_themes']):>+d}")

    ds_avg = cloud["avg_anchor_score"]
    lo_avg = local["avg_anchor_score"]
    delta = deltas["anchor_score_diff"]
    color = GREEN if abs(delta) < 0.05 else (YELLOW if abs(delta) < 0.10 else RED)
    print(f"  {'Avg Anchor Score':<32} {ds_avg:<18.4f} {lo_avg:<18.4f} "
          f"{color}{delta:>+.4f}{RESET}")

    print(f"  {'Total Claims (per-theme)':<32} {cloud['total_claims']:<18} "
          f"{local['total_claims']:<18} {deltas['claims_diff']:>+d}")
    print(f"  {'Cross-theme Claims':<32} {cloud['cross_claims']:<18} "
          f"{local['cross_claims']:<18}")
    print(f"  {'Elapsed (s)':<32} {cloud['elapsed_s']:<18.0f} {local['elapsed_s']:<18.0f}")

    # Per-theme anchoring scores
    print(f"\n  {CYAN}Per‑theme anchoring scores{RESET}:")
    all_themes = sorted(set(list(cloud["anchoring_scores"].keys()) + list(local["anchoring_scores"].keys())))
    if all_themes:
        print(f"  {'Theme':<50} {'Cloud':>8}  {'Local':>8}  {'Delta':>8}")
        print(f"  {'-' * 50} {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for theme in all_themes:
            cs = cloud["anchoring_scores"].get(theme, None)
            ls = local["anchoring_scores"].get(theme, None)
            cs_str = f"{cs:.4f}" if cs is not None else "N/A"
            ls_str = f"{ls:.4f}" if ls is not None else "N/A"
            if cs is not None and ls is not None:
                d = cs - ls
                d_str = f"{d:>+.4f}"
            else:
                d_str = ""
            print(f"  {theme[:50]:<50} {cs_str:>8}  {ls_str:>8}  {d_str:>8}")

    # Summary
    print(f"\n  {CYAN}Summary{RESET}:")
    if abs(delta) < 0.05:
        print(f"  {GREEN}Local Ollama matches DeepSeek API closely "
              f"(anchor delta: {delta:+.4f}){RESET}")
    elif abs(delta) < 0.10:
        print(f"  {YELLOW}Minor quality gap (anchor delta: {delta:+.4f}) — "
              f"local models are acceptable{RESET}")
    elif abs(delta) < 0.20:
        print(f"  {YELLOW}Moderate gap (anchor delta: {delta:+.4f}) — "
              f"acceptable for exploration, use API for publication{RESET}")
    else:
        print(f"  {RED}Large gap (anchor delta: {delta:+.4f}) — "
              f"local models may need tuning{RESET}")

    latency_ratio = local["elapsed_s"] / cloud["elapsed_s"] if cloud["elapsed_s"] else 0
    if latency_ratio > 0:
        print(f"  Local is {latency_ratio:.1f}× slower than cloud "
              f"({local['elapsed_s']:.0f}s vs {cloud['elapsed_s']:.0f}s)")

    print(f"\n{BOLD}{'=' * 70}{RESET}")


# ── Main orchestration ────────────────────────────────────────────────────

def run_cloud(query: str) -> Dict[str, Any]:
    """Run DeepSeek API pipeline and persist results."""
    logger.info("=" * 60)
    logger.info("CLOUD RUN: DeepSeek API (deepseek-chat + deepseek-v4-pro)")
    logger.info("=" * 60)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key or api_key in ("your-key-here",):
        raise RuntimeError("DEEPSEEK_API_KEY not set. Add it to .env.")

    if not query:
        query = _get_default_query()

    logger.info("Query: %s...", query[:80])
    results = _run_survey_for_provider(query, "deepseek")
    _save_run(results, COMPARISON_DIR / "cloud.json")

    logger.info("Cloud run complete: %d themes, avg anchor=%.4f, %.0fs",
                results["n_themes"], sum(results["anchoring_scores"].values()) / max(len(results["anchoring_scores"]), 1),
                results["elapsed_s"])
    return results


def run_local(query: str) -> Dict[str, Any]:
    """Run local Ollama pipeline and compare against saved cloud results."""
    cloud_results = _load_run("cloud")
    if not cloud_results:
        logger.warning("No saved cloud results found — run '--run cloud' first.")
        logger.info("Will still run local for baseline metrics.")

    logger.info("=" * 60)
    logger.info("LOCAL RUN: Ollama (gemma4:e4b + qwen3.6:35b)")
    logger.info("=" * 60)

    if not query and cloud_results:
        query = cloud_results.get("query", "")

    if not query:
        query = _get_default_query()

    logger.info("Query: %s...", query[:80])
    results = _run_survey_for_provider(query, "ollama")
    _save_run(results, COMPARISON_DIR / "local.json")

    logger.info("Local run complete: %d themes, avg anchor=%.4f, %.0fs",
                results["n_themes"], sum(results["anchoring_scores"].values()) / max(len(results["anchoring_scores"]), 1),
                results["elapsed_s"])

    if cloud_results:
        comparison = _compare(cloud_results, results)
        comparison_json_path = COMPARISON_DIR / "comparison_report.json"
        comparison_json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        _print_comparison(comparison)

    return results


# ── Metrics extraction (unchanged from original) ──────────────────────────

def extract_metrics_from_synthesis(
    synthesis_text: str,
    evidence_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute anchoring score and claim metrics for a synthesis."""
    claims = decompose_claims(scrub_unicode(synthesis_text or ""))
    score, ungrounded = compute_anchoring_score(claims, evidence_chunks)
    return {
        "text": synthesis_text,
        "claims": len(claims),
        "chars": len(synthesis_text or ""),
        "anchoring_score": round(score, 4),
        "ungrounded_count": len(ungrounded),
    }


def load_evidence_chunks(project_dir: str = "projects/default") -> List[Dict[str, Any]]:
    """Load all evidence chunks from ChromaDB."""
    chroma_path = str(Path(project_dir) / "chroma_data")
    chroma = ChromaClient(collection_name="public_corpus", persist_directory=chroma_path)
    all_data = chroma.collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    chunks = []
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        if meta.get("chunk_type") == "reference":
            continue
        chunks.append({"text": scrub_unicode(str(doc)), "metadata": meta})
    return chunks


# ── Legacy query loader (kept for pytest) ─────────────────────────────────

def load_cached_queries(project_dir: str = "projects/default",
                        max_queries: int = 5) -> List[Dict[str, Any]]:
    """Load cached query decompositions from the query cache."""
    cache_dir = Path(project_dir) / "query_cache"
    if not cache_dir.exists():
        return []

    queries: Dict[str, Dict[str, Any]] = {}
    for f in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("type") != "decomposition":
            continue
        q = data.get("query", "").strip()
        if not q or len(q) < 10:
            continue
        age = data.get("_cached_at", 0)
        if q not in queries or age > queries[q].get("_cached_at", 0):
            queries[q] = {**data, "key": f.stem[:12]}

    ordered = sorted(queries.values(), key=lambda d: d.get("_cached_at", 0), reverse=True)
    return ordered[:max_queries]


def load_cross_theme_for_query(query: str, project_dir: str = "projects/default") -> Optional[Dict[str, Any]]:
    """Load the cross-theme synthesis for a specific query."""
    cache_dir = Path(project_dir) / "query_cache"
    for f in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("type") == "cross_theme" and data.get("query", "").strip() == query:
            return data
    return None


# ── Pytest integration ───────────────────────────────────────────────────

def test_comparison_infra_loads() -> None:
    """Infrastructure can be initialized without errors."""
    infra = _get_infrastructure()
    assert infra["retriever"] is not None
    assert infra["graph_storage"] is not None


def test_cloud_results_valid() -> None:
    """Saved cloud comparison results are well-formed."""
    import pytest
    cloud = _load_run("cloud")
    if cloud is None:
        pytest.skip("No cloud results saved — run '--run cloud' first")
    assert cloud["n_themes"] > 0
    assert cloud["elapsed_s"] > 0
    assert len(cloud["anchoring_scores"]) > 0
    for score in cloud["anchoring_scores"].values():
        assert 0.0 <= score <= 1.0


def test_local_results_valid() -> None:
    """Saved local comparison results are well-formed."""
    import pytest
    local = _load_run("local")
    if local is None:
        pytest.skip("No local results saved — run '--run local' first")
    assert local["n_themes"] > 0
    assert local["elapsed_s"] > 0
    assert len(local["anchoring_scores"]) > 0
    for score in local["anchoring_scores"].values():
        assert 0.0 <= score <= 1.0


def test_comparison_report_valid() -> None:
    """Comparison report is well-formed when both runs exist."""
    import pytest
    cloud = _load_run("cloud")
    local = _load_run("local")
    if cloud is None or local is None:
        pytest.skip("Both cloud and local results required")
    comparison = _compare(cloud, local)
    assert "deltas" in comparison
    assert "anchor_score_diff" in comparison["deltas"]


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 6 API vs Local Comparison — true 1:1 survey graph",
    )
    parser.add_argument("--run", choices=["cloud", "local", "both"],
                        default=None,
                        help="Which provider(s) to run (default: local only if cloud saved)")
    parser.add_argument("--query", type=str, default=None,
                        help="Research query (default: most recent from cache)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour")
    args = parser.parse_args()

    if args.no_color:
        global GREEN, YELLOW, RED, CYAN, MAGENTA, RESET, BOLD
        GREEN = YELLOW = RED = CYAN = MAGENTA = RESET = BOLD = ""

    query = args.query or ""

    if args.run == "cloud":
        try:
            run_cloud(query)
        except RuntimeError as e:
            print(f"{RED}Error: {e}{RESET}")
            sys.exit(1)

    elif args.run == "local":
        run_local(query)

    elif args.run == "both":
        print(f"\n{BOLD}Running both cloud and local comparison...{RESET}")
        print(f"  Cloud: ~5 min | Local: ~5-8 min | Total: ~10-13 min")
        print(f"  WARNING: Cloud run will consume DeepSeek API credits.\n")

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key or api_key in ("your-key-here",):
            print(f"{RED}DEEPSEEK_API_KEY not set. Add it to .env.{RESET}")
            sys.exit(1)

        # Confirm
        proceed = input(f"  Proceed with cloud + local runs? [y/N] ").strip().lower()
        if proceed not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)

        try:
            run_cloud(query)
        except RuntimeError as e:
            print(f"{RED}Cloud run failed: {e}{RESET}")
            sys.exit(1)

        run_local(query)

    else:
        # Default: check if cloud results exist, run local if missing
        cloud_exists = (COMPARISON_DIR / "cloud.json").exists()
        local_exists = (COMPARISON_DIR / "local.json").exists()

        if cloud_exists and local_exists:
            cloud = _load_run("cloud")
            local = _load_run("local")
            if cloud and local:
                comparison = _compare(cloud, local)
                _print_comparison(comparison)
                return

        print(f"Usage: python phase5_api_comparison.py --run cloud|local|both")
        print(f"  --run cloud  : Run DeepSeek API (~$1, ~5 min)")
        print(f"  --run local  : Run local Ollama + compare against saved cloud (~8 min)")
        print(f"  --run both   : Run cloud then local sequentially (~13 min)")
        if cloud_exists:
            print(f"\n  {GREEN}Cloud results exist.{RESET} Run '--run local' to compare.")
        else:
            print(f"\n  {YELLOW}No saved results.{RESET} Run '--run cloud' first.")


if __name__ == "__main__":
    main()
