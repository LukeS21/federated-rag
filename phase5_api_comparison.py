#!/usr/bin/env python3
"""Phase 5 → 6 API vs Local Comparison.

Compares synthesis quality between DeepSeek v4-pro API and local Ollama
models on 3-5 cached queries.  Measures anchoring scores, claim counts,
entity appearance rate, and latency.

Two modes:
  1. Cached comparison: Compare two cached local runs (default, no API cost)
  2. Live API comparison: Re-run cached queries through DeepSeek API (--live)

Usage:
    python phase5_api_comparison.py                    # Compare cached local results
    python phase5_api_comparison.py --live             # Live DeepSeek API comparison
    python phase5_api_comparison.py --queries 3        # Number of queries to compare
    python -m pytest phase5_api_comparison.py -v       # Regression guard
"""

from __future__ import annotations

import json
import logging
import os
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

logger = logging.getLogger("api_comparison")

# ── ANSI ───────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Query loader ──────────────────────────────────────────────────────────

def load_cached_queries(project_dir: str = "projects/default",
                        max_queries: int = 5) -> List[Dict[str, Any]]:
    """Load cached query decompositions from the query cache.

    Returns list of dicts with {query, themes, cached_at, key}.
    """
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
        # Keep only the most recent version of each query text
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


# ── Metric extraction from a synthesis ─────────────────────────────────────

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


# ── Live API comparison ────────────────────────────────────────────────────

def run_deepseek_query(
    query: str,
    provider: str = "deepseek",
) -> Dict[str, Any]:
    """Run a survey query through DeepSeek API and return the synthesis.

    Temporarily switches LLM_PROVIDER to deepseek, runs the full pipeline,
    then restores the original provider.
    """
    original_provider = os.environ.pop("LLM_PROVIDER", "ollama")
    os.environ["LLM_PROVIDER"] = provider

    try:
        from src.agents.query_decomposer import QueryDecomposer
        from src.agents.synthesis_drafter import SynthesisDrafter

        # Step 1: Decompose query
        t0 = time.time()
        decomposer = QueryDecomposer()
        result = decomposer.decompose(query)
        themes = result.get("themes", [])
        decomp_latency = time.time() - t0
        logger.info("  Decomposed into %d themes (%.1fs)", len(themes), decomp_latency)

        # Step 2: Build summaries from evidence chunks
        evidence = load_evidence_chunks()
        summaries: List[str] = []
        for ch in evidence:
            meta = ch.get("metadata", {}) or {}
            cs = meta.get("chunk_summary", ch.get("text", "")[:300])
            if cs:
                summaries.append(cs)
        summary_text = "\n\n".join(summaries[:20])  # Cap to avoid context overflow
        summary_chunks = [{"text": summary_text, "metadata": {"source": "all_papers"}}]

        citations = sorted({(ch.get("metadata", {}) or {}).get("cite_key") or
                             (ch.get("metadata", {}) or {}).get("source", "unknown")
                             for ch in evidence})

        # Step 3: Per-theme synthesis (single-pass for comparison)
        t1 = time.time()
        drafter = SynthesisDrafter(model="deepseek-v4-pro")
        per_theme_results: Dict[str, Dict[str, Any]] = {}
        for theme in themes[:3]:  # Cap at 3 themes for cost
            theme_name = theme.get("theme", "theme")
            t2 = time.time()
            draft = drafter.draft(
                query=f"{query} [Theme: {theme_name}]",
                entities={},
                chunks=summary_chunks,
                citations=citations,
                kg_context={},
            )
            theme_latency = time.time() - t2
            metrics = extract_metrics_from_synthesis(draft, evidence)
            metrics["latency"] = round(theme_latency, 1)
            per_theme_results[theme_name] = metrics

        per_theme_latency = time.time() - t1

        # Step 4: Cross-theme synthesis
        t3 = time.time()
        combined = "\n\n".join(
            f"## {name}\n{m['text']}"
            for name, m in per_theme_results.items()
        )
        cross_chunks = [{"text": combined, "metadata": {"source": "cross_theme"}}]
        cross_drafter = SynthesisDrafter(model="deepseek-v4-pro")
        cross_synth = cross_drafter.draft(
            query=f"{query}\n\nSynthesize into unified narrative. Include inline citations. Output plain text.",
            entities={},
            chunks=cross_chunks,
            citations=citations,
            kg_context={},
        )
        cross_latency = time.time() - t3
        cross_metrics = extract_metrics_from_synthesis(cross_synth, evidence)
        cross_metrics["latency"] = round(cross_latency, 1)

        return {
            "query": query,
            "provider": provider,
            "themes": len(themes),
            "decomp_latency": round(decomp_latency, 1),
            "per_theme_latency": round(per_theme_latency, 1),
            "cross_theme_latency": round(cross_latency, 1),
            "total_latency": round(decomp_latency + per_theme_latency + cross_latency, 1),
            "per_theme": {
                name: {
                    "claims": m["claims"],
                    "chars": m["chars"],
                    "anchoring_score": m["anchoring_score"],
                    "latency": m["latency"],
                }
                for name, m in per_theme_results.items()
            },
            "cross_theme": cross_metrics,
        }
    finally:
        os.environ["LLM_PROVIDER"] = original_provider


# ── Comparison logic ───────────────────────────────────────────────────────

def compare_results(
    local_results: Dict[str, Any],
    deepseek_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare local Ollama results against DeepSeek API results."""
    # Extract local metrics
    local_theme_scores = []
    local_claims = 0
    for ts in local_results.get("per_theme_syntheses", {}).values():
        local_theme_scores.append(ts.get("anchoring_score", 0))
        local_claims += len(decompose_claims(scrub_unicode(ts.get("synthesis", "") or "")))

    local_cross_score = 0.0
    if local_results.get("cross_theme_synthesis"):
        local_cross_score = extract_metrics_from_synthesis(
            local_results["cross_theme_synthesis"],
            load_evidence_chunks(),
        )["anchoring_score"]

    comparison = {
        "query": local_results.get("user_query", "unknown")[:120],
        "local": {
            "themes": len(local_results.get("per_theme_syntheses", {})),
            "avg_anchor_score": round(sum(local_theme_scores) / max(len(local_theme_scores), 1), 4),
            "cross_anchor_score": round(local_cross_score, 4),
            "total_claims": local_claims,
        },
    }

    if deepseek_results:
        ds_theme_scores = [m["anchoring_score"] for m in deepseek_results.get("per_theme", {}).values()]
        comparison["deepseek"] = {
            "themes": deepseek_results.get("themes", 0),
            "avg_anchor_score": round(sum(ds_theme_scores) / max(len(ds_theme_scores), 1), 4),
            "cross_anchor_score": deepseek_results.get("cross_theme", {}).get("anchoring_score", 0),
            "total_claims": sum(m["claims"] for m in deepseek_results.get("per_theme", {}).values()),
            "total_latency": deepseek_results.get("total_latency", 0),
        }

        # Deltas
        local_avg = comparison["local"]["avg_anchor_score"]
        ds_avg = comparison["deepseek"]["avg_anchor_score"]
        comparison["deltas"] = {
            "anchor_score_diff": round(ds_avg - local_avg, 4),
            "cross_anchor_diff": round(
                comparison["deepseek"]["cross_anchor_score"] - comparison["local"]["cross_anchor_score"], 4
            ),
            "claims_diff": comparison["deepseek"]["total_claims"] - comparison["local"]["total_claims"],
        }

    return comparison


# ── Report printing ────────────────────────────────────────────────────────

def print_comparison_report(comparisons: List[Dict[str, Any]]) -> None:
    """Print a side-by-side comparison report."""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  API vs LOCAL COMPARISON REPORT{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    for i, comp in enumerate(comparisons):
        local = comp.get("local", {})
        deepseek = comp.get("deepseek", {})
        deltas = comp.get("deltas", {})

        print(f"\n  {CYAN}Query {i + 1}{RESET}: {comp.get('query', '?')[:80]}")

        print(f"    {'Metric':<30} {'Local (Ollama)':<20} {'DeepSeek API':<20} {'Delta':<15}")
        print(f"    {'-' * 30} {'-' * 20} {'-' * 20} {'-' * 15}")
        ds_themes = str(deepseek.get('themes',0)) if deepseek else "N/A"
        print(f"    {'Themes':<30} {local.get('themes',0):<20} {ds_themes:<20}")
        ds_avg = f"{deepseek.get('avg_anchor_score',0):.4f}" if deepseek else "N/A"
        ds_cross = f"{deepseek.get('cross_anchor_score',0):.4f}" if deepseek else "N/A"
        ds_claims = str(deepseek.get('total_claims',0)) if deepseek else "N/A"
        print(f"    {'Avg Anchor Score':<30} {local.get('avg_anchor_score',0):<20.4f} "
              f"{ds_avg:<20}")
        print(f"    {'Cross Anchor Score':<30} {local.get('cross_anchor_score',0):<20.4f} "
              f"{ds_cross:<20}")
        print(f"    {'Total Claims':<30} {local.get('total_claims',0):<20} "
              f"{ds_claims:<20}")

        if deltas:
            delta_sign = "+" if deltas.get("anchor_score_diff", 0) > 0 else ""
            color = GREEN if abs(deltas.get("anchor_score_diff", 0)) < 0.10 else (
                YELLOW if abs(deltas.get("anchor_score_diff", 0)) < 0.20 else RED
            )
            print(f"    {'Anchor Score Delta':<30} {'':<20} {'':<20} "
                  f"{color}{delta_sign}{deltas.get('anchor_score_diff',0):.4f}{RESET}")

            if deepseek and deepseek.get("total_latency"):
                print(f"    {'Total Latency':<30} {'N/A':<20} "
                      f"{deepseek.get('total_latency',0):.1f}s{'':<10}")

    print(f"\n{BOLD}{'=' * 70}{RESET}")

    # Summary
    if any(c.get("deltas") for c in comparisons):
        avg_delta = sum(
            abs(c.get("deltas", {}).get("anchor_score_diff", 0))
            for c in comparisons
        ) / max(len(comparisons), 1)

        if avg_delta < 0.05:
            print(f"  {GREEN}Local Ollama matches DeepSeek API quality closely "
                  f"(avg anchor delta: {avg_delta:.4f}){RESET}")
        elif avg_delta < 0.15:
            print(f"  {YELLOW}Moderate quality gap between local and API "
                  f"(avg anchor delta: {avg_delta:.4f}){RESET}")
        else:
            print(f"  {RED}Significant quality gap — local models may need tuning "
                  f"(avg anchor delta: {avg_delta:.4f}){RESET}")

    print(f"{BOLD}{'=' * 70}{RESET}\n")


# ── Pytest integration ─────────────────────────────────────────────────────

def test_api_local_comparison() -> None:
    """Pytest test: verifies that cached local results contain valid syntheses.

    Does NOT make live API calls.  Verifies the local results are well-formed.
    """
    queries = load_cached_queries(max_queries=1)
    if not queries:
        raise AssertionError(
            "No cached queries found. Run phase4_demo.py first."
        )

    query = queries[0]["query"]
    cross_data = load_cross_theme_for_query(query)

    if not cross_data:
        raise AssertionError(
            f"No cross-theme synthesis cached for query: {query[:60]}..."
        )

    cross_synth = cross_data.get("cross_theme_synthesis", "")
    assert len(cross_synth) > 100, f"Cross-theme synthesis too short: {len(cross_synth)} chars"

    gap = cross_data.get("gap_analysis", "")
    assert len(gap) > 50, f"Gap analysis too short: {len(gap)} chars"


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5 → 6 API vs Local Comparison",
    )
    parser.add_argument("--live", action="store_true",
                        help="Run live queries through DeepSeek API (costs API credits)")
    parser.add_argument("--queries", type=int, default=3,
                        help="Number of cached queries to compare (default: 3)")
    parser.add_argument("--project-dir", default="projects/default",
                        help="Project directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Save comparison JSON to path")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour")
    args = parser.parse_args()

    if args.no_color:
        global GREEN, YELLOW, RED, CYAN, MAGENTA, RESET, BOLD
        GREEN = YELLOW = RED = CYAN = MAGENTA = RESET = BOLD = ""

    # Load cached local queries
    cached = load_cached_queries(args.project_dir, max_queries=args.queries)
    if not cached:
        print(f"{RED}No cached queries found.{RESET}")
        print("Run 'python phase4_demo.py' first to generate cached syntheses.")
        sys.exit(1)

    print(f"Found {len(cached)} cached queries for comparison.")

    if args.live:
        # Check DeepSeek API key
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key or api_key == "your-key-here":
            print(f"{RED}DEEPSEEK_API_KEY not set. Set in .env to use live mode.{RESET}")
            sys.exit(1)

        print(f"\n{BOLD}Running live DeepSeek API comparison...{RESET}")
        print(f"  WARNING: This will consume API credits and send data to cloud.")
        print(f"  Queries to run: {len(cached)}")
        proceed = input("  Proceed? [y/N] ").strip().lower()
        if proceed not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)

        comparisons = []
        for i, c in enumerate(cached):
            query = c["query"]
            print(f"\n[{i + 1}/{len(cached)}] Query: {query[:80]}...")

            # Load cached local result
            cross_data = load_cross_theme_for_query(query, args.project_dir)
            if not cross_data:
                print(f"  {YELLOW}No cached cross-theme for this query — skipping.{RESET}")
                continue

            # Extract local metrics from cached
            local_synth = cross_data.get("cross_theme_synthesis", "")
            local_gap = cross_data.get("gap_analysis", "")
            evidence = load_evidence_chunks(args.project_dir)
            local_metrics = extract_metrics_from_synthesis(local_synth, evidence)

            local_results = {
                "user_query": query,
                "per_theme_syntheses": {
                    "combined": {
                        "synthesis": local_synth,
                        "anchoring_score": local_metrics["anchoring_score"],
                    }
                },
                "cross_theme_synthesis": local_synth,
                "gap_analysis": local_gap,
            }

            # Run DeepSeek API comparison
            print(f"  Calling DeepSeek API...")
            try:
                ds_results = run_deepseek_query(query, provider="deepseek")
            except Exception as e:
                print(f"  {RED}DeepSeek API call failed: {e}{RESET}")
                ds_results = None

            comparisons.append(compare_results(local_results, ds_results))
    else:
        # Cached-only mode: just analyze local results
        print(f"\n{BOLD}Cached-only comparison (no live API calls):{RESET}")
        comparisons = []
        evidence = load_evidence_chunks(args.project_dir)
        for i, c in enumerate(cached):
            query = c["query"]
            cross_data = load_cross_theme_for_query(query, args.project_dir)
            if not cross_data:
                continue
            local_synth = cross_data.get("cross_theme_synthesis", "")
            local_gap = cross_data.get("gap_analysis", "")
            local_metrics = extract_metrics_from_synthesis(local_synth, evidence)

            local_results = {
                "user_query": query,
                "per_theme_syntheses": {
                    "combined": {
                        "synthesis": local_synth,
                        "anchoring_score": local_metrics["anchoring_score"],
                    }
                },
                "cross_theme_synthesis": local_synth,
                "gap_analysis": local_gap,
            }
            comparisons.append(compare_results(local_results, None))

    if comparisons:
        print_comparison_report(comparisons)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(comparisons, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"Saved to {out_path}")
    else:
        print(f"\n{YELLOW}No comparisons produced — check cached data.{RESET}")


if __name__ == "__main__":
    main()
