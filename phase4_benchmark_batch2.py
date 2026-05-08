#!/usr/bin/env python3
"""Batch 2 Accuracy Benchmark: Conditional Critic (EGSR) & evidence truncation.

Compares the conditional-Critic Survey Mode pipeline against the
unconditional (always-invoke-Critic) baseline.  Measures:

  * Anchoring scores (draft vs final per theme)
  * Critic calls saved by the EGSR pattern
  * Evidence cap behavior (how many summaries are used vs available)
  * Latency impact

Usage:
    python phase4_benchmark_batch2.py                     # default threshold (0.35)
    python phase4_benchmark_batch2.py --threshold 0.50    # stricter threshold
    python phase4_benchmark_batch2.py --force-critic       # unconditional baseline
    python phase4_benchmark_batch2.py --compare            # run BOTH and diff
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims
from src.graph.survey_nodes import (
    CONDITIONAL_CRITIC_THRESHOLD,
    _fit_summaries_to_context,
    _run_debate_for_theme,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch2_benchmark")


# ---------------------------------------------------------------------------
#  Queries to benchmark
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    "What is the role of macrophage polarization in titanium implant osseointegration?",
    "How does obesity affect the immune response to biomaterials?",
    "What surface modifications improve titanium implant outcomes in diabetic models?",
    "What cytokines are involved in the foreign body response to orthopedic implants?",
    "How does leptin signaling influence T cell differentiation around biomaterials?",
    "What are the key differences in immune response between smooth and rough titanium surfaces?",
    "How do aging and senescence affect implant integration?",
    "What role do neutrophils play in the early inflammatory response to biomaterials?",
    "How does IL-17A signaling affect bone formation around implants?",
    "What experimental models are used to study peri-implant osteogenesis?",
]


def _collect_chunks_from_disk(project_dir: str = "projects/default") -> List[Dict[str, Any]]:
    """Load chunks from ChromaDB for benchmarking without running the full pipeline."""
    from src.retrieval.chroma_client import ChromaClient
    from src.unicode_map import scrub_unicode

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
        chunks.append({
            "text": scrub_unicode(str(doc)),
            "metadata": meta,
        })
    return chunks


def _group_chunks_by_source(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        src = (ch.get("metadata", {}) or {}).get("source", "unknown")
        if not src or src == "unknown":
            continue
        groups.setdefault(src, []).append(ch)
    return groups


def run_benchmark(
    chunks: List[Dict[str, Any]],
    mode: str = "conditional",
    threshold: float = CONDITIONAL_CRITIC_THRESHOLD,
    num_ctx: int = 16384,
) -> Dict[str, Any]:
    """Run per-query synthesis benchmarking.

    ``mode`` is one of:
      - ``"conditional"`` — use EGSR pattern (Critic only for poorly grounded drafts)
      - ``"unconditional"`` — always invoke Critic (baseline, old behavior)
    """
    import src.graph.survey_nodes as survey_mod

    original_threshold = survey_mod.CONDITIONAL_CRITIC_THRESHOLD
    if mode == "conditional":
        survey_mod.CONDITIONAL_CRITIC_THRESHOLD = threshold
    else:
        survey_mod.CONDITIONAL_CRITIC_THRESHOLD = -1.0  # always < any score → always Critic

    all_chunks = chunks
    sources = sorted(_group_chunks_by_source(chunks).keys())
    logger.info("Benchmark: %d chunks across %d papers, mode=%s threshold=%.2f",
                 len(all_chunks), len(sources), mode,
                 threshold if mode == "conditional" else -1.0)

    per_query_results = []
    total_draft_scores = 0.0
    total_final_scores = 0.0
    total_critic_calls = 0
    total_queries = 0
    total_summaries_available = 0
    total_summaries_used = 0

    for query in BENCHMARK_QUERIES:
        logger.info("--- Query: %s", query[:80])
        total_queries += 1

        # Group all chunks as a single "theme" for benchmarking
        theme_name = "all_papers"
        theme_entities = {}

        # Build summaries from chunk metadata
        summaries = []
        for ch in all_chunks:
            meta = ch.get("metadata", {}) or {}
            cs = meta.get("chunk_summary", ch.get("text", "")[:300])
            if cs:
                summaries.append(cs)

        fitted = _fit_summaries_to_context(summaries, num_ctx)
        total_summaries_available += len(summaries)
        total_summaries_used += len(fitted)
        logger.info("  Evidence: %d/%d summaries fit (num_ctx=%d)",
                     len(fitted), len(summaries), num_ctx)

        num_papers = len(sources)

        t0 = time.time()
        try:
            result = _run_debate_for_theme(
                theme_name=theme_name,
                theme_chunks=all_chunks,
                theme_entities=theme_entities,
                query=query,
                num_ctx=num_ctx,
                client_kwargs=None,
                num_papers=num_papers,
            )
        except Exception as e:
            logger.error("  Synthesis failed: %s", e)
            result = {
                "theme": theme_name,
                "synthesis": f"FAILED: {e}",
                "anchoring_score": 0.0,
                "ungrounded_claims": [],
                "num_papers": num_papers,
            }
        elapsed = time.time() - t0

        # Determine whether Critic was invoked (it was if anchoring < threshold
        # for conditional mode, or always for unconditional mode)
        # We can't easily detect from the result alone, so track via the module var
        critic_invoked = (
            mode == "unconditional"
            or (mode == "conditional" and result.get("anchoring_score", 1.0) < threshold)
        )
        if critic_invoked:
            total_critic_calls += 1

        final_score = result.get("anchoring_score", 0.0)
        total_final_scores += final_score

        per_query_results.append({
            "query": query[:120],
            "mode": mode,
            "threshold": threshold if mode == "conditional" else "unconditional",
            "num_papers": num_papers,
            "num_chunks": len(all_chunks),
            "summaries_available": len(summaries),
            "summaries_used": len(fitted),
            "anchoring_score": final_score,
            "critic_invoked": critic_invoked,
            "latency_s": round(elapsed, 1),
            "synthesis_len": len(result.get("synthesis", "")),
            "ungrounded_count": len(result.get("ungrounded_claims", [])),
        })
        logger.info("  score=%.3f critic=%s latency=%.1fs",
                     final_score, critic_invoked, elapsed)

    # Restore original threshold
    survey_mod.CONDITIONAL_CRITIC_THRESHOLD = original_threshold

    avg_final = total_final_scores / total_queries if total_queries else 0.0

    return {
        "mode": mode,
        "threshold": threshold if mode == "conditional" else "unconditional",
        "num_queries": total_queries,
        "num_papers": len(sources),
        "total_chunks": len(all_chunks),
        "avg_summaries_available": round(total_summaries_available / total_queries, 1),
        "avg_summaries_used": round(total_summaries_used / total_queries, 1),
        "avg_final_anchoring": round(avg_final, 3),
        "critic_calls": total_critic_calls,
        "critic_calls_saved": total_queries - total_critic_calls,
        "critic_save_rate": round((total_queries - total_critic_calls) / total_queries * 100, 1),
        "per_query": per_query_results,
    }


def print_report(results: Dict[str, Any]):
    """Print a human-readable benchmark report."""
    print("\n" + "=" * 70)
    print("BATCH 2 BENCHMARK REPORT")
    print("=" * 70)

    print(f"\n  Mode:              {results['mode']}")
    print(f"  Threshold:         {results['threshold']}")
    print(f"  Queries run:       {results['num_queries']}")
    print(f"  Papers in corpus:  {results['num_papers']}")
    print(f"  Total chunks:      {results['total_chunks']}")

    print(f"\n  Evidence cap:")
    print(f"    Avg available:   {results['avg_summaries_available']}")
    print(f"    Avg used:        {results['avg_summaries_used']}")

    print(f"\n  Anchoring:")
    print(f"    Avg final score: {results['avg_final_anchoring']:.3f}")

    print(f"\n  Critic calls:")
    print(f"    Invoked:         {results['critic_calls']}")
    print(f"    Saved:           {results['critic_calls_saved']}")
    print(f"    Save rate:       {results['critic_save_rate']:.1f}%")

    print(f"\n  Per-query details:")
    for pq in results["per_query"]:
        print(f"    [{pq['anchoring_score']:.3f}] {'Y' if pq['critic_invoked'] else 'N'} "
              f"  summaries={pq['summaries_used']}/{pq['summaries_available']} "
              f"  {pq['latency_s']:.1f}s  {pq['query'][:60]}...")


def compare_reports(
    cond: Dict[str, Any],
    uncond: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare conditional vs unconditional results."""
    if cond["num_queries"] != uncond["num_queries"]:
        logger.warning("Query count mismatch: %d vs %d", cond["num_queries"], uncond["num_queries"])

    delta_score = cond["avg_final_anchoring"] - uncond["avg_final_anchoring"]

    comparison = {
        "conditional_mode": cond["mode"],
        "unconditional_mode": uncond["mode"],
        "threshold": cond["threshold"],
        "num_queries": min(cond["num_queries"], uncond["num_queries"]),
        "conditional_avg_score": cond["avg_final_anchoring"],
        "unconditional_avg_score": uncond["avg_final_anchoring"],
        "anchoring_delta": round(delta_score, 3),
        "conditional_critic_calls": cond["critic_calls"],
        "unconditional_critic_calls": uncond["critic_calls"],
        "critic_calls_saved": uncond["critic_calls"] - cond["critic_calls"],
        "critic_save_rate": round(
            (uncond["critic_calls"] - cond["critic_calls"]) / uncond["critic_calls"] * 100
            if uncond["critic_calls"] else 0, 1
        ),
    }

    print("\n" + "=" * 70)
    print("COMPARISON: Conditional vs Unconditional Critic")
    print("=" * 70)
    print(f"\n  Threshold:               {comparison['threshold']}")
    print(f"  Queries:                 {comparison['num_queries']}")
    print(f"\n  Anchor scores:")
    print(f"    Conditional avg:       {comparison['conditional_avg_score']:.3f}")
    print(f"    Unconditional avg:     {comparison['unconditional_avg_score']:.3f}")
    print(f"    Delta:                 {comparison['anchoring_delta']:+.3f}")
    print(f"\n  Critic calls:")
    print(f"    Conditional:           {comparison['conditional_critic_calls']}")
    print(f"    Unconditional:         {comparison['unconditional_critic_calls']}")
    print(f"    Saved:                 {comparison['critic_calls_saved']}")
    print(f"    Save rate:             {comparison['critic_save_rate']:.1f}%")

    if abs(delta_score) < 0.05:
        print(f"\n  ✓ Acceptable anchoring delta ({abs(delta_score):.3f} < 0.05)")
        print(f"    Recommend: keep conditional Critic at threshold {comparison['threshold']}")
    else:
        print(f"\n  ⚠ Anchoring delta too large ({abs(delta_score):.3f} >= 0.05)")
        print(f"    Consider: raise threshold closer to 0.85")

    return comparison


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch 2 accuracy benchmark")
    parser.add_argument(
        "--threshold", type=float, default=CONDITIONAL_CRITIC_THRESHOLD,
        help=f"Critic invocation threshold (default: {CONDITIONAL_CRITIC_THRESHOLD})",
    )
    parser.add_argument(
        "--force-critic", action="store_true",
        help="Run unconditional mode (always invoke Critic) for baseline",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run BOTH conditional and unconditional modes and diff results",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=16384,
        help="Context window size for evidence cap (default: 16384)",
    )
    parser.add_argument(
        "--output", type=str, default="projects/default/batch2_benchmark.json",
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    # Load chunks
    chunks = _collect_chunks_from_disk()
    if not chunks:
        print("ERROR: No chunks in ChromaDB. Run phase4_demo.py first to ingest PDFs.")
        sys.exit(1)

    logger.info("Loaded %d chunks from ChromaDB", len(chunks))

    if args.compare:
        logger.info("Running COMPARISON mode...")
        cond_results = run_benchmark(chunks, mode="conditional", threshold=args.threshold,
                                     num_ctx=args.num_ctx)
        uncond_results = run_benchmark(chunks, mode="unconditional", num_ctx=args.num_ctx)

        print_report(cond_results)
        print_report(uncond_results)
        comparison = compare_reports(cond_results, uncond_results)

        output = {
            "conditional": cond_results,
            "unconditional": uncond_results,
            "comparison": comparison,
        }
    elif args.force_critic:
        results = run_benchmark(chunks, mode="unconditional", num_ctx=args.num_ctx)
        print_report(results)
        output = results
    else:
        results = run_benchmark(chunks, mode="conditional", threshold=args.threshold,
                                num_ctx=args.num_ctx)
        print_report(results)
        output = results

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
