#!/usr/bin/env python3
"""
Phase 9: Europe PMC + Semantic Scholar Pipeline Test

Comparison against the old Playwright/EZProxy PDF download pipeline
(typically 45-90 seconds per paper).

Usage:
    python phase9_europe_pmc_test.py          # Test with 10 papers
    python phase9_europe_pmc_test.py --count 50  # Test with 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv; load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase9_test")

SEARCH_QUERY = "titanium implant macrophage polarization osseointegration surface modification"


def test_pipeline(count: int = 10):
    from src.retrieval.europe_pmc import EuropePMCClient
    from src.ingestion.pmc_xml_parser import PMCXMLParser
    from src.retrieval.semantic_scholar import SemanticScholarClient

    epmc = EuropePMCClient()
    parser = PMCXMLParser()
    s2 = SemanticScholarClient()

    results = {
        "pipeline": "Europe PMC + Semantic Scholar",
        "query": SEARCH_QUERY,
        "target_count": count,
    }

    # ── Phase 1: Search Europe PMC ──
    logger.info("─ Phase 1: Searching Europe PMC (OA only)...")
    t0 = time.monotonic()
    papers = epmc.search(SEARCH_QUERY, oa_only=True, max_results=count)
    t_search = time.monotonic() - t0
    logger.info("  Found %d OA papers in %.2fs", len(papers), t_search)
    results["search_time_s"] = round(t_search, 3)
    results["papers_found"] = len(papers)

    if not papers:
        logger.warning("No OA papers found!")
        return

    # Print paper list
    for i, p in enumerate(papers):
        logger.info("  [%d] %s (%s) — %s", i + 1, p["title"][:70],
                     p.get("journal", "")[:30], p.get("pmcid", ""))

    # ── Phase 2: Fetch full-text XML ──
    logger.info("\n─ Phase 2: Fetching full-text XML...")
    t0 = time.monotonic()
    pmcids = [p["pmcid"] for p in papers if p.get("pmcid")]
    logger.info("  Fetching %d full-text XMLs...", len(pmcids))
    xml_docs = epmc.full_text_xml_batch(pmcids)
    t_fetch = time.monotonic() - t0
    results["fetch_time_s"] = round(t_fetch, 3)

    fetched = sum(1 for v in xml_docs.values() if v)
    empty = len(pmcids) - fetched
    logger.info("  Fetched %d XMLs (%d empty) in %.2fs", fetched, empty, t_fetch)

    # ── Phase 3: Parse XML into chunks ──
    logger.info("\n─ Phase 3: Parsing XML into chunks...")
    t0 = time.monotonic()
    all_chunks: list = []
    paper_stats: list = []

    for p in papers:
        pmcid = p.get("pmcid", "")
        if not pmcid:
            paper_stats.append({**p, "chunks": 0, "words": 0, "xml": False})
            continue

        xml = xml_docs.get(pmcid)
        if not xml:
            paper_stats.append({**p, "chunks": 0, "words": 0, "xml": False})
            continue

        chunks = parser.parse(xml, pmcid=pmcid, doi=p.get("doi", ""))
        total_words = sum(len(c["text"].split()) for c in chunks)

        all_chunks.extend(chunks)
        paper_stats.append({
            "title": p["title"][:80],
            "pmcid": pmcid,
            "doi": p.get("doi", ""),
            "chunks": len(chunks),
            "words": total_words,
            "sections": len([c for c in chunks if c["metadata"]["chunk_type"] == "text"]),
            "figures": len([c for c in chunks if c["metadata"]["chunk_type"] == "figure"]),
            "xml": True,
        })

    t_parse = time.monotonic() - t0
    results["parse_time_s"] = round(t_parse, 3)
    results["total_chunks"] = len(all_chunks)
    results["total_words"] = sum(s["words"] for s in paper_stats)

    # ── Phase 4: SPECTER2 embeddings (Semantic Scholar) ──
    logger.info("\n─ Phase 4: Fetching SPECTER2 embeddings (Semantic Scholar)...")
    t0 = time.monotonic()

    # Resolve papers via DOI → title fallback
    s2_ids = []
    for p in papers:
        if p.get("doi"):
            s2_paper = s2.resolve_paper(p["doi"], p.get("title", ""))
            if s2_paper and s2_paper.get("paper_id"):
                s2_ids.append(s2_paper["paper_id"])
                p["s2_paper_id"] = s2_paper["paper_id"]
                p["s2_tldr"] = s2_paper.get("tldr", "")

    # Batch fetch embeddings
    if s2_ids:
        embeddings = s2.get_embeddings_batch(s2_ids)
        for p in papers:
            pid = p.get("s2_paper_id", "")
            if pid and pid in embeddings:
                p["s2_embedding"] = embeddings[pid]

    t_s2 = time.monotonic() - t0
    results["s2_time_s"] = round(t_s2, 3)
    results["s2_papers_matched"] = len(s2_ids)
    results["s2_embeddings_fetched"] = sum(1 for p in papers if p.get("s2_embedding"))

    logger.info("  Matched %d papers, %d with embeddings in %.2fs",
                 len(s2_ids), results["s2_embeddings_fetched"], t_s2)

    # ── Summary ──
    total_time = t_search + t_fetch + t_parse + t_s2
    results["total_time_s"] = round(total_time, 3)
    results["paper_details"] = paper_stats

    # Comparison with old pipeline
    old_pipeline_per_paper_s = 60  # ~60s per paper typical
    old_pipeline_total_s = count * old_pipeline_per_paper_s
    speedup = old_pipeline_total_s / max(total_time, 0.001)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Papers searched:    %d", len(papers))
    logger.info("  Full-text fetched:  %d/%d", fetched, len(pmcids))
    logger.info("  Total chunks:       %d", len(all_chunks))
    logger.info("  Total words:        %d", results["total_words"])
    logger.info("  SPECTER2 embeddings: %d", results["s2_embeddings_fetched"])
    logger.info("")
    logger.info("  Search time:   %6.2fs", t_search)
    logger.info("  Fetch time:    %6.2fs", t_fetch)
    logger.info("  Parse time:    %6.2fs", t_parse)
    logger.info("  S2 time:       %6.2fs", t_s2)
    logger.info("  ─────────────────────")
    logger.info("  TOTAL:         %6.2fs", total_time)
    logger.info("")
    logger.info("  Old pipeline:  %6ds  (est. %ds/paper × %d)",
                 old_pipeline_total_s, old_pipeline_per_paper_s, count)
    logger.info("  Speedup:       %6.1f×", speedup)
    logger.info("")

    # Per-paper details
    logger.info("Per-paper breakdown:")
    for i, ps in enumerate(paper_stats):
        if ps.get("xml"):
            logger.info("  [%d] %s", i + 1, ps["title"][:60])
            logger.info("       chunks=%d  words=%d  sections=%d  figures=%d",
                         ps["chunks"], ps["words"], ps["sections"], ps["figures"])
        else:
            logger.info("  [%d] %s — NO XML", i + 1, ps["title"][:60])

    # Save results
    cache_path = Path("projects/default/phase9_europe_pmc_test.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("\nResults saved to %s", cache_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Europe PMC pipeline test")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of papers to test (default: 10)")
    parser.add_argument("--query", type=str, default=SEARCH_QUERY,
                        help="Search query")
    args = parser.parse_args()

    test_pipeline(count=args.count)


if __name__ == "__main__":
    main()
