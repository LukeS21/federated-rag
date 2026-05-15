#!/usr/bin/env python3
"""
Phase 9: Europe PMC + Semantic Scholar Pipeline Test

Comparison against the old Playwright/EZProxy PDF download pipeline
(typically 45-90 seconds per paper).

Usage:
    python phase9_europe_pmc_test.py               # Test with 10 papers
    python phase9_europe_pmc_test.py --count 50    # Test with 50
    python phase9_europe_pmc_test.py --ingest      # Ingest into ChromaDB + BM25 + KG
    python phase9_europe_pmc_test.py --coverage    # Run coverage diagnostic
    python phase9_europe_pmc_test.py --query "dental implant macrophage"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase9_test")

SEARCH_QUERY = "titanium implant macrophage polarization osseointegration surface modification"
PROJECT_DIR = Path("projects/default")
CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
BM25_CORPUS_PATH = str(PROJECT_DIR / "bm25_corpus.json")


def _phase4_fetch_embeddings(papers, s2, cache):
    """Phase 4: Fetch SPECTER2 embeddings with local caching.
    
    Returns dict with s2_time_s, s2_papers_matched, s2_embeddings_fetched,
    s2_cache_hits.
    """
    s2_ids = []
    s2_cache_hits = 0

    for p in papers:
        doi = p.get("doi", "")
        if not doi:
            continue

        # Check SPECTER2 cache first
        cached_emb = cache.get(doi)
        if cached_emb is not None:
            p["s2_embedding"] = cached_emb
            p["s2_cached"] = True
            s2_cache_hits += 1
            continue

        # Resolve via DOI → title fallback
        s2_paper = s2.resolve_paper(doi, p.get("title", ""))
        if s2_paper and s2_paper.get("paper_id"):
            s2_ids.append(s2_paper["paper_id"])
            p["s2_paper_id"] = s2_paper["paper_id"]
            p["s2_tldr"] = s2_paper.get("tldr", "")
            p["s2_cached"] = False

    # Batch fetch embeddings for non-cached papers
    if s2_ids:
        embeddings = s2.get_embeddings_batch(s2_ids)
        for p in papers:
            pid = p.get("s2_paper_id", "")
            if pid and pid in embeddings:
                emb = embeddings[pid]
                p["s2_embedding"] = emb
                # Store in cache
                doi = p.get("doi", "")
                if doi and emb:
                    cache.put(doi, pid, emb)

    # Flush cache to disk
    if s2_ids:
        cache.flush()

    return {
        "s2_papers_matched": len(s2_ids) + s2_cache_hits,
        "s2_embeddings_fetched": sum(
            1 for p in papers if p.get("s2_embedding")
        ),
        "s2_cache_hits": s2_cache_hits,
    }


def _phase5_ingest(papers, xml_docs, parser, results, retriever=None, chroma=None,
                   bm25=None, graph_storage=None, ingest_figures=False):
    """Phase 5: Ingest into ChromaDB + BM25 + KG + Figures.

    Returns (t_ingest, ingested, skipped, figure_counts).
    """
    from src.utils.ingest_progress import IngestProgress
    from src.anchoring.evidence_check import set_anchoring_chroma

    progress = IngestProgress()

    if retriever is None:
        from src.retrieval.chroma_client import ChromaClient
        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.hybrid_retriever import HybridRetriever

        chroma = ChromaClient(
            collection_name="public_corpus", persist_directory=CHROMA_PATH,
        )
        bm25 = BM25Index(persist_dir=PROJECT_DIR / "bm25_index")
        set_anchoring_chroma(chroma)

        if not bm25.load():
            logger.debug("  No existing BM25 corpus — starting fresh")

        retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)
    else:
        set_anchoring_chroma(retriever.chroma)

    ingested = 0
    skipped = 0
    all_figure_counts = {"found": 0, "downloaded": 0, "described": 0,
                         "embedded": 0, "failed": 0}

    for p in papers:
        pmcid = p.get("pmcid", "")
        if not pmcid:
            continue
        if progress.is_completed(pmcid):
            skipped += 1
            logger.debug("  Skipping %s (already ingested)", pmcid)
            continue
        xml = xml_docs.get(pmcid)
        if not xml:
            continue
        chunks = parser.parse(xml, pmcid=pmcid, doi=p.get("doi", ""))
        if not chunks:
            continue

        # 5a: ChromaDB + BM25
        retriever.ingest(chunks)

        # 5b: Figure pipeline (download XML <graphic> images → vision_ingest)
        if ingest_figures:
            try:
                from src.vision.vision_ingest import vision_ingest_xml_figures
                fc = vision_ingest_xml_figures(
                    chunks, retriever, describe=False,
                )
                all_figure_counts["found"] += fc["found"]
                all_figure_counts["downloaded"] += fc["downloaded"]
                all_figure_counts["described"] += fc["described"]
                all_figure_counts["embedded"] += fc["embedded"]
                all_figure_counts["failed"] += fc["failed"]
                if fc["found"] > 0:
                    logger.debug("  Figures: %d found, %d downloaded, %d embedded",
                                 fc["found"], fc["downloaded"], fc["embedded"])
            except Exception as e:
                logger.debug("  Figure ingest for %s: %s", pmcid, e)

        # 5c: PreExtractor — run KG updates at ingest time
        if graph_storage is not None:
            try:
                from src.ingestion.pre_extractor import PreExtractor
                pre = PreExtractor()
                pre.extract_paper(
                    paper_id=pmcid, chunks=chunks,
                    graph_storage=graph_storage,
                )
                logger.debug("  PreExtractor + KG updated for %s", pmcid)
            except Exception as e:
                logger.debug("  PreExtractor failed for %s: %s", pmcid, e)

        progress.checkpoint(pmcid)
        results["ingested_pmcids"].append(pmcid)
        ingested += 1
        logger.debug("  Ingested %s: %d chunks", pmcid, len(chunks))

    progress.finalize()
    if bm25:
        bm25.save()
    if graph_storage:
        try:
            graph_storage.save()
            logger.debug("  Knowledge graph saved")
        except Exception as e:
            logger.debug("  KG save failed: %s", e)

    return ingested, skipped, all_figure_counts


def test_pipeline(
    count: int = 10,
    ingest: bool = False,
    query: str = "",
    coverage: bool = False,
    ingest_figures: bool = False,
    use_graph: bool = False,
):
    from src.retrieval.europe_pmc import EuropePMCClient
    from src.ingestion.pmc_xml_parser import PMCXMLParser
    from src.retrieval.semantic_scholar import SemanticScholarClient
    from src.utils.spector2_cache import Spector2Cache

    query = query or SEARCH_QUERY
    epmc = EuropePMCClient()
    parser = PMCXMLParser()
    s2 = SemanticScholarClient()
    spector2_cache = Spector2Cache()

    results = {
        "pipeline": "Europe PMC + Semantic Scholar",
        "query": query,
        "target_count": count,
    }

    # ── Phase 1: Search Europe PMC ──
    logger.info("─ Phase 1: Searching Europe PMC (OA only)...")
    t0 = time.monotonic()
    papers = epmc.search(query, oa_only=True, max_results=count)
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

    # ── Phase 4: SPECTER2 embeddings with caching ──
    logger.info("\n─ Phase 4: Fetching SPECTER2 embeddings (Semantic Scholar + cache)...")
    t0 = time.monotonic()

    s2_info = _phase4_fetch_embeddings(papers, s2, spector2_cache)

    t_s2 = time.monotonic() - t0
    results["s2_time_s"] = round(t_s2, 3)
    results["s2_papers_matched"] = s2_info["s2_papers_matched"]
    results["s2_embeddings_fetched"] = s2_info["s2_embeddings_fetched"]
    results["s2_cache_hits"] = s2_info["s2_cache_hits"]

    logger.info("  Matched %d papers, %d with embeddings, %d cache hits in %.2fs",
                 s2_info["s2_papers_matched"],
                 s2_info["s2_embeddings_fetched"],
                 s2_info["s2_cache_hits"],
                 t_s2)

    # ── Phase 4b: Coverage diagnostic (optional) ──
    results["coverage_diagnostic"] = {}
    if coverage:
        logger.info("\n─ Phase 4b: Coverage diagnostic (PMC vs Semantic Scholar)...")
        try:
            from src.retrieval.coverage import run_coverage_diagnostic
            cov = run_coverage_diagnostic(query, max_results=min(count, 20))
            results["coverage_diagnostic"] = cov
            logger.info(
                "  Coverage: %d/%d S2 papers (%.1f%%) have PMC full text",
                cov["matched"], cov["s2_total"], cov["coverage_pct"],
            )
        except Exception as e:
            logger.warning("  Coverage diagnostic failed: %s", e)

    # ── Phase 5: Ingest into ChromaDB + BM25 + KG (optional) ──
    t_ingest = 0.0
    results["ingested_pmcids"] = []
    results["figure_counts"] = {}
    results["kg_updated"] = False

    if ingest:
        logger.info("\n─ Phase 5: Ingesting into ChromaDB + BM25" +
                     (" + KG" if use_graph else "") +
                     (" + Figures" if ingest_figures else "") + "...")

        t0 = time.monotonic()

        # Set up graph storage if requested
        graph_storage = None
        if use_graph:
            from src.graph import create_graph_storage
            graph_storage = create_graph_storage(
                file_path=PROJECT_DIR / "project_graph.json",
            )
            graph_storage.load()
            results["kg_updated"] = True

        ingested, skipped, figure_counts = _phase5_ingest(
            papers, xml_docs, parser, results,
            graph_storage=graph_storage,
            ingest_figures=ingest_figures,
        )

        t_ingest = time.monotonic() - t0
        results["ingest_time_s"] = round(t_ingest, 3)
        results["papers_ingested"] = ingested
        results["figure_counts"] = figure_counts

        logger.info("  Ingested %d papers in %.2fs (%d skipped, %d total)",
                     ingested, t_ingest, skipped, len(papers))
        if figure_counts.get("found", 0) > 0:
            logger.info("  Figures: %d found, %d downloaded, %d embedded, %d failed",
                         figure_counts["found"], figure_counts["downloaded"],
                         figure_counts["embedded"], figure_counts["failed"])

    # ── Summary ──
    total_time = t_search + t_fetch + t_parse + t_s2 + t_ingest
    results["total_time_s"] = round(total_time, 3)
    results["paper_details"] = paper_stats

    # Comparison with old pipeline
    old_pipeline_per_paper_s = 60
    old_pipeline_total_s = count * old_pipeline_per_paper_s
    speedup = old_pipeline_total_s / max(total_time, 0.001)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Papers searched:     %d", len(papers))
    logger.info("  Full-text fetched:   %d/%d", fetched, len(pmcids))
    logger.info("  Total chunks:        %d", len(all_chunks))
    logger.info("  Total words:         %d", results["total_words"])
    logger.info("  SPECTER2 embeddings: %d (%d cache hits)",
                 results["s2_embeddings_fetched"],
                 results.get("s2_cache_hits", 0))
    if coverage and results["coverage_diagnostic"]:
        cov = results["coverage_diagnostic"]
        logger.info("  PMC coverage:        %d/%d (%.1f%%)",
                     cov["matched"], cov["s2_total"], cov["coverage_pct"])
    if ingest:
        logger.info("  Papers ingested:     %d", results.get("papers_ingested", 0))
        if results.get("kg_updated"):
            logger.info("  KG updated:          yes")
        fc = results.get("figure_counts", {})
        if fc.get("found", 0) > 0:
            logger.info("  Figures embedded:    %d", fc.get("embedded", 0))
    logger.info("")
    logger.info("  Search time:    %6.2fs", t_search)
    logger.info("  Fetch time:     %6.2fs", t_fetch)
    logger.info("  Parse time:     %6.2fs", t_parse)
    logger.info("  S2 time:        %6.2fs", t_s2)
    if ingest:
        logger.info("  Ingest time:    %6.2fs", t_ingest)
    logger.info("  ─────────────────────")
    logger.info("  TOTAL:          %6.2fs", total_time)
    logger.info("")
    logger.info("  Old pipeline:   %6ds  (est. %ds/paper × %d)",
                 old_pipeline_total_s, old_pipeline_per_paper_s, count)
    logger.info("  Speedup:        %6.1f×", speedup)
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
    cache_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("\nResults saved to %s", cache_path)

    return results


def main():
    p = argparse.ArgumentParser(description="Europe PMC pipeline test")
    p.add_argument("--count", type=int, default=10,
                   help="Number of papers to test (default: 10)")
    p.add_argument("--query", type=str, default="",
                   help="Search query (default: pre-built query)")
    p.add_argument("--ingest", action="store_true",
                   help="Ingest parsed chunks into ChromaDB + BM25")
    p.add_argument("--coverage", action="store_true",
                   help="Run coverage diagnostic (PMC vs Semantic Scholar)")
    p.add_argument("--figures", action="store_true",
                   help="Download and embed XML <graphic> figures (with --ingest)")
    p.add_argument("--graph", action="store_true",
                   help="Update the knowledge graph at ingest time (with --ingest)")
    args = p.parse_args()

    test_pipeline(
        count=args.count,
        ingest=args.ingest,
        query=args.query,
        coverage=args.coverage,
        ingest_figures=args.figures,
        use_graph=args.graph,
    )


if __name__ == "__main__":
    main()
