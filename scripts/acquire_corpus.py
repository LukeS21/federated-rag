#!/usr/bin/env python
"""
Phase 8: Corpus Acquisition — Batch Search & Download

Searches PubMed + Semantic Scholar across diverse BME queries,
deduplicates results, downloads OA PDFs via the 4-layer resolution chain,
and saves results to ``data/external/``.

Usage:
    python scripts/acquire_corpus.py              # full run
    python scripts/acquire_corpus.py --dry-run    # search only, no download
    python scripts/acquire_corpus.py --cached     # view cached results
    python scripts/acquire_corpus.py --max 50     # cap at 50 papers
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.retrieval.pubmed import PubMedClient
from src.retrieval.semantic_scholar import SemanticScholarClient
from src.retrieval.unpaywall import UnpaywallClient
from src.retrieval.pmc_oa import PMCOAClient
from src.retrieval.pdf_downloader import PDFDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acquire_corpus")

PROJECT_DIR = Path("projects/default")
CACHE_PATH = PROJECT_DIR / "corpus_acquisition.json"

# ── Diverse BME search queries (target: 100+ papers) ──────────────────────────
QUERIES = [
    "titanium implant macrophage polarization immune response",
    "surface modification titanium osseointegration in vivo",
    "CD4 CD8 T cell bone healing implant",
    "cytokine profile titanium surface roughness macrophage",
    "peri-implant inflammation immunomodulation biomaterial",
    "mesenchymal stem cell recruitment implant osseointegration",
    "antibacterial coating titanium implant silver copper",
    "hydrogel coating drug delivery titanium implant",
    "nanotube anodization titanium surface osteoblast",
    "Wnt BMP signaling biomaterial implant bone formation",
    "macrophage M1 M2 polarization biomaterial surface",
    "titanium alloy Ti-6Al-4V immune response implant",
    "rough hydrophilic titanium implant osteogenesis",
    "neutrophil extracellular trap biomaterial implant",
    "dendritic cell biomaterial immune response",
    "regulatory T cell implant tolerance",
    "osteoclast osteoblast titanium particle wear debris",
    "push-out torque biomechanical testing implant",
    "microCT histomorphometry titanium implant bone",
    "acid-etched sandblasted SLA titanium surface osteoblast",
]


def search_all(
    queries: List[str],
    provider: str = "pubmed",
    max_per_query: int = 15,
) -> Dict:
    """Search PubMed and/or Semantic Scholar across all queries."""
    pm = PubMedClient() if provider in ("pubmed", "both") else None
    s2 = SemanticScholarClient() if provider in ("semantic_scholar", "both") else None

    all_papers: List[Dict] = []
    seen_dois: Set[str] = set()
    seen_titles: Set[str] = set()

    for i, query in enumerate(queries):
        logger.info("[%d/%d] Searching: %s", i + 1, len(queries), query[:80])

        # ── PubMed ──
        if pm:
            try:
                results = pm.search(query, max_results=max_per_query)
                for r in results:
                    doi = (r.get("doi", "") or "").lower()
                    title = (r.get("title", "") or "").lower()[:80]
                    if doi and doi in seen_dois:
                        continue
                    if title and title in seen_titles:
                        continue
                    if doi:
                        seen_dois.add(doi)
                    if title:
                        seen_titles.add(title)
                    all_papers.append({
                        "title": r.get("title", ""),
                        "abstract": r.get("abstract", ""),
                        "year": r.get("pub_date", "")[:4] if r.get("pub_date") else None,
                        "doi": r.get("doi", ""),
                        "pmid": r.get("pmid", ""),
                        "authors": r.get("authors", []),
                        "journal": r.get("source", ""),
                        "source": "pubmed",
                        "query": query[:80],
                    })
                time.sleep(0.15)  # respect rate limit
            except Exception as e:
                logger.warning("PubMed search failed: %s", e)

        # ── Semantic Scholar ──
        if s2:
            try:
                s2_results = s2.search(query, limit=max_per_query)
                for r in s2_results:
                    doi = (r.get("doi", "") or "").lower()
                    title = (r.get("title", "") or "").lower()[:80]
                    if doi and doi in seen_dois:
                        continue
                    if title and title in seen_titles:
                        continue
                    if doi:
                        seen_dois.add(doi)
                    if title:
                        seen_titles.add(title)
                    all_papers.append({
                        "title": r.get("title", ""),
                        "abstract": r.get("abstract", "") or r.get("tldr", ""),
                        "year": r.get("year"),
                        "doi": r.get("doi", ""),
                        "pmid": r.get("pmid", ""),
                        "authors": r.get("authors", []),
                        "journal": r.get("journal", ""),
                        "open_access_pdf": r.get("open_access_pdf", ""),
                        "source": "semantic_scholar",
                        "query": query[:80],
                    })
                time.sleep(1.3)  # S2 free tier: 1 req/s
            except Exception as e:
                logger.warning("S2 search failed: %s", e)

    return {
        "total_found": len(all_papers),
        "queries_run": len(queries),
        "provider": provider,
        "papers": all_papers,
        "timestamp": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Corpus Acquisition")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search only, skip PDF download")
    parser.add_argument("--cached", action="store_true",
                        help="View cached search results")
    parser.add_argument("--max", type=int, default=200,
                        help="Max papers to download (default: 200)")
    parser.add_argument("--provider", type=str, default="pubmed",
                        choices=["pubmed", "semantic_scholar", "both"])
    parser.add_argument("--per-query", type=int, default=15,
                        help="Max results per query (default: 15)")
    args = parser.parse_args()

    if args.cached:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            print(f"\n{'='*70}")
            print("  CACHED SEARCH RESULTS")
            print(f"{'='*70}")
            print(f"  Papers found: {data['total_found']}")
            print(f"  Queries run:  {data['queries_run']}")
            print(f"  Provider:     {data['provider']}")
            print(f"  Timestamp:    {time.ctime(data['timestamp'])}")
            if data.get("download_stats"):
                s = data["download_stats"]
                print(f"\n  Downloads:    {s.get('downloaded',0)}")
                print(f"  Skipped:      {s.get('skipped_exists',0)}")
                print(f"  Unfetchable:  {s.get('unfetchable',0)}")
            print(f"{'='*70}\n")
        else:
            print("No cached results. Run without --cached first.")
        return

    t0 = time.monotonic()

    # ── Step 1: Search ──
    print(f"\n{'='*70}")
    print("  PHASE 8 — CORPUS ACQUISITION")
    print(f"{'='*70}")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  Provider: {args.provider}")
    print(f"{'='*70}\n")

    result = search_all(QUERIES, provider=args.provider, max_per_query=args.per_query)
    elapsed = time.monotonic() - t0
    print(f"\n  Search complete: {result['total_found']} unique papers in {elapsed:.0f}s\n")

    # Save search results
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Search results cached to %s", CACHE_PATH)

    if args.dry_run:
        for p in result["papers"][:10]:
            print(f"  [{p.get('year','?')}] {p.get('title','')[:100]}")
            print(f"         DOI: {p.get('doi','') or 'N/A'}, PMID: {p.get('pmid','') or 'N/A'}")
        print(f"\n  ... and {len(result['papers'])-10} more papers")
        print("  Dry run — skipping PDF download.")
        return

    # ── Step 2: Download PDFs ──
    papers = result["papers"][:args.max]
    print(f"\n  Downloading PDFs for {len(papers)} papers...\n")

    downloader = PDFDownloader()
    pmc = PMCOAClient()
    uw = UnpaywallClient()

    results = []
    for i, paper in enumerate(papers):
        if i % 10 == 0:
            logger.info("Download progress: %d/%d", i, len(papers))
        dl_result = downloader.download(paper, pmc_client=pmc, unpaywall_client=uw)
        results.append(dl_result)

        # Status logging
        if dl_result["status"] == "downloaded":
            logger.info("  OK  [%s] %s", dl_result.get("source", "?"),
                         paper.get("title", "")[:70])
        time.sleep(0.2)

    # ── Step 3: Summary ──
    elapsed_total = time.monotonic() - t0
    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    skipped = sum(1 for r in results if r["status"] == "skipped_exists")
    unfetchable = sum(1 for r in results if r["status"] == "unfetchable")

    print(f"\n{'='*70}")
    print("  ACQUISITION COMPLETE")
    print(f"{'='*70}")
    print(f"  Searched:      {result['total_found']} papers")
    print(f"  Downloaded:    {downloaded}")
    print(f"  Already exist: {skipped}")
    print(f"  Unfetchable:   {unfetchable}")
    print(f"  Total time:    {elapsed_total:.0f}s")
    print(f"  Output:        data/external/")
    print(f"{'='*70}\n")

    # Save summary
    summary = {
        **result,
        "download_stats": {
            "downloaded": downloaded,
            "skipped_exists": skipped,
            "unfetchable": unfetchable,
            "errors": downloader.stats["errors"],
            "total_time": elapsed_total,
        },
        "download_results": [
            {"title": r.get("title","")[:80], "status": r["status"],
             "source": r.get("source"), "path": r.get("path")}
            for r in results
        ],
    }
    CACHE_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Also save parsed manifest
    manifest_path = Path("data/external/manifest.json")
    manifest = {
        "downloaded": downloaded,
        "total_attempted": len(papers),
        "timestamp": time.time(),
        "papers": [
            {"title": r.get("title","")[:100], "path": r.get("path",""),
             "year": r.get("year"), "doi": r.get("doi","")}
            for r in results if r["status"] == "downloaded"
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Manifest: data/external/manifest.json\n")


if __name__ == "__main__":
    sys.exit(main())
