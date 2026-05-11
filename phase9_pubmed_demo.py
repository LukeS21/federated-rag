#!/usr/bin/env python
"""
Phase 9 POC: Literature Discovery Demo

Queries Semantic Scholar (free, no key needed) to find papers that would add
novelty to the local corpus.  Measures coverage delta without modifying the
pipeline graph.  Results cached to projects/default/literature_discovery.json.

Usage:
    # Search for novel papers on titanium implant immune response
    python phase9_pubmed_demo.py

    # View cached results (instant)
    python phase9_pubmed_demo.py --cached

    # Custom query
    python phase9_pubmed_demo.py --query "macrophage polarization biomaterials"

    # Search both PubMed and Semantic Scholar
    python phase9_pubmed_demo.py --provider both
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load env for API keys
from dotenv import load_dotenv
load_dotenv(override=True)

from src.retrieval.semantic_scholar import SemanticScholarClient
from src.retrieval.pubmed import PubMedClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase9_demo")

PROJECT_DIR = Path("projects/default")
CACHE_PATH = PROJECT_DIR / "literature_discovery.json"

QUERIES = [
    "titanium implant macrophage polarization T cell immune response",
    "CD4 CD8 T cells bone healing osseointegration",
    "cytokine profile titanium surface modification in vivo",
    "mesenchymal stem cell recruitment biomaterial implant",
    "peri-implant inflammation immunomodulation surface roughness",
]


def get_local_paper_titles() -> Set[str]:
    """Get titles of papers already in the local corpus."""
    chroma_path = PROJECT_DIR / "chroma_data"
    if not chroma_path.exists():
        return set()

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection("public_corpus")
        data = collection.get(include=["metadatas"])
        sources = set()
        for meta in (data.get("metadatas", []) or []):
            if isinstance(meta, dict) and meta.get("source"):
                sources.add(meta["source"].replace(".pdf", ""))
        return sources
    except Exception:
        return set()


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy comparison."""
    return " ".join(title.lower().split())[:80]


def search_and_compare(
    queries: List[str],
    provider: str = "semantic_scholar",
    max_per_query: int = 10,
) -> Dict:
    """Search external sources and identify novel papers not in local corpus.

    Returns a dict with: queries, total_found, novel_papers, local_overlap,
                        novel_themes, known_titles.
    """
    local_titles = get_local_paper_titles()
    local_norm = {normalize_title(t) for t in local_titles}

    client = SemanticScholarClient() if provider == "semantic_scholar" else None
    pm_client = PubMedClient() if provider in ("pubmed", "both") else None

    all_papers: List[Dict] = []
    novel_papers: List[Dict] = []
    known_papers: List[Dict] = []

    for query in queries:
        logger.info("Searching: %s", query[:80])

        papers = []
        if provider in ("semantic_scholar", "both"):
            try:
                papers = client.search(query, limit=max_per_query)
                time.sleep(1.1)  # respect rate limit (no key = 1 req/s)
            except Exception as e:
                logger.warning("S2 search failed for '%s': %s", query[:40], e)

        if provider in ("pubmed", "both") and pm_client:
            try:
                pm_results = pm_client.search(query, max_results=max_per_query)
                # Convert to match S2 format
                pm_papers = [{
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "year": p.get("pub_date", "")[:4] if p.get("pub_date") else None,
                    "doi": p.get("doi", ""),
                    "pmid": p.get("pmid", ""),
                    "authors": p.get("authors", []),
                    "journal": p.get("source", ""),
                    "source": "pubmed",
                } for p in pm_results]
                papers.extend(pm_papers)
            except Exception as e:
                logger.warning("PubMed search failed for '%s': %s", query[:40], e)

        for paper in papers:
            all_papers.append({**paper, "query": query[:80]})
            title_norm = normalize_title(paper.get("title", ""))
            is_novel = not any(title_norm[:40] in lt for lt in local_norm)
            if is_novel:
                novel_papers.append({**paper, "query": query[:80], "novel": True})
            else:
                known_papers.append({**paper, "query": query[:80], "novel": False})

    # Identify novel themes/keywords
    novel_keywords: Dict[str, int] = {}
    for paper in novel_papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '') or paper.get('tldr', '')}"
        keywords = extract_keywords(text)
        for kw in keywords:
            novel_keywords[kw] = novel_keywords.get(kw, 0) + 1

    return {
        "queries": queries,
        "provider": provider,
        "timestamp": time.time(),
        "local_papers_count": len(local_titles),
        "total_found": len(all_papers),
        "novel_count": len(novel_papers),
        "known_count": len(known_papers),
        "novelty_rate": round(len(novel_papers) / max(len(all_papers), 1), 3),
        "novel_papers": novel_papers,
        "known_papers": known_papers,
        "local_titles": sorted(local_titles),
        "novel_keywords_top": sorted(novel_keywords.items(), key=lambda x: -x[1])[:20],
    }


def extract_keywords(text: str) -> List[str]:
    """Extract simple biomedical keywords from text."""
    keywords = [
        "macrophage", "neutrophil", "lymphocyte", "monocyte", "dendritic",
        "cytokine", "chemokine", "interleukin", "TNF", "IFN",
        "titanium", "implant", "osseointegration", "biomaterial", "scaffold",
        "osteoblast", "osteoclast", "mesenchymal", "MSC", "bone",
        "polarization", "M1", "M2", "anti-inflammatory", "pro-inflammatory",
        "surface", "roughness", "hydrophilic", "coating", "modification",
        "in vivo", "in vitro", "mouse", "rat", "rabbit", "porcine",
        "CD4", "CD8", "T cell", "B cell", "regulatory",
        "wnt", "BMP", "TGF", "VEGF", "PDGF",
        "histology", "microCT", "histomorphometry", "push-out", "torque",
        "nanotube", "anodization", "sandblasted", "acid-etched", "SLA",
        "hydrogel", "drug delivery", "antibacterial", "silver", "copper",
    ]
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    return found


def view_cached():
    """Display cached literature discovery results."""
    if not CACHE_PATH.exists():
        print("No cached results. Run: python phase9_pubmed_demo.py")
        return

    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    print("\n" + "=" * 80)
    print("  PHASE 9 POC — Literature Discovery Results")
    print("=" * 80)
    print(f"  Provider: {data['provider']}")
    print(f"  Queries run: {len(data['queries'])}")
    print(f"  Local papers: {data['local_papers_count']}")
    print(f"  Papers found: {data['total_found']}")
    print(f"  Novel papers: {data['novel_count']} ({data['novelty_rate']*100:.0f}%)")
    print(f"  Already known: {data['known_count']}")
    print()

    if data.get("novel_keywords_top"):
        print("  Novel keywords (not well-covered locally):")
        for kw, count in data["novel_keywords_top"][:10]:
            print(f"    {kw}: {count} papers")
        print()

    print("  Top novel papers:")
    for p in data.get("novel_papers", [])[:5]:
        title = p.get("title", "")[:100]
        year = p.get("year", "?")
        authors = ", ".join(p.get("authors", [])[:3])
        print(f"    [{year}] {title}")
        if authors:
            print(f"           {authors}")
        abstract = p.get("abstract", "") or p.get("tldr", "")
        if abstract:
            print(f"           {abstract[:150]}...")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Literature Discovery POC")
    parser.add_argument("--cached", action="store_true", help="View cached results")
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--provider", type=str, default="semantic_scholar",
                        choices=["semantic_scholar", "pubmed", "both"],
                        help="API provider (default: semantic_scholar)")
    parser.add_argument("--max", type=int, default=10, help="Max results per query")
    args = parser.parse_args()

    if args.cached:
        view_cached()
        return

    queries = [args.query] if args.query else QUERIES

    print(f"\nSearching {len(queries)} queries via {args.provider}...")
    t0 = time.monotonic()

    result = search_and_compare(
        queries=queries,
        provider=args.provider,
        max_per_query=args.max,
    )

    elapsed = time.monotonic() - t0

    # ── Display ──
    print(f"\n{'=' * 80}")
    print(f"  PHASE 9 POC — Literature Discovery")
    print(f"{'=' * 80}")
    print(f"  Provider: {result['provider']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Local papers: {result['local_papers_count']}")
    print(f"  Found: {result['total_found']}")
    print(f"  Novel: {result['novel_count']} ({result['novelty_rate']*100:.0f}%)")
    print(f"  Known: {result['known_count']}")
    print()

    if result.get("novel_keywords_top"):
        print("  Top novel keywords:")
        for kw, count in result["novel_keywords_top"][:8]:
            print(f"    {kw}: {count}")
        print()

    if result["novel_papers"]:
        print("  Top novel papers (first 3):")
        for p in result["novel_papers"][:3]:
            print(f"    {p.get('title','')[:120]}")
            print(f"    Year: {p.get('year','?')}, Authors: {', '.join(p.get('authors',[])[:3])}")
            tldr = p.get('tldr','') or p.get('abstract','')[:150]
            if tldr:
                print(f"    {tldr[:150]}")
            print()
    print(f"{'=' * 80}")

    # ── Cache ──
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nResults cached to: {CACHE_PATH}")


if __name__ == "__main__":
    sys.exit(main())
