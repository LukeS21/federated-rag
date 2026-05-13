#!/usr/bin/env python
"""
Phase 7b: Sectioned Survey Benchmark & Demo

Demonstrates multi-turn section writing with claim/citation ledger and figure
integration.  Results cached to projects/default/sectioned_survey_result.json.

Usage:
    # Full sectioned survey (requires Ollama, ~5-10 min)
    python phase7_section_writing.py

    # View cached results (instant)
    python phase7_section_writing.py --cached

    # Custom query and sections
    python phase7_section_writing.py --query "Synthesize titanium implant immune response" --sections Introduction,Results
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Load .env before any imports that read environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.synthesis.claim_ledger import ClaimLedger
from src.agents.synthesis_drafter import SynthesisDrafter
from src.llm import resolve_model, get_chat_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase7b")

PROJECT_DIR = Path("projects/default")
RESULT_PATH = PROJECT_DIR / "sectioned_survey_result.json"
LEDGER_PATH = PROJECT_DIR / "section_ledger.json"

SECTIONS_IMRAD = [
    {"name": "Introduction", "description": "Background on titanium implant immune response"},
    {"name": "Methods", "description": "Experimental models and measurement techniques"},
    {"name": "Results", "description": "Key findings on T cells, macrophages, and bone healing"},
    {"name": "Discussion", "description": "Synthesis, implications, and research gaps"},
]


def build_retriever() -> HybridRetriever:
    """Build retriever from persisted ChromaDB. BM25 is in-memory only."""
    chroma = ChromaClient(
        "public_corpus",
        persist_directory=str(PROJECT_DIR / "chroma_data"),
    )
    bm25 = BM25Index()
    return HybridRetriever(chroma, bm25)


def run_sectioned_survey(
    query: str,
    sections: List[Dict] | None = None,
    retriever: HybridRetriever | None = None,
) -> Dict:
    """Run a complete sectioned survey synthesis.

    Returns a dict with: query, sections, section_drafts, claim_ledger,
    figure_count, ledger_report, timing.
    """
    if sections is None:
        sections = SECTIONS_IMRAD

    if retriever is None:
        retriever = build_retriever()

    ledger = ClaimLedger(ledger_path=LEDGER_PATH)
    section_drafts: Dict[str, str] = {}
    total_figures = 0
    timings: Dict[str, float] = {}

    t_total = time.monotonic()

    for i, section in enumerate(sections):
        sec_name = section["name"]
        sec_desc = section["description"]
        logger.info("Drafting section %d/%d: %s", i + 1, len(sections), sec_name)

        t0 = time.monotonic()

        # ── Retrieve ──
        section_query = f"{sec_desc}: {query}"
        chunks = retriever.query(
            section_query,
            similarity_threshold=1.5,
            max_chunks=20,
            filter_references=True,
            include_figures=True,
        )

        text_chunks = [c for c in chunks if (c.get("metadata", {}) or {}).get("chunk_type") != "figure"]
        figure_chunks = [c for c in chunks if (c.get("metadata", {}) or {}).get("chunk_type") == "figure"]
        total_figures += len(figure_chunks)

        # ── Build summaries ──
        summaries: List[str] = []
        for ch in text_chunks:
            meta = ch.get("metadata", {}) or {}
            cs = meta.get("chunk_summary", ch.get("text", "")[:300])
            if cs:
                summaries.append(cs)

        for fig in figure_chunks:
            desc = fig.get("text", "")
            caption = (fig.get("metadata", {}) or {}).get("caption", "")
            if desc:
                summaries.append(f"[Figure: {caption}] {desc}")

        summary_text = "\n\n".join(summaries[:15])
        summary_chunks = [{"text": summary_text, "metadata": {"source": f"section:{sec_name}"}}]

        citations = sorted({
            (ch.get("metadata", {}) or {}).get("cite_key") or
            (ch.get("metadata", {}) or {}).get("source", "unknown")
            for ch in text_chunks
        })

        # ── Draft ──
        drafter = SynthesisDrafter(
            model_name=sec_name,
            model=resolve_model("small"),
        )

        draft = drafter.draft(
            query=f"{query} [Section: {sec_name}]",
            entities={},
            chunks=summary_chunks,
            citations=citations,
            kg_context=_build_prior_context(section_drafts, sections, i),
        )

        # ── Ledger tracking ──
        claims = [line.strip() for line in draft.split("\n") if line.strip()]
        new_claims = ledger.filter_new_claims(claims)
        for c_text in new_claims:
            ledger.add_claim(c_text, section=sec_name)

        section_drafts[sec_name] = "\n".join(new_claims)

        elapsed = time.monotonic() - t0
        timings[sec_name] = round(elapsed, 2)

        dupe_count = len(claims) - len(new_claims)
        logger.info(
            "  %s: %d claims (%d new, %d duplicates) in %.1fs",
            sec_name, len(claims), len(new_claims), dupe_count, elapsed,
        )

    # ── Assemble ──
    parts = []
    for section in sections:
        name = section["name"]
        draft = section_drafts.get(name, "")
        parts.append(f"## {name.upper()}\n\n{draft.strip()}")

    manuscript = "\n\n".join(parts)

    # ── Coverage report ──
    all_citations = sorted({
        (ch.get("metadata", {}) or {}).get("cite_key") or
        (ch.get("metadata", {}) or {}).get("source", "unknown")
        for ch in retriever.query(query, max_chunks=50, filter_references=True)
        if (ch.get("metadata", {}) or {}).get("chunk_type") != "figure"
    })
    report = ledger.coverage_report(set(all_citations))

    # ── Save ──
    ledger.save(LEDGER_PATH)

    result = {
        "phase": "7b_sectioned_survey",
        "query": query,
        "sections": [s["name"] for s in sections],
        "section_drafts": section_drafts,
        "manuscript": manuscript,
        "claim_ledger_summary": {
            "total_claims": len(ledger),
            "unique_citations": report.get("unique_citations_used", 0),
            "coverage_rate": report.get("coverage_rate"),
            "duplicate_count": report.get("duplicate_count", 0),
            "ungrounded_count": report.get("ungrounded_count", 0),
        },
        "per_section": report.get("per_section", {}),
        "figure_chunks_total": total_figures,
        "timing": {
            **timings,
            "total_s": round(time.monotonic() - t_total, 2),
        },
    }

    RESULT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Result saved to %s", RESULT_PATH)

    return result


def _build_prior_context(section_drafts: Dict[str, str], sections: List[Dict], current_idx: int) -> str:
    """Build context from prior sections for cross-section continuity."""
    parts = []
    for i in range(current_idx):
        name = sections[i]["name"]
        draft = section_drafts.get(name, "")
        if draft:
            parts.append(f"[Prior Section: {name}]\n{draft[:500]}")
    if parts:
        return "Cross-section context from prior sections:\n" + "\n\n".join(parts)
    return ""


def view_cached():
    """Print cached sectioned survey results."""
    if not RESULT_PATH.exists():
        print("No cached results. Run: python phase7_section_writing.py")
        return

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    print("\n" + "=" * 80)
    print("  PHASE 7b SECTIONED SURVEY — Cached Results")
    print("=" * 80)
    print(f"\n  Query: {result['query']}")
    print(f"  Sections: {', '.join(result['sections'])}")
    print(f"  Total claims: {result['claim_ledger_summary']['total_claims']}")
    print(f"  Unique citations: {result['claim_ledger_summary']['unique_citations']}")
    print(f"  Coverage rate: {result['claim_ledger_summary']['coverage_rate']}")
    print(f"  Duplicate claims filtered: {result['claim_ledger_summary']['duplicate_count']}")
    print(f"  Figure chunks retrieved: {result.get('figure_chunks_total', 0)}")
    print(f"  Total time: {result['timing']['total_s']}s")
    print()

    for section in result["sections"]:
        draft = result["section_drafts"].get(section, "")
        claims_count = len([l for l in draft.split("\n") if l.strip()])
        sec_info = result.get("per_section", {}).get(section, {})
        print(f"  --- {section.upper()} ---")
        print(f"  Claims: {claims_count}  |  Citations: {sec_info.get('unique_citations', '?')}  |  Ungrounded: {sec_info.get('ungrounded', 0)}")
        print(f"  Draft preview:")
        for line in draft.split("\n")[:3]:
            print(f"    {line}")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 7b Sectioned Survey")
    parser.add_argument("--cached", action="store_true", help="View cached results")
    parser.add_argument("--query", type=str, default=None,
                        help="Research query (default: built-in biomedical query)")
    parser.add_argument("--sections", type=str, default=None,
                        help="Comma-separated section names (default: IMRaD)")
    args = parser.parse_args()

    if args.cached:
        view_cached()
        return

    query = args.query or (
        "Synthesize what is known about the immune response to titanium "
        "implants, focusing on CD4+ and CD8+ T cells, macrophage polarization, "
        "and bone healing outcomes."
    )

    sections = SECTIONS_IMRAD
    if args.sections:
        sec_names = [s.strip() for s in args.sections.split(",")]
        sections = [{"name": n, "description": f"{n} section content"} for n in sec_names]

    result = run_sectioned_survey(query, sections)

    print("\n" + "=" * 80)
    print(f"  PHASE 7b SECTIONED SURVEY — Complete")
    print("=" * 80)
    print(f"  Query: {query}")
    print(f"  Sections: {len(sections)}")
    print(f"  Total claims: {result['claim_ledger_summary']['total_claims']}")
    print(f"  Unique citations: {result['claim_ledger_summary']['unique_citations']}")
    if result['claim_ledger_summary'].get('coverage_rate'):
        print(f"  Coverage: {result['claim_ledger_summary']['coverage_rate']*100:.0f}%")
    print(f"  Figure chunks: {result.get('figure_chunks_total', 0)}")
    print(f"  Total time: {result['timing']['total_s']}s")
    print(f"\n  Output saved to: {RESULT_PATH}")
    print(f"  Ledger saved to: {LEDGER_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
