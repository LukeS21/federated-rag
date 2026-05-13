#!/usr/bin/env python
"""
Phase 8: Publication-Scale Benchmark Runner

Ingests external PDFs, runs the full survey pipeline on the expanded corpus,
builds the L0 claim index, and produces benchmark artifacts.

Usage:
    python phase8_benchmark.py              # full run (ingest + benchmark)
    python phase8_benchmark.py --skip-ingest  # benchmark only (already ingested)
    python phase8_benchmark.py --cached      # view cached benchmark results
    python phase8_benchmark.py --naive-only  # naive RAG baseline only

Results saved to:
    projects/default/phase8_benchmark.json     # full benchmark
    projects/default/phase8_naive_rag.json     # naive RAG baseline
    projects/default/survey_result_phase8.json # survey pipeline output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase8_benchmark")

PROJECT_DIR = Path("projects/default")
BENCHMARK_CACHE = PROJECT_DIR / "phase8_benchmark.json"
NAIVE_CACHE = PROJECT_DIR / "phase8_naive_rag.json"
SURVEY_CACHE = PROJECT_DIR / "survey_result_phase8.json"
EXTERNAL_DIR = Path("data/external")
DATA_DIR = Path("data")

BENCHMARK_QUERY = (
    "How do titanium implant surface modifications influence macrophage "
    "polarization, T cell responses, and osseointegration outcomes?"
)


def ingest_pdfs():
    """Ingest all PDFs from data/ and data/external/ into ChromaDB,
    run pre-extraction, vision pipeline, and build KG.

    Skips already-ingested PDFs."""
    from src.ingestion.pdf_parser import PDFParser
    from src.retrieval.chroma_client import ChromaClient
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.graph.networkx_json_storage import NetworkXJSONStorage
    from src.graph.graph_builder import GraphBuilder
    from src.ingestion.pre_extractor import PreExtractor
    from src.ingestion.pre_summarizer import PreSummarizer
    from src.vision.vision_ingest import vision_ingest_pdf
    from src.citation_manager.citekey_utils import resolve_cite_key, parse_paper_metadata, try_zotero_add
    from src.unicode_map import scrub_unicode

    CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
    GRAPH_PATH = str(PROJECT_DIR / "project_graph.json")

    chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
    bm25 = BM25Index(persist_dir=PROJECT_DIR / "bm25_index")
    if not bm25.load():
        # Rebuild from ChromaDB if no persisted copy
        all_docs = chroma.collection.get(include=["documents", "metadatas"])
        if all_docs.get("documents"):
            bm25.add_documents([d for d, m in zip(all_docs["documents"],
                                    all_docs.get("metadatas") or [])
                                if (m or {}).get("chunk_type") != "reference"])

    hybrid = HybridRetriever(chroma_client=chroma, bm25_index=bm25)
    graph_storage = NetworkXJSONStorage(file_path=str(GRAPH_PATH))
    parser = PDFParser()
    summarizer = PreSummarizer()

    # PreExtractor with caching
    from src.llm import resolve_model
    extract_model = resolve_model("small")
    pre_extractor = PreExtractor(model=extract_model)

    stats = {"ingested": 0, "skipped": 0, "errors": 0, "figures_extracted": 0}

    for src_dir in [DATA_DIR, EXTERNAL_DIR]:
        if not src_dir.exists():
            continue
        for pdf_path in sorted(src_dir.glob("*.pdf")):
            if pdf_path.name == "test.pdf" or pdf_path.name == "test2.pdf":
                continue

            # Check if already in ChromaDB
            try:
                existing = chroma.collection.get(
                    where={"source": pdf_path.name}, limit=1
                )
                if existing.get("ids"):
                    stats["skipped"] += 1
                    logger.info("SKIP (already ingested): %s", pdf_path.name)
                    continue
            except Exception:
                pass

            try:
                chunks = parser.parse(pdf_path)
                chunks_text = " ".join(c.get("text", "") for c in chunks if c.get("chunk_type") == "text")

                # Generate cite key
                cite_key = resolve_cite_key(pdf_path.name, chunks_text[:2000])

                # Add cite_key to chunk metadata
                for c in chunks:
                    c["metadata"]["cite_key"] = cite_key

                # Summarize chunks
                chunks = summarizer.summarize_all(chunks)

                # Ingest into retriever
                hybrid.ingest(chunks)

                # Pre-extract entities
                try:
                    pre_extractor.extract_paper(
                        paper_id=pdf_path.name,
                        chunks=chunks,
                        graph_storage=graph_storage,
                    )
                except Exception as e:
                    logger.warning("Pre-extraction failed for %s: %s", pdf_path.name, e)

                # Vision pipeline (caption-only fast path for scale)
                try:
                    vision_result = vision_ingest_pdf(pdf_path, hybrid, describe=False)
                    stats["figures_extracted"] += vision_result.get("extracted", 0)
                except Exception as e:
                    logger.warning("Vision ingest failed for %s: %s", pdf_path.name, e)

                stats["ingested"] += 1
                logger.info("INGESTED: %s (%d chunks)", pdf_path.name, len(chunks))

            except Exception as e:
                stats["errors"] += 1
                logger.error("Ingest error for %s: %s", pdf_path.name, e)

    graph_storage.save()
    logger.info("Ingestion complete: %s", stats)
    return stats, hybrid, graph_storage


def run_survey_pipeline(hybrid_retriever, graph_storage):
    """Run the full survey pipeline and return results."""
    from src.graph.graph_builder import build_survey_graph
    from src.state import AgentState
    from langgraph.graph import StateGraph

    initial_state: AgentState = {
        "user_query": BENCHMARK_QUERY,
        "query_scope": "public",
        "mode": "survey",
        "public_context": [],
        "secure_context": [],
        "extracted_entities": {},
        "synthesis_draft": "",
        "citations_used": [],
        "final_output": "",
        "human_approved": False,
        "routes": {},
        "discovered_categories": {},
        "knowledge_graph_snapshot": {},
        "critic_feedback": "",
        "synthesis_revised": "",
        "anchoring_score": 0.0,
        "ungrounded_claims": [],
        "chunk_summary": "",
        "ner_entities": [],
        "decomposed_themes": [],
        "thematic_clusters": {},
        "per_paper_extractions": {},
        "per_theme_syntheses": {},
        "cross_theme_synthesis": "",
        "gap_analysis": "",
        "chaptered_drafts": {},
    }

    logger.info("Building survey graph...")
    graph = build_survey_graph(hybrid_retriever, graph_storage)

    logger.info("Running survey pipeline...")
    t0 = time.monotonic()
    result = graph.invoke(initial_state)
    elapsed = time.monotonic() - t0

    return result, elapsed


def run_naive_rag_baseline(hybrid_retriever):
    """Run a naive single-pass RAG (retrieve → draft, no clustering/debate/KG)."""
    from src.llm import resolve_model, get_chat_model
    from src.anchoring.evidence_check import compute_anchoring_score
    from src.unicode_map import scrub_unicode

    logger.info("Running naive RAG baseline...")
    t0 = time.monotonic()

    # Retrieve
    chunks = hybrid_retriever.query(BENCHMARK_QUERY, similarity_threshold=1.5, max_chunks=30)
    chunk_texts = "\n\n".join(
        f"Chunk {i+1}: {c['text'][:300]}" for i, c in enumerate(chunks[:15])
    )

    # Draft
    model = resolve_model("small")  # gemma4:e4b — faster, more focused output
    llm = get_chat_model(model, temperature=0.0)
    prompt = (
        "You are a biomedical literature analyst. Given the evidence chunks below, "
        "list evidence-backed claims, one per line, no preamble or commentary. "
        "Each line must be a single factual claim supported by the evidence. "
        "Use ONLY exact citation keys from the evidence — never invent new ones. "
        "Output plain ASCII only.\n\n"
        f"Evidence:\n{chunk_texts}\n\n"
        "Claims (one per line):"
    )
    response = llm.invoke(prompt)
    synthesis = scrub_unicode(getattr(response, "content", str(response)))

    # Decompose into claims
    claims = [s.strip() for s in synthesis.replace("\n\n", "\n").split("\n") if s.strip() and len(s.strip()) > 20]

    # Anchoring
    anchor_score, ungrounded = compute_anchoring_score(claims, chunks)

    elapsed = time.monotonic() - t0

    result = {
        "query": BENCHMARK_QUERY,
        "synthesis": synthesis,
        "claims": claims,
        "claim_count": len(claims),
        "anchoring_score": float(anchor_score),
        "ungrounded_count": len(ungrounded),
        "chunks_retrieved": len(chunks),
        "elapsed_seconds": elapsed,
        "model": model,
        "timestamp": time.time(),
    }

    NAIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    NAIVE_CACHE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Naive RAG baseline cached to %s", NAIVE_CACHE)
    return result


def compute_benchmarks(survey_result, naive_result, ingest_stats):
    """Compute Tier A benchmarks from survey output."""
    from src.anchoring.evidence_check import compute_anchoring_score

    # Extract per-theme syntheses
    themes = survey_result.get("per_theme_syntheses", {})
    theme_count = len(themes)
    all_claims = []
    total_anchor = 0.0
    theme_anchors = {}

    for theme_name, theme_data in themes.items():
        if isinstance(theme_data, dict):
            synthesis = theme_data.get("synthesis", "")
            anchor = theme_data.get("anchoring_score", 0)
            claims = [s.strip() for s in synthesis.split("\n") if s.strip() and len(s.strip()) > 20]
            all_claims.extend(claims)
            total_anchor += float(anchor) if anchor else 0
            theme_anchors[theme_name] = float(anchor) if anchor else 0

    avg_anchor = total_anchor / max(theme_count, 1)

    # Cross-theme
    cross_theme = survey_result.get("cross_theme_synthesis", "")
    gap_analysis = survey_result.get("gap_analysis", "")

    # KG stats
    kg = survey_result.get("knowledge_graph_snapshot", {})
    kg_nodes = len(kg.get("nodes", [])) if isinstance(kg, dict) else 0

    benchmark = {
        "query": BENCHMARK_QUERY,
        "total_papers_ingested": ingest_stats.get("ingested", 0) + ingest_stats.get("skipped", 0),
        "papers_newly_ingested": ingest_stats.get("ingested", 0),
        "themes_discovered": theme_count,
        "per_theme_claims": len(all_claims),
        "mean_anchoring_score": round(avg_anchor, 4),
        "theme_anchors": theme_anchors,
        "kg_nodes": kg_nodes,
        "cross_theme_length": len(cross_theme),
        "gap_analysis_length": len(gap_analysis),
        "survey_elapsed_seconds": survey_result.get("_elapsed", 0),
        "naive_rag_claims": naive_result.get("claim_count", 0),
        "naive_rag_anchoring": naive_result.get("anchoring_score", 0),
        "claim_ratio": len(all_claims) / max(naive_result.get("claim_count", 1), 1),
        "timestamp": time.time(),
    }

    BENCHMARK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CACHE.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Benchmark cached to %s", BENCHMARK_CACHE)
    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Benchmark Runner")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip PDF ingestion")
    parser.add_argument("--cached", action="store_true", help="View cached results")
    parser.add_argument("--naive-only", action="store_true", help="Naive RAG baseline only")
    args = parser.parse_args()

    if args.cached:
        for name, path in [("Benchmark", BENCHMARK_CACHE), ("Naive RAG", NAIVE_CACHE)]:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                print(f"\n{'='*60}")
                print(f"  {name} Results")
                print(f"{'='*60}")
                for k, v in data.items():
                    if k not in ("synthesis", "cross_theme_synthesis", "gap_analysis"):
                        print(f"  {k}: {v}")
                if data.get("synthesis"):
                    print(f"\n  Synthesis preview:\n  {data['synthesis'][:500]}...")
                print(f"{'='*60}")
            else:
                print(f"No cached {name} results. Run without --cached.")
        return

    # ── Ingest ──
    if not args.skip_ingest:
        logger.info("=" * 50)
        logger.info("PHASE 8: INGESTION")
        logger.info("=" * 50)
        ingest_stats, hybrid, graph_storage = ingest_pdfs()
    else:
        from src.retrieval.chroma_client import ChromaClient
        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.graph.networkx_json_storage import NetworkXJSONStorage

        CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
        chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
        bm25 = BM25Index(persist_dir=PROJECT_DIR / "bm25_index")
        bm25.load()
        hybrid = HybridRetriever(chroma_client=chroma, bm25_index=bm25)
        graph_storage = NetworkXJSONStorage(file_path=str(PROJECT_DIR / "project_graph.json"))
        ingest_stats = {"ingested": 0, "skipped": 0}

    # ── Naive RAG Baseline ──
    logger.info("=" * 50)
    logger.info("PHASE 8: NAIVE RAG BASELINE")
    logger.info("=" * 50)
    naive_result = run_naive_rag_baseline(hybrid)

    if args.naive_only:
        print(f"\n  Naive RAG: {naive_result['claim_count']} claims, "
              f"anchoring={naive_result['anchoring_score']:.3f}")
        return

    # ── Survey Pipeline ──
    logger.info("=" * 50)
    logger.info("PHASE 8: SURVEY PIPELINE")
    logger.info("=" * 50)
    survey_result, elapsed = run_survey_pipeline(hybrid, graph_storage)
    survey_result["_elapsed"] = elapsed

    # Cache survey output
    SURVEY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SURVEY_CACHE.write_text(
        json.dumps({
            "cross_theme_synthesis": survey_result.get("cross_theme_synthesis", ""),
            "gap_analysis": survey_result.get("gap_analysis", ""),
            "theme_count": len(survey_result.get("per_theme_syntheses", {})),
            "elapsed": elapsed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Benchmarks ──
    benchmark = compute_benchmarks(survey_result, naive_result, ingest_stats)

    print(f"\n{'='*60}")
    print("  PHASE 8 — BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Papers ingested:   {benchmark['total_papers_ingested']}")
    print(f"  Themes:            {benchmark['themes_discovered']}")
    print(f"  Pipeline claims:   {benchmark['per_theme_claims']}")
    print(f"  Mean anchoring:    {benchmark['mean_anchoring_score']:.4f}")
    print(f"  Naive RAG claims:  {benchmark['naive_rag_claims']}")
    print(f"  Naive RAG anchor:  {benchmark['naive_rag_anchoring']:.4f}")
    print(f"  Claim ratio:       {benchmark['claim_ratio']:.1f}x")
    print(f"  KG nodes:          {benchmark['kg_nodes']}")
    print(f"  Survey time:       {benchmark['survey_elapsed_seconds']:.0f}s")
    print(f"{'='*60}")
    print(f"\n  Detailed results:  {BENCHMARK_CACHE}")
    print(f"  Naive RAG:         {NAIVE_CACHE}")
    print(f"  Survey output:     {SURVEY_CACHE}")


if __name__ == "__main__":
    sys.exit(main())
