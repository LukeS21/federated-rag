#!/usr/bin/env python3
"""Phase 4 Benchmark: DeepSeek v4 Pro vs DeepSeek Chat for per-document extraction.

Compares extraction quality, entity count, category coverage, and latency
to determine whether deepseek-chat can replace deepseek-v4-pro for the
high-volume per-document extraction step in Survey Mode.

Uses already-ingested ChromaDB data (pre-summarized chunks from the demo).
No PDF parsing or pre-summarization on this run.

Usage:
    python phase4_benchmark.py
"""

import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.extraction_agent import ExtractionAgent
from src.retrieval.chroma_client import ChromaClient
from src.unicode_map import scrub_unicode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase4_benchmark")

MAX_CHUNKS_PER_PAPER = 25


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def count_entities(extracted: dict) -> int:
    total = 0
    for key, val in extracted.items():
        if isinstance(val, list):
            total += len(val)
        elif isinstance(val, dict) and "discovered_categories" not in key.lower():
            total += count_entities(val)
    return total


def count_categories(discovered: dict) -> int:
    cats = discovered.get("discovered_categories", [])
    return len(cats) if isinstance(cats, list) else 0


def safe_entity_types(extracted: dict) -> list:
    types = []
    for key, val in extracted.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            types.append(f"{key} ({len(val)})")
    return types


def sample_chunks(chunks: list, max_n: int) -> list:
    if len(chunks) <= max_n:
        return chunks
    step = len(chunks) / max_n
    return [chunks[int(i * step)] for i in range(max_n)]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 4 BENCHMARK: DeepSeek v4 Pro vs DeepSeek Chat")
    print("Per-document extraction quality comparison")
    print("=" * 70)

    PROJECT_DIR = Path("projects/default")
    CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
    chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)

    # Get all documents with metadata from ChromaDB
    all_data = chroma.collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []

    if not docs:
        print("No documents in ChromaDB. Run phase3_demo.py first to ingest PDFs.")
        return

    # Group chunks by source PDF
    papers = {}
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        meta = meta or {}
        source = meta.get("source", "unknown")
        if source not in papers:
            papers[source] = []
        chunk_type = meta.get("chunk_type", "text")
        papers[source].append({
            "text": scrub_unicode(str(doc)),
            "metadata": {**meta, "chunk_type": chunk_type},
        })

    # Filter references and sample
    for source in papers:
        body = [ch for ch in papers[source] if ch["metadata"].get("chunk_type") != "reference"]
        papers[source] = sample_chunks(body, MAX_CHUNKS_PER_PAPER)

    pdf_sources = sorted(papers.keys())
    print(f"\nFound {len(pdf_sources)} paper(s) in ChromaDB:")
    for src in pdf_sources:
        print(f"  - {src} ({len(papers[src])} body chunks)")

    query = (
        "What are the key findings, materials, cell types, cytokines, "
        "experimental methods, and model systems described in this paper?"
    )

    models = ["deepseek-v4-pro", "deepseek-chat"]
    results = {}

    for model_name in models:
        print(f"\n{'─' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'─' * 70}")

        agent = ExtractionAgent(model=model_name)
        model_results = []
        total_entities = 0
        total_categories = 0
        total_latency = 0.0

        for source in pdf_sources:
            body_chunks = papers[source]
            print(f"\n  === {source} ({len(body_chunks)} chunks) ===")

            # Build summary-only chunks from pre-written metadata summaries
            summary_chunks = []
            for ch in body_chunks:
                meta = ch.get("metadata", {})
                summary = meta.get("chunk_summary", ch.get("text", "")[:200])
                summary_chunks.append({"text": summary, "metadata": meta})

            t0 = time.time()
            categories = agent.discover_categories(summary_chunks, query)
            cat_latency = time.time() - t0
            n_cats = count_categories(categories)
            logger.info("  Category discovery: %.1fs → %d categories", cat_latency, n_cats)

            t0 = time.time()
            entities = agent.extract_entities(body_chunks, categories, query)
            ext_latency = time.time() - t0
            n_ents = count_entities(entities)
            ent_types = safe_entity_types(entities)
            logger.info("  Entity extraction: %.1fs → %d entities across %d types",
                         ext_latency, n_ents, len(ent_types))

            total_entities += n_ents
            total_categories += n_cats
            total_latency += cat_latency + ext_latency

            model_results.append({
                "source": source,
                "num_chunks": len(body_chunks),
                "categories_found": n_cats,
                "entity_count": n_ents,
                "entity_types": ent_types,
                "cat_latency": round(cat_latency, 2),
                "ext_latency": round(ext_latency, 2),
                "total_latency": round(cat_latency + ext_latency, 2),
            })

        results[model_name] = {
            "results": model_results,
            "total_entities": total_entities,
            "total_categories": total_categories,
            "total_latency": round(total_latency, 2),
            "num_papers": len(papers),
        }

    # -------------------------------------------------------------------
    #  Comparison Report
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    pro = results.get("deepseek-v4-pro", {})
    chat = results.get("deepseek-chat", {})

    pro_res = pro.get("results", [])
    chat_res = chat.get("results", [])

    for pr, cr in zip(pro_res, chat_res):
        src = pr["source"]
        print(f"\n--- {src} ({pr['num_chunks']} chunks) ---")
        print(f"  v4 Pro:  {pr['total_latency']:.0f}s | {pr['entity_count']} entities | {pr['categories_found']} cats")
        print(f"  Chat:    {cr['total_latency']:.0f}s | {cr['entity_count']} entities | {cr['categories_found']} cats")
        if pr["entity_count"] > 0:
            diff = abs(cr["entity_count"] - pr["entity_count"]) / pr["entity_count"] * 100
            print(f"  Entity diff: {diff:.1f}%")
        print(f"  v4 Pro types: {pr['entity_types']}")
        print(f"  Chat    types: {cr['entity_types']}")

    print(f"\n--- TOTALS ---")
    print(f"  v4 Pro: {pro.get('total_entities', 0)} entities, {pro.get('total_latency', 0):.0f}s")
    print(f"  Chat:   {chat.get('total_entities', 0)} entities, {chat.get('total_latency', 0):.0f}s")

    pro_total = pro.get("total_entities", 0)
    chat_total = chat.get("total_entities", 0)
    if pro_total > 0:
        diff_pct = abs(chat_total - pro_total) / pro_total * 100
        print(f"  Entity count diff: {diff_pct:.1f}%")

    # Cost: Chat ~half v4 Pro price
    print(f"\n  Estimated cost reduction: ~50% (Chat is ~half v4 Pro API price)")

    print(f"\n--- RECOMMENDATION ---")
    if pro_total > 0:
        diff_pct = abs(chat_total - pro_total) / pro_total * 100
        print(f"  Entity count difference: {diff_pct:.1f}% (threshold: 10%)")
        if diff_pct <= 10:
            print(f"  ✓ Use deepseek-chat for all Phase 4 per-document extraction.")
        else:
            print(f"  ✗ Chat differs by >10%. Check per-paper breakdown above.")
    else:
        print("  Unable to compute recommendation.")

    output_path = Path("projects/default/phase4_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
