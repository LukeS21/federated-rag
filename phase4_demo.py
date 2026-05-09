#!/usr/bin/env python3
"""Phase 4 Demo — Survey Mode with query decomposition, thematic clustering,
parallel per-document extraction, per-theme deep synthesis, and cross-theme
gap analysis.

Usage:
    OLLAMA_KEEP_ALIVE=30s python phase4_demo.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.errors import GraphRecursionError

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
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.networkx_json_storage import NetworkXJSONStorage
from src.graph.graph_builder import build_survey_graph
from src.unicode_map import scrub_unicode

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path("projects/default")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
BM25_CORPUS_PATH = str(PROJECT_DIR / "bm25_corpus.json")
GRAPH_PATH = str(PROJECT_DIR / "project_graph.json")

NUM_CTX = 16384
LLM_TIMEOUT = 900
os.environ.setdefault("LLM_TIMEOUT", str(LLM_TIMEOUT))
CLIENT_KWARGS = {"timeout": LLM_TIMEOUT}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase4_demo")


def _chroma_doc_count(chroma_client: ChromaClient) -> int:
    try:
        return int(chroma_client.collection.count())
    except Exception:
        return 0


def _ensure_bm25_loaded(chroma_client: ChromaClient, bm25: BM25Index, path: str) -> None:
    p = Path(path)
    if p.exists():
        try:
            corpus = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(corpus, list) and corpus:
                bm25.add_documents([str(x) for x in corpus])
                return
        except Exception:
            pass
    try:
        data = chroma_client.collection.get(include=["documents"])
        docs = (data or {}).get("documents") or []
        if docs:
            bm25.add_documents([str(d) for d in docs])
            p.write_text(json.dumps(bm25.corpus, indent=2), encoding="utf-8")
    except Exception:
        return


# ---------------------------------------------------------------------------
#  Step 1: Ingest PDFs (incremental)
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 4 SURVEY MODE DEMO")
print("=" * 70)

pdf_parser = PDFParser()
chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
bm25 = BM25Index()
retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)

existing_pdfs = set()
if _chroma_doc_count(chroma) > 0:
    try:
        all_meta = chroma.collection.get(include=["metadatas"])
        for m in (all_meta.get("metadatas") or []):
            src = (m or {}).get("source", "")
            if src:
                existing_pdfs.add(src)
    except Exception:
        pass

pdf_files = sorted(Path("data").glob("*.pdf"))
new_pdfs = [p for p in pdf_files if p.name not in existing_pdfs]
pre_summarizer = PreSummarizer()

# Load KG for pre-extraction
graph_storage = NetworkXJSONStorage(GRAPH_PATH)
pre_extractor = PreExtractor()

if new_pdfs:
    logger.info(f"Found {len(new_pdfs)} new PDF(s) to ingest.")
    for pdf_path in new_pdfs:
        logger.info(f"Ingesting: {pdf_path.name}")
        chunks = pdf_parser.parse(pdf_path)
        for ch in chunks:
            ch["text"] = scrub_unicode(ch["text"])

        # ── Content deduplication ──
        content_hash = compute_content_hash(chunks)
        existing_duplicate = check_content_duplicate(content_hash)
        if existing_duplicate:
            logger.info("SKIP: content identical to '%s' (hash: %s). Using existing data.",
                         existing_duplicate, content_hash)
            continue

        # ── Tiered cite key resolution ──
        extracted_title = extract_title_from_chunks(chunks)
        cite_key = resolve_cite_key(pdf_path.name, extracted_title)
        for ch in chunks:
            ch["metadata"]["cite_key"] = cite_key

        # ── Zotero integration (create or find existing) ──
        paper_meta = parse_paper_metadata(pdf_path.name)
        # Override title from chunks if available (better for Zotero search)
        if extracted_title:
            paper_meta["title"] = extracted_title
        zotero_key = try_zotero_add(paper_meta)
        if zotero_key:
            logger.info("Zotero: item -> %s (cite key: %s)", zotero_key, cite_key)
            for ch in chunks:
                ch["metadata"]["zotero_key"] = zotero_key
        else:
            logger.debug("Zotero: no item created (API not configured or failed)")

        # ── Register content hash ──
        save_content_hash(content_hash, pdf_path.name)

        # TF-IDF extractive pre-summarization
        chunks = pre_summarizer.summarize_all(chunks)
        # Pre-extract entities into KG + disk cache
        if not PreExtractor.is_extracted(pdf_path.name):
            pre_extractor.extract_paper(pdf_path.name, chunks, graph_storage=graph_storage)
        retriever.ingest(chunks)
        logger.info(f"  -> {len(chunks)} chunks ingested + pre-extracted.")
    Path(BM25_CORPUS_PATH).unlink(missing_ok=True)
else:
    logger.info(f"All {len(pdf_files)} PDF(s) already indexed (~{_chroma_doc_count(chroma)} chunks).")

_ensure_bm25_loaded(chroma, bm25, BM25_CORPUS_PATH)

# ---------------------------------------------------------------------------
#  Step 2: Build Survey Mode graph (KG already loaded above)
# ---------------------------------------------------------------------------
logger.info(f"Knowledge graph loaded from {GRAPH_PATH}")

# ---------------------------------------------------------------------------
#  Step 3: Build Survey Mode graph
# ---------------------------------------------------------------------------
app = build_survey_graph(retriever, graph_storage)
logger.info("Survey Mode LangGraph pipeline compiled.")

# ---------------------------------------------------------------------------
#  Step 4: Interactive loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print('Survey Mode ready. Enter broad research questions (or "quit" to exit).')
print("Example: Map the understanding of immune response to biomaterial surfaces")
print(f"Context: {NUM_CTX} tokens | Timeout: {LLM_TIMEOUT}s")
print("=" * 70)

while True:
    try:
        query = input("\nQuery: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        initial_state = {
            "user_query": query,
            "query_scope": "public",
            "mode": "survey",
            "num_ctx": NUM_CTX,
            "client_kwargs": CLIENT_KWARGS,
        }

        logger.info(f"Running Survey Mode for: {query}")

        config = {"configurable": {"thread_id": f"survey-{int(time.time())}"}}
        t0 = time.time()

        # Phase 1: Run pipeline — will stop at survey_scrub (human-in-the-loop)
        logger.info("Starting pipeline (will interrupt before scrub for review)...")
        final_state = app.invoke(initial_state, config)
        elapsed = time.time() - t0
        logger.info("Pipeline paused at human-in-the-loop checkpoint (%.1fs total)", elapsed)

        # ---- Display Results for review ----
        print("\n" + "=" * 70)
        print("REVIEW — Pipeline paused before final output.")
        print("Review the results below. Approve to finalize, or provide feedback.")
        print("=" * 70)

        # Save results for visualization tool
        try:
            viz_data = {
                "decomposed_themes": final_state.get("decomposed_themes", []),
                "thematic_clusters": final_state.get("thematic_clusters", {}),
                "per_theme_syntheses": final_state.get("per_theme_syntheses", {}),
                "cross_theme_synthesis": final_state.get("cross_theme_synthesis", ""),
                "gap_analysis": final_state.get("gap_analysis", ""),
            }
            from pathlib import Path as _Path
            _Path("projects/default").mkdir(parents=True, exist_ok=True)
            _Path("projects/default/survey_result.json").write_text(
                json.dumps(viz_data, indent=2, ensure_ascii=False, default=str)
            )
        except Exception:
            pass

        # ---- Display Results ----
        print("\n" + "-" * 50)
        print("DECOMPOSED THEMES")
        print("-" * 50)
        themes = final_state.get("decomposed_themes", [])
        if themes:
            for t in themes:
                print(f"  - {t.get('theme', '?')}")
                print(f"    Query: {t.get('sub_query', '')[:120]}")
        else:
            print("(none)")

        print("\n" + "-" * 50)
        print("THEMATIC CLUSTERS")
        print("-" * 50)
        clusters = final_state.get("thematic_clusters", {})
        if clusters:
            for theme_name, paper_ids in clusters.items():
                print(f"  {theme_name}: {len(paper_ids)} paper(s)")
                for pid in paper_ids[:5]:
                    print(f"    - {pid}")
                if len(paper_ids) > 5:
                    print(f"    ... and {len(paper_ids) - 5} more")
        else:
            print("(none)")

        print("\n" + "-" * 50)
        print("PER-PAPER EXTRACTION")
        print("-" * 50)
        per_paper = final_state.get("per_paper_extractions", {})
        if per_paper:
            for src, entities in per_paper.items():
                n_ents = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
                print(f"  {src}: {n_ents} entities across {len(entities)} categories")
        else:
            print("(none)")

        print("\n" + "-" * 50)
        print("PER-THEME SYNTHESES")
        print("-" * 50)
        theme_syntheses = final_state.get("per_theme_syntheses", {})
        if theme_syntheses:
            for name, ts in theme_syntheses.items():
                score = ts.get("anchoring_score", "?")
                npapers = ts.get("num_papers", "?")
                text = ts.get("synthesis", "")
                print(f"\n  [{name}]  score={score}  papers={npapers}")
                print(f"  {'-' * 46}")
                print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        else:
            print("(none)")

        print("\n" + "=" * 70)
        print("CROSS-THEME SYNTHESIS")
        print("=" * 70)
        cross = final_state.get("cross_theme_synthesis", "")
        print(cross if cross else "(none)")

        print("\n" + "=" * 70)
        print("GAP ANALYSIS")
        print("=" * 70)
        gaps = final_state.get("gap_analysis", "")
        print(gaps if gaps else "(none)")

        print("\n" + "=" * 70)

        # ---- Human Review Gate ----
        while True:
            choice = input("\nApprove results? [y/n/edit/quit]: ").strip().lower()
            if choice in ("y", "yes", ""):
                logger.info("User approved results — resuming to finalize output.")
                # Resume from interrupt — runs survey_scrub and completes
                final_state = app.invoke(None, config)
                print("\n" + "=" * 70)
                print("FINAL OUTPUT")
                print("=" * 70)
                print(final_state.get("final_output", "(empty)"))
                print("\n" + "=" * 70)
                break
            elif choice in ("n", "no"):
                print("Results discarded. Ask a new query or refine your question.")
                break
            elif choice == "edit":
                feedback = input("Enter feedback to refine the synthesis: ").strip()
                if feedback:
                    # Update state with feedback and resume
                    app.update_state(config, {"user_query": f"{query}\n\n[User feedback: {feedback}]"})
                    final_state = app.invoke(None, config)
                    print("\nFinal output with feedback applied:")
                    print(final_state.get("final_output", "(empty)"))
                    break
            elif choice == "quit":
                print("Exiting.")
                sys.exit(0)
            else:
                print("Choices: y (approve), n (discard), edit (provide feedback), quit")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except GraphRecursionError as e:
        print(f"\nGraph recursion limit reached: {e}")
    except Exception as e:
        logger.exception("Error during survey processing")
        print(f"\nError: {e}")

print(f"\nDemo finished. Knowledge graph saved to {GRAPH_PATH}")
