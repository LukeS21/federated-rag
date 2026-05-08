#!/usr/bin/env python3
"""Phase 3 Demo – Deep Mode with streaming token output & single execution.

Adds:
- Real-time node progress via LangGraph streaming
- Token streaming (prints tokens as Ollama generates them)
- 10-minute timeout per LLM call
- Reduced context window (8192) to prevent memory deadlocks
- Single execution (no second `invoke` after streaming)
"""

import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)  # ensure .env values take precedence over shell env

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.errors import GraphRecursionError

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.pre_summarizer import PreSummarizer
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.networkx_json_storage import NetworkXJSONStorage
from src.graph.graph_builder import build_graph
from src.unicode_map import scrub_unicode
from src.scrubber import final_scrub
from src.streaming_handler import TokenStreamHandler

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
PDF_PATH = Path("data/test.pdf")
PROJECT_DIR = Path("projects/default")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
BM25_CORPUS_PATH = str(PROJECT_DIR / "bm25_corpus.json")
GRAPH_PATH = str(PROJECT_DIR / "project_graph.json")

# Context window & timeout
NUM_CTX = 16384
LLM_TIMEOUT = 600  # seconds (10 minutes)
CLIENT_KWARGS = {"timeout": LLM_TIMEOUT}

# Rough token estimate threshold for warnings (4 chars ≈ 1 token)
TOKEN_WARN_CHARS = int(NUM_CTX * 0.8 * 4)  # 80% of 8192 tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase3_demo")


def _chroma_doc_count(chroma_client: ChromaClient) -> int:
    try:
        return int(chroma_client.collection.count())
    except Exception:
        return 0


def _ensure_bm25_loaded_from_disk_or_chroma(
    *,
    chroma_client: ChromaClient,
    bm25: BM25Index,
    bm25_corpus_path: str,
) -> None:
    p = Path(bm25_corpus_path)
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
#  Helper: token‑limit warning
# ---------------------------------------------------------------------------
def warn_if_large(prompt_text: str, label: str) -> None:
    char_len = len(prompt_text)
    if char_len > TOKEN_WARN_CHARS:
        logger.warning(
            "%s is %d chars (~%d tokens) – close to 80%% of context limit.",
            label,
            char_len,
            char_len // 4,
        )


# ---------------------------------------------------------------------------
#  Step 1: Ingest PDF
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 3 DEEP MODE DEMO (with streaming & timeout)")
print("=" * 70)

pdf_parser = PDFParser()
chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
bm25 = BM25Index()
retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)

existing_docs = _chroma_doc_count(chroma)

if existing_docs == 0:
    logger.info(f"Ingesting PDF: {PDF_PATH}")
    chunks = pdf_parser.parse(PDF_PATH)
    for ch in chunks:
        ch["text"] = scrub_unicode(ch["text"])

    # Pre-summarize chunks at ingest time (one-time LLM cost per document).
    # Stored in chunk metadata so query-time Summarize node can skip the LLM call.
    pre_summarizer = PreSummarizer()
    chunks = pre_summarizer.summarize_all(chunks)

    retriever.ingest(chunks)
    bm25_corpus = [ch["text"] for ch in chunks]
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_corpus, f)
    logger.info(f"Ingested {len(chunks)} chunks (pre-summarized).")
else:
    logger.info(f"Using existing index with ~{existing_docs} documents.")
    _ensure_bm25_loaded_from_disk_or_chroma(
        chroma_client=chroma,
        bm25=bm25,
        bm25_corpus_path=BM25_CORPUS_PATH,
    )

# ---------------------------------------------------------------------------
#  Step 2: Knowledge graph
# ---------------------------------------------------------------------------
graph_storage = NetworkXJSONStorage(GRAPH_PATH)
logger.info(f"Knowledge graph loaded from {GRAPH_PATH}")

# ---------------------------------------------------------------------------
#  Step 3: Build LangGraph app
# ---------------------------------------------------------------------------
app = build_graph(retriever, graph_storage)
logger.info("LangGraph Deep Mode pipeline compiled.")

# ---------------------------------------------------------------------------
#  Step 4: Interactive query loop with streaming
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print('Deep Mode ready. Enter queries (or "quit" to exit).')
print(f"Context window: {NUM_CTX} tokens | LLM timeout: {LLM_TIMEOUT}s")
print("=" * 70)

while True:
    try:
        query = input("\nQuery: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        warn_if_large(query, "User query")

        initial_state = {
            "user_query": query,
            "query_scope": "public",
            "mode": "deep",
            "num_ctx": NUM_CTX,
            "client_kwargs": CLIENT_KWARGS,
        }

        logger.info(f"Running deep pipeline for: {query}")

        final_state: dict = dict(initial_state)
        config = {"configurable": {"thread_id": f"demo-{int(time.time())}"}}
        interrupt_count = 0
        current_input = initial_state

        # Loop handles up to two interrupts: sci_ner (category review), human_gate (final review)
        while True:
            stream_input = current_input if interrupt_count == 0 else None
            stream_ended_normally = True
            last_time = time.time()  # reset per stream cycle (avoids counting user think time)

            for event in app.stream(stream_input, config):
                for node_name, output_dict in event.items():
                    if node_name == "__interrupt__":
                        stream_ended_normally = False
                        continue

                    now = time.time()
                    elapsed = now - last_time
                    logger.info(f"▸ {node_name} ({elapsed:.1f}s)")
                    last_time = now

                    if isinstance(output_dict, dict) and output_dict:
                        final_state.update(output_dict)

            if stream_ended_normally:
                break  # Graph completed, no more interrupts

            # ── Handle interrupt ──
            interrupt_count += 1
            if interrupt_count == 1:
                # Category checkpoint (before sci_ner)
                cats = final_state.get("discovered_categories", {})
                print("\n" + "-" * 50)
                print("CATEGORY REVIEW (edit or press Enter to continue)")
                print("-" * 50)
                if isinstance(cats, dict) and cats.get("discovered_categories"):
                    for c in cats["discovered_categories"]:
                        print(f"  - {c.get('name', '?')}: {c.get('description', '')[:80]}")
                print("-" * 50)
                action = input("Press Enter to continue, or type new categories (JSON): ").strip()
                if action:
                    try:
                        new_cats = json.loads(action)
                        app.update_state(config, {"discovered_categories": new_cats})
                        final_state["discovered_categories"] = new_cats
                        logger.info("Categories updated by user.")
                    except json.JSONDecodeError:
                        print("Invalid JSON, using discovered categories.")
                logger.info("Resuming pipeline…")
            else:
                # Human gate checkpoint (final review)
                logger.info("⏸ Pipeline paused at human gate. Reviewing results below.")

        # ---- Display Results ----
        print("\n" + "-" * 50)
        print("DISCOVERED CATEGORIES")
        print("-" * 50)
        categories = final_state.get("discovered_categories", {})
        if categories:
            print(json.dumps(categories, indent=2, ensure_ascii=True))
        else:
            print("(none)")

        print("\n" + "-" * 50)
        print("EXTRACTED ENTITIES")
        print("-" * 50)
        entities = final_state.get("extracted_entities", {})
        if entities:
            print(json.dumps(entities, indent=2, ensure_ascii=True))
        else:
            print("(none)")

        print("\n" + "-" * 50)
        print("ANCHORING SCORE")
        print("-" * 50)
        score = final_state.get("anchoring_score")
        if score is not None:
            print(f"Score: {score:.2f}")
        else:
            print("(not computed)")

        ungrounded = final_state.get("ungrounded_claims", [])
        if ungrounded:
            print(f"Ungrounded claims ({len(ungrounded)}):")
            for uc in ungrounded:
                if isinstance(uc, dict):
                    claim = uc.get("claim", str(uc))
                    sim = uc.get("similarity", "")
                    evidence = uc.get("best_evidence_sentence", "")
                    print(f" - Claim: {claim}")
                    if sim:
                        print(f"   Similarity: {sim}")
                    if evidence:
                        print(f"   Best evidence: {evidence[:100]}...")
                else:
                    print(f" - {str(uc)[:120]}")
        else:
            print("All claims grounded.")

        print("\n" + "-" * 50)
        print("FINAL SYNTHESIS (scrubbed)")
        print("-" * 50)
        final_output = final_state.get("final_output") or ""
        if not final_output:
            final_output = final_scrub(
                final_state.get("synthesis_revised") or final_state.get("synthesis_draft") or ""
            )
        print(final_output if final_output else "(no synthesis produced)")

        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except GraphRecursionError as e:
        print(f"\nGraph recursion limit reached: {e}")
    except Exception as e:
        logger.exception("Error during query processing")
        print(f"\nError: {e}")

print("\nDemo finished. Knowledge graph saved to", GRAPH_PATH)