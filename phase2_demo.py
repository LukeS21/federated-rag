"""
Phase 2 Demo — Ingests a biomedical PDF and demonstrates hybrid search.
Usage:
  python phase2_demo.py data/test.pdf
If no argument, uses a built-in example corpus.
"""
import sys
from pathlib import Path

from src.ingestion.pdf_parser import PDFParser
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever


def main(pdf_path=None):
    # Set up retriever (project-local persistence; see projects/default/)
    project_chroma = Path(__file__).resolve().parent / "projects" / "default" / "chroma_data"
    chroma = ChromaClient("demo_public", persist_directory=str(project_chroma))
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma, bm25)

    if pdf_path:
        print(f"📄 Ingesting: {pdf_path}")
        parser = PDFParser()
        chunks = parser.parse(Path(pdf_path))
        print(f"✅ Extracted {len(chunks)} chunks (text + tables)")
        retriever.ingest(chunks)
        print("📚 Indexed into ChromaDB + BM25\n")
    else:
        # Use a tiny built-in corpus for quick test
        print("ℹ️  No PDF provided, using built-in example corpus.\n")
        chunks = [
            {"text": "TiO2 nanotubes improve osseointegration in rat tibiae.", "metadata": {"source": "demo"}},
            {"text": "IL-6 levels increase with titanium wear particles in murine model.", "metadata": {"source": "demo"}},
            {"text": "Ti-6Al-4V surface treatment with NaOH enhances cell adhesion.", "metadata": {"source": "demo"}},
            {"text": "Stem cell therapy combined with BMP-2 accelerated bone regeneration.", "metadata": {"source": "demo"}},
        ]
        retriever.ingest(chunks)

    print("🔍 Enter queries (type 'quit' to exit):")
    while True:
        try:
            q = input("\nQuery: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue

            results = retriever.query(q, n_results=3)
            print("\n── Fused Results (Hybrid) ──")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['text'][:120]}...")
                meta = res.get("metadata", {})
                if meta:
                    print(f"   📋 Source: {meta.get('source', 'unknown')}, Type: {meta.get('chunk_type', 'text')}")
        except (EOFError, KeyboardInterrupt):
            break

    print("\nSession ended. Index persists under projects/default/chroma_data/ for later queries.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(pdf_path=sys.argv[1])
    else:
        main()