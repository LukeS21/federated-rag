from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_ingest_and_query():
    # In-memory test
    chroma = ChromaClient("test_hybrid_public")
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma, bm25)

    chunks = [
        {"text": "TiO2 nanotubes improve osseointegration", "metadata": {"source": "a.pdf"}},
        {"text": "IL-6 levels increase with titanium wear particles", "metadata": {"source": "b.pdf"}},
        {"text": "Stem cells + BMP-2 accelerate bone repair", "metadata": {"source": "c.pdf"}},
    ]
    retriever.ingest(chunks)

    # Concept query: should retrieve TiO2 even though word "bone" not present
    results = retriever.query("bone growth")
    assert len(results) > 0
    texts = [r["text"] for r in results]
    assert any("TiO2" in t or "osseointegration" in t for t in texts)

    # Exact keyword query: "IL-6" must be top
    results2 = retriever.query("IL-6")
    assert "IL-6" in results2[0]["text"]