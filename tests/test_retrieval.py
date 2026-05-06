from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index

def test_chroma_add_and_search():
    client = ChromaClient("test_public")
    client.add_documents(["doc1", "doc2"], ["Titanium implants show osseointegration.", "IL-6 cytokine levels increase."])
    results = client.query("titanium implant")
    assert len(results['documents'][0]) > 0

def test_bm25():
    idx = BM25Index()
    docs = ["Titanium dental implants", "Stem cell therapy", "IL-6 and TNF alpha"]
    idx.add_documents(docs)
    res = idx.query("titanium implant")
    assert "Titanium dental implants" in res
