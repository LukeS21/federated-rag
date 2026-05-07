"""
Hybrid retriever that combines dense (ChromaDB) and sparse (BM25) results.
"""
from typing import List, Dict, Tuple
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index


def reciprocal_rank_fusion(
    dense_results: List[str],
    sparse_results: List[str],
    k: int = 60
) -> List[str]:
    """
    Fuse two ranked lists using RRF. Returns a single deduplicated ranked list.
    k is a constant that prevents very low ranks from having too much influence.
    """
    scores = {}
    # Score from dense
    for rank, doc in enumerate(dense_results, start=1):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    # Score from sparse
    for rank, doc in enumerate(sparse_results, start=1):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)

    # Sort by descending score and return text
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]


class HybridRetriever:
    """Combines a ChromaDB collection and a BM25 index for one corpus."""

    def __init__(self, chroma_client: ChromaClient, bm25_index: BM25Index):
        self.chroma = chroma_client
        self.bm25 = bm25_index

    def ingest(self, chunks: List[Dict]):
        """Add documents to both indexes. chunk must have 'text' and 'metadata'."""
        texts = [c["text"] for c in chunks]
        ids = [f"{c['metadata']['source']}_{c['metadata'].get('chunk_index', '')}_{i}" for i, c in enumerate(chunks)]
        metadatas = [c["metadata"] for c in chunks]

        # ChromaDB
        self.chroma.add_documents(ids=ids, documents=texts, metadatas=metadatas)

        # BM25 (only needs texts)
        self.bm25.add_documents(texts)

    def query(self, query: str, n_results: int = 5, filter_references: bool = True) -> List[Dict]:
        # Dense
        chroma_res = self.chroma.collection.query(query_texts=[query], n_results=n_results)
        dense_texts = chroma_res["documents"][0] if chroma_res["documents"] else []
        dense_metadatas = chroma_res["metadatas"][0] if chroma_res["metadatas"] else []

        # Sparse
        sparse_texts = self.bm25.query(query, n_results=n_results * 2)

        # Map text -> metadata for Chroma results
        text_meta = {t: m for t, m in zip(dense_texts, dense_metadatas)}

        # Fuse lists preserving order
        fused_texts = reciprocal_rank_fusion(dense_texts, sparse_texts)

        # Build final result list with metadata where possible
        results = []
        seen = set()
        for text in fused_texts:
            if text in seen:
                continue
            seen.add(text)
            meta = text_meta.get(text, {})

            # Skip reference chunks when filter_references is True
            if filter_references and meta.get("chunk_type") == "reference":
                continue

            results.append({"text": text, "metadata": meta})
            if len(results) >= n_results:
                break
        return results