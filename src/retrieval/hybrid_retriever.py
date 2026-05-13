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

    def query(
        self,
        query: str,
        n_results: int = 5,
        filter_references: bool = True,
        similarity_threshold: float | None = None,
        max_chunks: int = 20,
        include_figures: bool = False,
    ) -> List[Dict]:
        """Query both indexes and fuse results.

        When ``similarity_threshold`` is set, ChromaDB results are filtered
        by L2 distance before fusion (lower = more similar).  The final result
        list is capped at ``max_chunks`` to prevent context‑window overflow.

        When ``include_figures=True``, figure description chunks are interleaved
        with text results (up to 1/3 of max_chunks).  Figure chunks are retrieved
        via ChromaDB only (not BM25 — descriptions are AI‑generated).
        """
        base_n = n_results * (2 if include_figures else 1)
        base_max = max_chunks * (2 if include_figures else 1)
        fetch_n = max(base_n * 5, base_max * 2, 50)

        # Dense
        chroma_res = self.chroma.query(
            query_text=query,
            n_results=fetch_n,
            include_distances=similarity_threshold is not None,
        )
        dense_texts = chroma_res["documents"][0] if chroma_res["documents"] else []
        dense_metadatas = chroma_res["metadatas"][0] if chroma_res["metadatas"] else []
        dense_distances = chroma_res.get("distances", [[]])[0] if similarity_threshold is not None else []

        # Filter dense results by distance threshold (if provided)
        if similarity_threshold is not None and dense_distances:
            filtered_texts = []
            filtered_metadatas = []
            for t, m, d in zip(dense_texts, dense_metadatas, dense_distances):
                if d <= similarity_threshold:
                    filtered_texts.append(t)
                    filtered_metadatas.append(m)
            dense_texts = filtered_texts
            dense_metadatas = filtered_metadatas

        # Sparse
        sparse_texts = self.bm25.query(query, n_results=fetch_n)

        # Map text -> metadata for Chroma results
        text_meta = {t: m for t, m in zip(dense_texts, dense_metadatas)}

        # Fuse lists preserving order
        fused_texts = reciprocal_rank_fusion(dense_texts, sparse_texts)

        # Build raw result list with metadata where possible
        raw_results = []
        seen = set()
        for text in fused_texts:
            if text in seen:
                continue
            seen.add(text)
            meta = text_meta.get(text, {})
            raw_results.append({"text": text, "metadata": meta})
            if len(raw_results) >= base_max:
                break

        if include_figures:
            # Separate figures from text/reference, then interleave
            figures = []
            texts = []
            for r in raw_results:
                meta = r.get("metadata", {})
                if meta.get("chunk_type") == "figure":
                    figures.append(r)
                elif not (filter_references and meta.get("chunk_type") == "reference"):
                    texts.append(r)
            final = texts[:max_chunks - min(len(figures), max_chunks // 3)]
            final.extend(figures[:max_chunks // 3])
            return final[:max_chunks]
        else:
            # Filter out figure AND reference chunks
            results = []
            for r in raw_results:
                meta = r.get("metadata", {})
                if meta.get("chunk_type") in ("figure",):
                    continue
                if filter_references and meta.get("chunk_type") == "reference":
                    continue
                results.append(r)
                if len(results) >= max_chunks:
                    break
            return results