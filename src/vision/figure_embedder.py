"""
Figure-to-text embedding for cross-modal retrieval.

Embeds figure descriptions into ChromaDB alongside text chunks, using a
``chunk_type="figure"`` metadata tag.  Extends the existing ``HybridRetriever``
with an ``include_figures`` parameter for cross-modal queries.

Figure embeddings use the same ChromaDB collection as text chunks (shared
semantic space from the same ``all-MiniLM-L6-v2`` embedding model), enabling
fusion-based cross-modal retrieval without a separate index.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.retrieval.chroma_client import ChromaClient
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class FigureEmbedder:
    """Embed figure descriptions into ChromaDB and enable cross-modal retrieval.

    Usage::

        embedder = FigureEmbedder(retriever)
        embedder.embed(figures, pdf_name="avery2022.pdf")

        # Later: search across text + figures
        results = retriever.query(
            "IL-6 levels in obese mice",
            include_figures=True,
        )
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def embed(self, figures: List[Dict], pdf_source: str) -> int:
        """Embed figure descriptions into ChromaDB.

        Each figure becomes a document with:
          - text: the vision model's description (or caption fallback)
          - metadata: chunk_type="figure", source=pdf_name, page_no, bbox,
                      figure_index, caption (original)

        The BM25 index is NOT updated — figure descriptions are not part of the
        BM25 keyword corpus (they're AI-generated, not author-authored).

        Args:
            figures: List of figure dicts with ``description``, ``caption``,
                     ``page_no``, ``file_path``, etc.
            pdf_source: PDF filename for metadata.

        Returns:
            Number of figures embedded.
        """
        if not figures:
            return 0

        ids = []
        documents = []
        metadatas = []

        for fig_idx, fig in enumerate(figures):
            description = fig.get("description", "")
            caption = fig.get("caption", "")
            text = description if description else caption

            if not text.strip():
                continue

            fid = f"{pdf_source}_figure_{fig_idx}"
            ids.append(fid)
            documents.append(text)
            metadatas.append({
                "source": pdf_source,
                "chunk_type": "figure",
                "page_no": fig.get("page_no", 0),
                "bbox": json.dumps(fig.get("bbox", {})) if fig.get("bbox") else "",
                "figure_index": fig_idx,
                "caption": caption[:500],
                "width": fig.get("width", 0),
                "height": fig.get("height", 0),
            })

        if not ids:
            return 0

        # Add to ChromaDB only (not BM25 — AI-generated text doesn't belong
        # in the keyword index)
        self.retriever.chroma.add_documents(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Embedded %d figure descriptions into ChromaDB", len(ids))
        return len(ids)


def query_with_figures(
    retriever: HybridRetriever,
    query: str,
    n_results: int = 5,
    filter_references: bool = True,
    include_figures: bool = False,
    max_chunks: int = 20,
    **kwargs,
) -> List[Dict]:
    """Query the hybrid retriever with optional figure results.

    When ``include_figures=True``, figure descriptions are included in results
    alongside text chunks.  The ChromaDB query is run with a broader fetch to
    ensure figure results surface.

    Args:
        retriever: Configured HybridRetriever.
        query: Search query string.
        n_results: Number of results to return.
        filter_references: Exclude reference chunks.
        include_figures: Include figure description results.
        max_chunks: Cap on result count.
        **kwargs: Passed to ``retriever.query()``.

    Returns:
        List of ``{"text": str, "metadata": dict}`` results.
    """
    if not include_figures:
        return retriever.query(
            query,
            n_results=n_results,
            filter_references=filter_references,
            max_chunks=max_chunks,
            **kwargs,
        )

    # When including figures, we need to NOT filter by chunk_type in the
    # query call, then do our own filtering that preserves figures.
    results = retriever.query(
        query,
        n_results=n_results * 2,
        filter_references=False,  # we'll filter below
        max_chunks=max_chunks * 2,
        **kwargs,
    )

    # Separate figures and text/reference, then interleave
    figures = []
    texts = []
    for r in results:
        meta = r.get("metadata", {})
        if meta.get("chunk_type") == "figure":
            figures.append(r)
        elif not (filter_references and meta.get("chunk_type") == "reference"):
            texts.append(r)

    # Return text results first, then figures
    final = texts[:max_chunks - min(len(figures), max_chunks // 3)]
    final.extend(figures[: max_chunks // 3])
    return final[:max_chunks]


# Monkey-patch HybridRetriever to support include_figures
def _extend_hybrid_retriever() -> None:
    """Extend HybridRetriever.query to accept include_figures parameter."""
    _original_query = HybridRetriever.query

    def extended_query(
        self,
        query: str,
        n_results: int = 5,
        filter_references: bool = True,
        similarity_threshold: float | None = None,
        max_chunks: int = 20,
        include_figures: bool = False,
    ) -> List[Dict]:
        # ── Fetch raw results (no figure/reference filtering yet) ──
        raw_results = _original_query(
            self, query,
            n_results=n_results * (2 if include_figures else 1),
            filter_references=False,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks * (2 if include_figures else 1),
        )

        if include_figures:
            # Separate figures from text, then interleave
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
            return [
                r for r in raw_results
                if r.get("metadata", {}).get("chunk_type") not in ("figure",)
                and not (filter_references and r.get("metadata", {}).get("chunk_type") == "reference")
            ][:max_chunks]

    HybridRetriever.query = extended_query


# Apply the extension at module load time
_extend_hybrid_retriever()
