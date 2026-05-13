"""
Figure-to-text embedding for cross-modal retrieval.

Embeds figure descriptions into ChromaDB alongside text chunks, using a
``chunk_type="figure"`` metadata tag.  The ``HybridRetriever.query()``
accepts ``include_figures=True`` for cross-modal queries.

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
