"""
Corpus-level claim index (L0 cache) for publication-scale retrieval.

Pre-extracts atomic factual claims from all ingested papers and indexes
them in a dedicated ChromaDB collection (``corpus_claims``).  At query
time, the claim index is consulted before full extraction, acting as an
instant-coverage cache.

Claims are derived from pre-extracted entities — each entity's evidence
phrase becomes a claim indexed in the claim collection.

Usage::

    from src.retrieval.claim_index import ClaimIndex

    ci = ClaimIndex(base_persist_dir="projects/default/chroma_data")
    # At ingest time: after pre-extraction, index the claims
    ci.index_paper_claims("avery2022.pdf", entities_dict)
    # At query time:
    claims = ci.query("macrophage polarization titanium", n_results=50)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.retrieval.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

CLAIMS_COLLECTION = "corpus_claims"


class ClaimIndex:
    """Pre-extracted corpus‑wide claim index backed by ChromaDB.

    Each claim is stored as a document with metadata linking back to
    the source paper, entity category, and original evidence phrase.
    """

    def __init__(self, persist_dir: str | Path = "projects/default/chroma_data"):
        self._persist_dir = str(persist_dir)
        self._chroma = ChromaClient(
            collection_name=CLAIMS_COLLECTION,
            persist_directory=self._persist_dir,
        )

    def index_paper_claims(
        self,
        paper_id: str,
        entities: Dict[str, List[Dict]],
        dedup: bool = True,
    ) -> int:
        """Index claims derived from pre-extracted entities for one paper.

        Each entity category → entity becomes a claim document:
          "IL-6 is elevated in obese mice post-implantation"
        with metadata: source paper, category, evidence phrase.

        Args:
            paper_id: PDF filename (e.g., "avery2022.pdf").
            entities: Pre-extracted entity dict {category: [entity_obj, ...]}.
            dedup: If True, skip claims already in the index.

        Returns:
            Number of claims indexed.
        """
        if dedup:
            existing = self._get_paper_claims(paper_id)
            existing_texts = {c["text"] for c in existing}
        else:
            existing_texts = set()

        ids = []
        documents = []
        metadatas = []

        for category, entity_list in entities.items():
            for ei, entity in enumerate(entity_list):
                text = entity.get("evidence", "")
                if not text:
                    text = f"{entity.get('entity', '')}: {entity.get('context', '')}"[:500]
                text = text.strip()
                if not text or text in existing_texts:
                    continue

                cid = f"{paper_id}_claim_{category}_{ei}"
                ids.append(cid)
                documents.append(text)
                metadatas.append({
                    "source": paper_id,
                    "category": category,
                    "entity": entity.get("entity", ""),
                    "evidence_phrase": text[:500],
                    "claim_index": ei,
                })

        if not ids:
            return 0

        self._chroma.add_documents(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("ClaimIndex: indexed %d claims for %s", len(ids), paper_id)
        return len(ids)

    def query(
        self,
        query_text: str,
        n_results: int = 50,
        similarity_threshold: float | None = 1.5,
    ) -> List[Dict]:
        """Retrieve claims relevant to a query from the corpus index.

        Args:
            query_text: Search query.
            n_results: Max claims to return.
            similarity_threshold: L2 distance filter (lower = more similar).

        Returns:
            List of {"text": str, "metadata": dict} claim dicts.
        """
        try:
            resp = self._chroma.query(
                query_text=query_text,
                n_results=n_results,
                include_distances=True,
            )
            docs = resp.get("documents", [[]])[0]
            metas = resp.get("metadatas", [[]])[0]
            dists = resp.get("distances", [[]])[0]

            results = []
            for text, meta, dist in zip(docs, metas, dists):
                if similarity_threshold is not None and dist > similarity_threshold:
                    continue
                results.append({"text": text, "metadata": meta, "distance": dist})
            return results
        except Exception as e:
            logger.warning("ClaimIndex query failed: %s", e)
            return []

    def _get_paper_claims(self, paper_id: str) -> List[Dict]:
        """Get all claims already indexed for a paper (for dedup)."""
        try:
            data = self._chroma.collection.get(
                where={"source": paper_id},
                include=["documents"],
            )
            return [
                {"text": d}
                for d in data.get("documents", [])
            ]
        except Exception:
            return []

    def claim_count(self) -> int:
        """Return total number of claims in the index."""
        try:
            return self._chroma.collection.count()
        except Exception:
            return 0

    def paper_count(self) -> int:
        """Return number of unique papers with indexed claims."""
        try:
            data = self._chroma.collection.get(include=["metadatas"])
            sources = set()
            for meta in (data.get("metadatas", []) or []):
                if isinstance(meta, dict) and meta.get("source"):
                    sources.add(meta["source"])
            return len(sources)
        except Exception:
            return 0
