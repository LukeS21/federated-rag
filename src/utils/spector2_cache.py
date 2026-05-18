"""
SPECTER2 embedding cache — stores embeddings locally to avoid re-fetching
from the Semantic Scholar API. Eliminates ~84% of pipeline time on re-runs.

Cache is a JSON file mapping DOI → (s2_paper_id, embedding_vector, fetched_at).
DOI is the primary key because S2 paper_ids can change over time, but DOIs
are stable across API calls.

Usage::

    from src.utils.spector2_cache import Spector2Cache

    cache = Spector2Cache()
    emb = cache.get("10.1016/j.bioactmat.2021.01.030")
    if emb is None:
        s2_paper = s2.resolve_paper(doi, title)
        if s2_paper and s2_paper.get("paper_id"):
            emb = embeddings.get(s2_paper["paper_id"])
            cache.put(doi, s2_paper["paper_id"], emb)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path("projects/default/spector2_cache.json")


class Spector2Cache:
    """JSON-based SPECTER2 embedding cache with DOI-keyed lookup."""

    def __init__(self, cache_path: Path | str = CACHE_PATH):
        self._path = Path(cache_path)
        self._data: Dict[str, Dict] = self._load()

    def _load(self) -> Dict[str, Dict]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Could not load SPECTER2 cache: %s", e)
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get(self, doi: str) -> Optional[List[float]]:
        """Return a cached SPECTER2 embedding for *doi*, or None."""
        doi_key = doi.lower().strip()
        entry = self._data.get(doi_key)
        if entry and "embedding" in entry:
            emb = entry["embedding"]
            if emb is not None and len(emb) == 768:
                return emb
        return None

    def put(
        self,
        doi: str,
        s2_paper_id: str,
        embedding: Optional[List[float]],
    ) -> None:
        """Store a SPECTER2 embedding in the cache."""
        doi_key = doi.lower().strip()
        if not doi_key or embedding is None or len(embedding) != 768:
            return
        self._data[doi_key] = {
            "doi": doi,
            "s2_paper_id": s2_paper_id,
            "embedding": embedding,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def has(self, doi: str) -> bool:
        """Check if a DOI has a cached embedding."""
        doi_key = doi.lower().strip()
        entry = self._data.get(doi_key)
        return entry is not None and "embedding" in entry

    def find_similar(
        self,
        doi: str,
        *,
        min_score: float | None = None,
    ) -> list[dict]:
        """Return papers similar to *doi* by cosine similarity of SPECTER2 embeddings.

        Returns a list of ``{doi, s2_paper_id, score}`` dicts, sorted by
        descending similarity, filtering to ``score >= min_score``.
        Graceful degradation: returns ``[]`` if *doi* is not cached or
        no cached papers meet the threshold.

        *min_score* defaults to the ``SPECTOR2_SIMILARITY_THRESHOLD`` env
        var, or 0.6 if unset.
        """
        import numpy as np
        import os

        if min_score is None:
            min_score = float(os.getenv("SPECTOR2_SIMILARITY_THRESHOLD", "0.6"))

        emb = self.get(doi)
        if emb is None:
            return []

        query_vec = np.array(emb, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results: list[dict] = []
        for cached_doi, entry in self._data.items():
            if cached_doi == doi.lower().strip():
                continue
            cached_emb = entry.get("embedding")
            if cached_emb is None or len(cached_emb) != 768:
                continue
            candidate_vec = np.array(cached_emb, dtype=np.float32)
            candidate_norm = np.linalg.norm(candidate_vec)
            if candidate_norm == 0:
                continue
            score = float(np.dot(query_vec, candidate_vec) / (query_norm * candidate_norm))
            if score >= min_score:
                results.append({
                    "doi": entry.get("doi", cached_doi),
                    "s2_paper_id": entry.get("s2_paper_id", ""),
                    "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = len(self._data)
        with_embedding = sum(1 for e in self._data.values() if e.get("embedding"))
        return {"total_entries": total, "with_embedding": with_embedding}

    def flush(self) -> None:
        """Persist the current cache to disk."""
        self._save()
        cnt = self.stats()
        logger.info(
            "SPECTER2 cache flushed: %d entries, %d with embeddings",
            cnt["total_entries"], cnt["with_embedding"],
        )
