"""
Thin Semantic Scholar API wrapper.

Uses Semantic Scholar's free REST API for paper search and metadata retrieval.
Rate limit: 100 req/s with API key.  No key: 1 req/s.

Usage::

    from src.retrieval.semantic_scholar import SemanticScholarClient

    ss = SemanticScholarClient()
    results = ss.search("titanium implant macrophage polarization", limit=20)
    for r in results:
        print(r["title"], r.get("tldr", ""))
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient:
    """Lightweight Semantic Scholar API client for paper search and metadata."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("S2_API_KEY", "")
        self._last_request = 0.0
        # Free tier: 1 req/s (May 2026). Use 3.0s margin to avoid 429 errors.
        self._min_interval = 3.0

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with rate limiting and 429 retry handling."""
        max_retries = 3
        backoff_base = 10.0  # base backoff for 429s (seconds)
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.request(method, url, **kwargs)
                if resp.status_code == 429 and attempt < max_retries - 1:
                    retry_after = resp.headers.get("Retry-After", str(backoff_base * (2 ** attempt)))
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff_base * (2 ** attempt)
                    wait = max(wait, backoff_base * (2 ** attempt))
                    logger.debug("S2 429 rate-limit, waiting %.1fs (attempt %d/%d)",
                                 wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    self._last_request = time.monotonic()  # reset rate timer
                    continue
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def search(
        self,
        query: str,
        limit: int = 20,
        fields: str | None = None,
        year: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers matching *query*.

        Args:
            query: Search query string.
            limit: Maximum results to return (default 20, max 1000).
            fields: Comma-separated fields to return. Default: title,paperId,
                    url,abstract,tldr,year,authors,externalIds,publicationTypes,
                    openAccessPdf,journal,citationCount.
            year: Year filter (e.g., "2024" or "2023-2025").

        Returns:
            List of dicts with paper metadata.
        """
        if fields is None:
            fields = (
                "title,paperId,url,abstract,tldr,year,authors,externalIds,"
                "publicationTypes,openAccessPdf,journal,citationCount"
            )

        params: Dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }
        if year:
            params["year"] = year

        try:
            resp = self._request("GET", f"{S2_BASE}/paper/search",
                                 params=params, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            logger.info(
                "Semantic Scholar search: %d results for '%s'",
                len(papers), query[:80],
            )
            return [
                self._normalize_paper(p) for p in papers
            ]
        except Exception as e:
            logger.error("Semantic Scholar search failed: %s", e)
            return []

    def get_paper(self, paper_id: str, fields: str | None = None) -> Dict[str, Any] | None:
        """Fetch details for a single paper by Semantic Scholar paper ID."""
        if fields is None:
            fields = "title,paperId,url,abstract,tldr,year,authors,externalIds,openAccessPdf,journal,citationCount,references,citations"

        try:
            resp = self._request("GET",
                f"{S2_BASE}/paper/{paper_id}",
                params={"fields": fields}, headers=self._headers(), timeout=30,
            )
            if resp.status_code == 404:
                return None
            return self._normalize_paper(resp.json())
        except Exception as e:
            logger.error("Semantic Scholar paper fetch failed: %s", e)
            return None

    def search_by_doi(self, doi: str) -> Dict[str, Any] | None:
        """Look up a paper by DOI."""
        try:
            resp = self._request("GET",
                f"{S2_BASE}/paper/DOI:{doi}",
                params={"fields": "title,paperId,url,abstract,tldr,year,authors,externalIds,openAccessPdf,journal,citationCount,embedding"},
                headers=self._headers(), timeout=30,
            )
            if resp.status_code == 404:
                return None
            data = resp.json()
            if not data or not isinstance(data, dict):
                return None
            return self._normalize_paper(data)
        except Exception as e:
            logger.warning("Semantic Scholar DOI lookup failed (%s): %s", doi, e)
            return None

    def search_by_title(self, title: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search Semantic Scholar by title keywords (fallback for DOI misses).

        Uses first 10 words of the title as the query to find the closest match.
        """
        query = " ".join(title.split()[:10])
        return self.search(query, limit=limit, fields=(
            "title,paperId,url,abstract,tldr,year,authors,externalIds,"
            "openAccessPdf,journal,citationCount,embedding"
        ))

    def resolve_paper(
        self, doi: str, title: str = ""
    ) -> Dict[str, Any] | None:
        """Resolve a paper with DOI-first lookup, falling back to title search.

        Args:
            doi: Paper DOI.
            title: Paper title (used as fallback query).

        Returns:
            Normalized paper dict with paper_id and embedding if available,
            or None if the paper cannot be found in Semantic Scholar.
        """
        # Try DOI first
        paper = self.search_by_doi(doi)
        if paper and paper.get("paper_id"):
            return paper

        # Fallback: title search
        if title:
            logger.debug("DOI %s not in S2, trying title search...", doi)
            results = self.search_by_title(title, limit=3)
            for r in results:
                # Prefer exact DOI match, but accept any reasonable match
                if r.get("doi") == doi or r.get("title", "").lower() == title.lower():
                    return r
            # Accept first result as best-effort match
            if results:
                logger.debug("  Title fallback matched: %s", results[0].get("title", "")[:60])
                return results[0]

        return None

    def _normalize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Semantic Scholar paper dict to a standard format."""
        external_ids = paper.get("externalIds", {}) or {}
        authors = paper.get("authors", []) or []
        open_access = paper.get("openAccessPdf", {}) or {}
        journal = paper.get("journal", {}) or {}

        return {
            "paper_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "tldr": (paper.get("tldr") or {}).get("text", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
            "doi": external_ids.get("DOI", ""),
            "pmid": external_ids.get("PubMed", ""),
            "authors": [a.get("name", "") for a in authors[:10]],
            "journal": journal.get("name", ""),
            "citation_count": paper.get("citationCount"),
            "open_access_pdf": open_access.get("url", ""),
            "pub_types": paper.get("publicationTypes", []),
            "embedding": (paper.get("embedding") or {}).get("vector"),
        }

    # ----------------------------------------------------------------- SPECTER2

    def get_embeddings_batch(self, s2_paper_ids: List[str]) -> Dict[str, Optional[List[float]]]:
        """Fetch SPECTER2 embeddings for a batch of Semantic Scholar paper IDs.

        Args:
            s2_paper_ids: List of Semantic Scholar paper IDs.

        Returns:
            Dict mapping paper_id → 768-dim embedding vector (or None on failure).
        """
        if not s2_paper_ids:
            return {}

        try:
            resp = self._request("POST",
                f"{S2_BASE}/paper/batch",
                params={"fields": "paperId,embedding"},
                headers=self._headers(),
                json={"ids": s2_paper_ids},
                timeout=60,
            )
            resp.raise_for_status()
            result: Dict[str, Optional[List[float]]] = {}
            for paper in resp.json():
                pid = paper.get("paperId", "")
                emb = (paper.get("embedding", {}) or {}).get("vector")
                result[pid] = emb
            logger.info("Fetched %d SPECTER2 embeddings", len(result))
            return result
        except Exception as e:
            logger.error("SPECTER2 batch fetch failed: %s", e)
            return {}
