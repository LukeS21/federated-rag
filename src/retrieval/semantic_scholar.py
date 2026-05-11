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
        self._min_interval = 0.01 if self.api_key else 1.0  # 100/s with key, 1/s without

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

        self._rate_limit()
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }
        if year:
            params["year"] = year

        try:
            resp = requests.get(
                f"{S2_BASE}/paper/search",
                params=params,
                headers=self._headers(),
                timeout=30,
            )
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

        self._rate_limit()
        try:
            resp = requests.get(
                f"{S2_BASE}/paper/{paper_id}",
                params={"fields": fields},
                headers=self._headers(),
                timeout=30,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._normalize_paper(resp.json())
        except Exception as e:
            logger.error("Semantic Scholar paper fetch failed: %s", e)
            return None

    def search_by_doi(self, doi: str) -> Dict[str, Any] | None:
        """Look up a paper by DOI."""
        self._rate_limit()
        try:
            resp = requests.get(
                f"{S2_BASE}/paper/DOI:{doi}",
                params={"fields": "title,paperId,url,abstract,tldr,year,authors,externalIds,openAccessPdf,journal,citationCount"},
                headers=self._headers(),
                timeout=30,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._normalize_paper(resp.json())
        except Exception as e:
            logger.error("Semantic Scholar DOI lookup failed: %s", e)
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
            "tldr": (paper.get("tldr", {}) or {}).get("text", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
            "doi": external_ids.get("DOI", ""),
            "pmid": external_ids.get("PubMed", ""),
            "authors": [a.get("name", "") for a in authors[:10]],
            "journal": journal.get("name", ""),
            "citation_count": paper.get("citationCount"),
            "open_access_pdf": open_access.get("url", ""),
            "pub_types": paper.get("publicationTypes", []),
        }
