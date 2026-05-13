"""
Thin Unpaywall API wrapper.

Resolves legal OA PDF URLs from DOIs.  Free tier: email required for
rate limiting.  No API key needed.

Usage::

    from src.retrieval.unpaywall import UnpaywallClient

    uw = UnpaywallClient(email="user@example.edu")
    result = uw.lookup("10.1038/s41551-019-0450-z")
    if result["is_oa"]:
        print("Best OA PDF:", result["best_pdf_url"])
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


class UnpaywallClient:
    """Lightweight Unpaywall API client for OA PDF resolution.

    Unpaywall indexes ~47% of articles with at least one OA version.
    Returns best available OA PDF URL, including:
      - Gold/Hybrid OA from publishers
      - Green OA from institutional repositories
      - Preprint versions
    """

    def __init__(self, email: str | None = None):
        self.email = email or os.getenv("UNPAYWALL_EMAIL", "")
        self._last_request = 0.0
        self._min_interval = 0.5  # generous margin

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def lookup(self, doi: str) -> Dict[str, Any]:
        """Look up a DOI in Unpaywall and return OA availability info.

        Args:
            doi: DOI string (e.g., "10.1038/s41551-019-0450-z").

        Returns:
            Dict with: is_oa, best_pdf_url, oa_status, oa_locations,
                       title, journal, year, publisher.
            If lookup fails, returns {"is_oa": False, "doi": doi, "error": ...}.
        """
        if not doi:
            return {"is_oa": False, "doi": doi, "error": "No DOI provided"}

        self._rate_limit()
        doi_clean = doi.strip().replace("https://doi.org/", "")
        params: Dict[str, str] = {}
        if self.email:
            params["email"] = self.email

        try:
            resp = requests.get(
                f"{UNPAYWALL_BASE}/{doi_clean}",
                params=params,
                timeout=15,
            )
            if resp.status_code == 404:
                return {"is_oa": False, "doi": doi_clean, "error": "DOI not found"}
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Unpaywall lookup failed for %s: %s", doi_clean, e)
            return {"is_oa": False, "doi": doi_clean, "error": str(e)}

        best = data.get("best_oa_location") or {}

        return {
            "is_oa": data.get("is_oa", False),
            "doi": doi_clean,
            "title": data.get("title", ""),
            "journal": data.get("journal_name", ""),
            "year": data.get("year"),
            "publisher": data.get("publisher", ""),
            "oa_status": data.get("oa_status", "closed"),
            "best_pdf_url": best.get("url_for_pdf", ""),
            "best_landing_url": best.get("url_for_landing_page", ""),
            "oa_locations_count": len(data.get("oa_locations", [])),
        }

    def get_pdf_url(self, doi: str) -> Optional[str]:
        """Convenience: return just the best OA PDF URL, or None."""
        result = self.lookup(doi)
        url = result.get("best_pdf_url", "")
        return url if url else None
