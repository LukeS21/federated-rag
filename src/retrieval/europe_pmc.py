"""
Thin Europe PMC REST API wrapper.

Europe PMC mirrors PubMed Central plus additional content (~33M publications).
Provides search, full-text XML (for OA papers), and metadata retrieval.
No API key required.  Rate limit: ~10 req/s (polite).

Key capabilities:
  - Search with OPEN_ACCESS:Y filter (returns only papers with full text)
  - Fetch full-text JATS XML (structured sections, figures, references)
  - Fetch article metadata (citations, cross-references)

Usage::

    from src.retrieval.europe_pmc import EuropePMCClient

    epmc = EuropePMCClient()
    results = epmc.search("titanium implant macrophage", oa_only=True, max_results=20)
    for r in results:
        xml = epmc.full_text_xml(r["pmcid"])
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCClient:
    """Lightweight Europe PMC REST client."""

    def __init__(self, min_interval: float = 0.15):
        self._min_interval = min_interval
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FederatedRAG/1.0 (mailto:researcher@example.com)",
            "Accept": "application/json",
        })

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        self._rate_limit()
        resp = self.session.get(url, params=params, timeout=45)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        oa_only: bool = True,
        max_results: int = 100,
        page_size: int = 100,
        sort: str = "CITED desc",
    ) -> List[Dict[str, Any]]:
        """Search Europe PMC.

        Args:
            query: Search terms (supports Europe PMC query syntax).
            oa_only: If True, add OPEN_ACCESS:Y filter (only full-text papers).
            max_results: Maximum results to return.
            page_size: Results per page (max 1000). Larger values = fewer requests.
            sort: Sort order. Default: most cited first.

        Returns:
            List of result dicts with keys: doi, pmid, pmcid, title, authorString,
            journalTitle, pubYear, abstractText, citedByCount, isOpenAccess,
            source, inEPMC, inPMC.
        """
        if oa_only:
            query = f"({query}) AND OPEN_ACCESS:Y"

        results: List[Dict[str, Any]] = []
        cursor = "*"
        remaining = max_results

        while remaining > 0:
            params = {
                "query": query,
                "resultType": "core",
                "pageSize": min(page_size, remaining),
                "format": "json",
                "sort": sort,
                "cursorMark": cursor,
            }
            resp = self._get(f"{EPMC_BASE}/search", params=params)
            data = resp.json()
            hit_list = data.get("resultList", {}).get("result", [])
            for hit in hit_list:
                results.append(self._normalise_result(hit))
            new_cursor = data.get("nextCursorMark")
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor
            remaining -= len(hit_list)

        return results[:max_results]

    @staticmethod
    def _normalise_result(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten Europe PMC result into a consistent dict."""
        return {
            "doi": raw.get("doi", ""),
            "pmid": raw.get("pmid", ""),
            "pmcid": raw.get("pmcid", ""),
            "title": (raw.get("title", "") or "").strip(),
            "authors": raw.get("authorString", ""),
            "journal": raw.get("journalTitle", ""),
            "year": int(raw.get("pubYear") or 0),
            "abstract": (raw.get("abstractText", "") or "").strip(),
            "cited_by": int(raw.get("citedByCount") or 0),
            "is_oa": raw.get("isOpenAccess", "") == "Y",
            "source": raw.get("source", ""),
            "has_full_text": raw.get("inPMC", "") == "Y" or raw.get("hasPDF", "") == "Y",
            "first_author": raw.get("authorString", "").split(",")[0].strip() if raw.get("authorString") else "",
        }

    # ------------------------------------------------------------ full text

    def full_text_xml(self, pmcid: str) -> Optional[str]:
        """Fetch full-text JATS XML for a paper by PMCID.

        Args:
            pmcid: PubMed Central ID (e.g. "PMC13059311").

        Returns:
            XML string, or None if full text is not available.
        """
        try:
            self._rate_limit()
            resp = self.session.get(
                f"{EPMC_BASE}/{pmcid}/fullTextXML",
                headers={"Accept": "text/xml, application/xml, */*"},
                timeout=45,
            )
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
            logger.debug("No full text for %s (status=%d, len=%d)",
                         pmcid, resp.status_code, len(resp.text))
            return None
        except requests.HTTPError as e:
            logger.debug("HTTP error fetching %s: %s", pmcid, e)
            return None
        except Exception as e:
            logger.warning("Error fetching %s: %s", pmcid, e)
            return None

    def full_text_xml_batch(self, pmcids: List[str]) -> Dict[str, Optional[str]]:
        """Fetch full text for multiple PMCIDs.

        Returns:
            Dict mapping PMCID → XML string (or None if unavailable).
        """
        results: Dict[str, Optional[str]] = {}
        for pmcid in pmcids:
            results[pmcid] = self.full_text_xml(pmcid)
        return results

    # ------------------------------------------------------------- metadata

    def article_meta(self, pmcid: str) -> Optional[Dict[str, Any]]:
        """Fetch richer metadata for a specific PMCID (using search by ID)."""
        results = self.search(f"PMCID:{pmcid}", oa_only=False, max_results=1)
        return results[0] if results else None
