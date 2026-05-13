"""
Thin PMC OA (PubMed Central Open Access) Service wrapper.

Resolves full-text PDF/XML URLs from PMIDs for papers in the PMC
open access subset (~4 million articles).  Free, no API key needed.

Usage::

    from src.retrieval.pmc_oa import PMCOAClient

    pmc = PMCOAClient()
    result = pmc.lookup("12345678")
    if result["has_oa"]:
        print("PDF:", result["pdf_url"])
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


class PMCOAClient:
    """Lightweight PMC OA Service client for full-text resolution."""

    def __init__(self, email: str | None = None):
        self.email = email or os.getenv("PUBMED_EMAIL", "")
        self._last_request = 0.0
        self._min_interval = 0.5

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def lookup(self, pmid: str) -> Dict[str, Any]:
        """Check if a PMID has an OA full-text version in PMC.

        Args:
            pmid: PubMed ID (as string, e.g., "12345678").

        Returns:
            Dict with: has_oa, pdf_url, tgz_url, format, license,
                       citation, error (on failure).
        """
        if not pmid:
            return {"has_oa": False, "pmid": pmid, "error": "No PMID provided"}

        self._rate_limit()
        pmid_clean = pmid.strip()
        params: Dict[str, str] = {"id": f"PMC{pmid_clean}", "format": "pdf"}
        if self.email:
            params["email"] = self.email

        # Try with PMID prefix first
        for id_fmt in [pmid_clean, f"PMC{pmid_clean}"]:
            params["id"] = id_fmt
            try:
                resp = requests.get(
                    PMC_OA_BASE,
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as e:
                logger.debug("PMC OA lookup failed for %s: %s", id_fmt, e)
                continue

            text = resp.text
            if "<error" in text.lower() or "no" in text.lower() and "record" in text.lower():
                continue

            # Parse the XML response for OA links
            links = re.findall(
                r'<link\s+format="([^"]+)"\s+href="([^"]+)"',
                text,
            )
            if not links:
                continue

            pdf_url = ""
            for fmt, href in links:
                if fmt.lower() == "pdf":
                    pdf_url = href
                    break

            return {
                "has_oa": True,
                "pmid": pmid_clean,
                "pdf_url": pdf_url,
                "links": [{"format": f, "url": h} for f, h in links],
            }

        return {"has_oa": False, "pmid": pmid_clean, "error": "No OA version in PMC"}

    def get_pdf_url(self, pmid: str) -> Optional[str]:
        """Convenience: return just the PDF URL, or None."""
        result = self.lookup(pmid)
        return result.get("pdf_url") if result.get("has_oa") else None
