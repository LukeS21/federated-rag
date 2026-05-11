"""
Thin PubMed API wrapper (NCBI E-utilities).

Uses NCBI's free E-utilities REST API for search (esearch) and fetch (efetch).
No API key required for 3 req/s.  With API key: 10 req/s.

Rate limits are handled via a 350ms delay between requests (no-key default)
or 100ms with key.

Usage::

    from src.retrieval.pubmed import PubMedClient

    pm = PubMedClient()
    results = pm.search("titanium implant macrophage polarization", max_results=20)
    for r in results:
        print(r["title"], r["pmid"])

    # Get full metadata
    details = pm.fetch_details(["12345678", "23456789"])
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Lightweight NCBI E-utilities client for PubMed search and fetch."""

    def __init__(
        self,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "federated_rag",
    ):
        self.api_key = api_key or os.getenv("PUBMED_API_KEY", "")
        self.email = email or os.getenv("PUBMED_EMAIL", "")
        self.tool = tool
        self._last_request = 0.0
        self._min_interval = 0.1 if self.api_key else 0.35  # 10/s with key, ~3/s without

    def _rate_limit(self) -> None:
        """Enforce rate limit between requests."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def _build_params(self, **extra: str) -> Dict[str, str]:
        """Build standard query parameters."""
        params: Dict[str, str] = {"tool": self.tool}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        params.update(extra)
        return params

    def search(
        self,
        query: str,
        max_results: int = 20,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> List[Dict[str, Any]]:
        """Search PubMed for articles matching *query*.

        Args:
            query: PubMed search query (supports full PubMed syntax).
            max_results: Maximum number of results to return.
            retmax: Chunk size per request (max 100,000).
            sort: Sort order ("relevance", "pub_date", "first_author").

        Returns:
            List of dicts with: pmid, title, pub_date, source, doi, abstract.
        """
        self._rate_limit()

        # Step 1: esearch — get PMIDs
        search_params = self._build_params(
            db="pubmed",
            term=query,
            retmax=str(min(max_results, retmax)),
            sort=sort,
            retmode="json",
            usehistory="n",
        )
        try:
            resp = requests.get(
                f"{EUTILS_BASE}/esearch.fcgi",
                params=search_params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error("PubMed search failed: %s", e)
            return []

        if not pmids:
            logger.info("PubMed search returned 0 results for: %s", query[:80])
            return []

        # Step 2: efetch — get metadata for PMIDs
        self._rate_limit()
        return self.fetch_details(pmids)

    def fetch_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article metadata for a list of PMIDs.

        Returns:
            List of dicts with: pmid, title, pub_date, source, doi,
                                abstract, authors, pub_types.
        """
        if not pmids:
            return []

        self._rate_limit()
        fetch_params = self._build_params(
            db="pubmed",
            id=",".join(pmids),
            retmode="xml",
            rettype="abstract",
        )
        try:
            resp = requests.get(
                f"{EUTILS_BASE}/efetch.fcgi",
                params=fetch_params,
                timeout=60,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error("PubMed fetch failed: %s", e)
            return []

        return self._parse_efetch_xml(resp.text)

    def _parse_efetch_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubMed efetch XML into structured dicts."""
        results: List[Dict[str, Any]] = []
        try:
            # Strip namespace for easier parsing
            xml_clean = ""
            for line in xml_text.split("\n"):
                if "<PubmedArticle" in line:
                    xml_clean += line.replace("PubmedArticle>", "PubmedArticle>") + "\n"
            # Parse ignoring namespaces
            it = ET.iterparse(io.BytesIO(xml_text.encode("utf-8")), events=("end",))
            # Simpler: use regex-based extraction
            results = self._parse_efetch_regex(xml_text)
        except Exception as e:
            logger.error("PubMed XML parsing failed: %s", e)
        return results

    def _parse_efetch_regex(self, xml_text: str) -> List[Dict[str, Any]]:
        """Extract article data from efetch XML using simple XML parsing."""
        import re
        results: List[Dict[str, Any]] = []

        # Split into individual articles
        articles = re.split(r"<(?:PubmedArticle|PubmedBookArticle)", xml_text)[1:]

        for art_xml in articles:
            pmid = self._extract_tag(art_xml, "PMID")
            title = self._extract_tag(art_xml, "ArticleTitle")
            abstract = self._extract_tag(art_xml, "AbstractText") or ""

            # Multiple AbstractText elements may exist (with Label attributes)
            if not abstract:
                abstract_parts = re.findall(
                    r'<AbstractText[^>]*>(.*?)</AbstractText>', art_xml, re.DOTALL
                )
                abstract = " ".join(abstract_parts)

            pub_date = self._extract_pub_date(art_xml)
            source = self._extract_tag(art_xml, "Title") or ""  # Journal title

            # DOI
            doi = ""
            doi_match = re.search(r'<ELocationID[^>]*EIdType="doi"[^>]*>(.*?)</ELocationID>', art_xml)
            if doi_match:
                doi = doi_match.group(1).strip()

            # Authors
            authors = []
            author_blocks = re.findall(
                r'<Author[^>]*>(.*?)</Author>', art_xml, re.DOTALL
            )
            for ab in author_blocks:
                last = self._extract_tag(ab, "LastName") or ""
                fore = self._extract_tag(ab, "ForeName") or ""
                if last:
                    authors.append(f"{fore} {last}".strip())

            # Pub types
            pub_types = re.findall(
                r'<PublicationType[^>]*>(.*?)</PublicationType>', art_xml
            )

            if pmid and title:
                results.append({
                    "pmid": pmid.strip(),
                    "title": self._strip_html(title.strip()),
                    "abstract": self._strip_html(abstract.strip()),
                    "pub_date": pub_date,
                    "source": source.strip(),
                    "doi": doi.strip(),
                    "authors": authors[:10],  # cap at 10
                    "pub_types": [pt.strip() for pt in pub_types],
                })

        return results

    def _extract_tag(self, xml: str, tag: str) -> str:
        """Extract the text content of an XML tag."""
        import re
        match = re.search(
            rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", xml, re.DOTALL
        )
        return match.group(1).strip() if match else ""

    def _extract_pub_date(self, xml: str) -> str:
        """Extract publication date from XML."""
        import re
        year = self._extract_tag(xml, "Year") or ""
        month = self._extract_tag(xml, "Month") or ""
        day = self._extract_tag(xml, "Day") or ""

        if year:
            parts = [year]
            if month:
                parts.append(month)
            if day:
                parts.append(day)
            return "-".join(parts)

        # MedlineDate fallback
        medline = re.search(r"<MedlineDate>(.*?)</MedlineDate>", xml)
        if medline:
            return medline.group(1).strip()
        return ""

    @staticmethod
    def _strip_html(text: str) -> str:
        """Strip HTML tags and decode entities."""
        import re, html
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(text).strip()
