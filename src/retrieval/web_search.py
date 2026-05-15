"""
Discovery-only web search client (DuckDuckGo, no API key).

Provides topic discovery for the orchestrator agent — identifies emerging
concepts, research directions, and surface-level information from the open
web. Results are tagged ``source_type: "discovery"`` and are NEVER used as
evidence. All claims must be grounded in peer-reviewed papers with full text.

Uses DuckDuckGo Instant Answer API (free, no key) as primary source,
with optional ``duckduckgo_search`` library for richer results.

Usage::

    from src.retrieval.web_search import WebSearchClient

    ws = WebSearchClient()
    results = ws.search("osteoporosis biomaterial surface modification")
    for r in results:
        print(r["title"], r["url"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

DDG_API = "https://api.duckduckgo.com"
DDG_SEARCH = "https://lite.duckduckgo.com/lite"


class WebSearchClient:
    """Discovery-only web search via DuckDuckGo.

    All results are marked ``source_type: "discovery"`` — they guide topic
    exploration but are never admissible as evidence. The orchestrator uses
    these results to formulate structured queries for Europe PMC / Semantic
    Scholar, not to answer user questions directly.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FederatedRAG/1.0 (mailto:researcher@example.com)",
        })

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform a discovery web search.

        Args:
            query: Search query string.
            max_results: Maximum results to return.

        Returns:
            List of dicts with keys: title, url, snippet, source_type.
            All items have ``source_type == "discovery"``.
        """
        results: List[Dict[str, Any]] = []

        # Primary: use ddgs library if available (richer results)
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=min(max_results, 20)))
                for h in hits:
                    results.append({
                        "title": h.get("title", ""),
                        "url": h.get("href", ""),
                        "snippet": h.get("body", ""),
                        "source_type": "discovery",
                    })
                logger.info("Web search (ddgs): %d results for '%s'",
                            len(results), query[:80])
                return results[:max_results]
        except ImportError:
            logger.debug("duckduckgo_search not installed, falling back to DDG API")
        except Exception as e:
            logger.debug("ddgs search failed: %s, falling back to DDG API", e)

        # Fallback: DuckDuckGo Instant Answer API (related topics)
        try:
            resp = self.session.get(
                DDG_API,
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Abstract / definition
            abstract = data.get("AbstractText", "")
            abstract_url = data.get("AbstractURL", "")
            if abstract:
                results.append({
                    "title": data.get("Heading", ""),
                    "url": abstract_url,
                    "snippet": abstract[:300],
                    "source_type": "discovery",
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " ").title(),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "source_type": "discovery",
                    })

            logger.info("Web search (DDG API): %d results for '%s'",
                        len(results), query[:80])
        except Exception as e:
            logger.warning("DDG API search failed: %s", e)

        return results[:max_results]

    def discover_topics(
        self,
        seed_terms: List[str],
        results_per_term: int = 3,
    ) -> List[Dict[str, Any]]:
        """Discover emerging topics from a list of seed search terms.

        Args:
            seed_terms: List of search queries to run.
            results_per_term: Max results per term.

        Returns:
            Combined list of discovery results across all terms.
        """
        all_results: List[Dict[str, Any]] = []
        for term in seed_terms:
            results = self.search(term, max_results=results_per_term)
            for r in results:
                r["search_term"] = term
            all_results.extend(results)
        return all_results
