"""
Coverage diagnostic — compares Europe PMC full-text coverage against
Semantic Scholar search results for the same query.

Answers the question: "Of the papers Semantic Scholar finds on this topic,
what fraction have PMC full-text XML available for full ingestion?"

Usage::

    from src.retrieval.coverage import run_coverage_diagnostic

    stats = run_coverage_diagnostic("dental implant macrophage polarization")
    # stats["coverage_pct"] = 78.5  → "78.5% have PMC full text"
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

_TITLE_SIMILARITY_THRESHOLD = 0.6


def _normalize_for_match(text: str) -> str:
    """Normalize a string for fuzzy comparison: lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def _titles_overlap(title_a: str, title_b: str) -> float:
    """Return 0–1 similarity score between two paper titles."""
    a = _normalize_for_match(title_a)
    b = _normalize_for_match(title_b)
    if not a or not b:
        return 0.0
    # Both SequenceMatcher (character-level) and word-set overlap
    char_sim = SequenceMatcher(None, a, b).ratio()
    words_a: Set[str] = set(a.split())
    words_b: Set[str] = set(b.split())
    if not words_a or not words_b:
        return char_sim
    word_sim = len(words_a & words_b) / len(words_a | words_b)
    return max(char_sim, word_sim)


def _match_pmcids_to_s2_results(
    s2_results: List[Dict[str, Any]],
    epmc_results: List[Dict[str, Any]],
) -> None:
    """Annotate S2 results with PMC availability info (mutates S2 results in place).

    Matching strategy (in order of confidence):
      1. Exact DOI match (case-insensitive, stripped)
      2. Title fuzzy match (> 0.6 similarity)
    """
    # Build DOI → EPMC paper and title → EPMC paper maps
    epmc_by_doi: Dict[str, Dict[str, Any]] = {}
    epmc_titles: List[Dict[str, Any]] = []
    for pp in epmc_results:
        doi = (pp.get("doi") or "").lower().strip()
        if doi:
            epmc_by_doi[doi] = pp
        if pp.get("title"):
            epmc_titles.append(pp)

    for s2r in s2_results:
        s2r["in_pmc"] = False
        s2r["matched_pmcid"] = ""
        s2r["matched_doi"] = ""
        s2r["match_method"] = ""

        # --- Strategy 1: exact DOI match ---
        doi = (s2r.get("doi") or "").lower().strip()
        if doi and doi in epmc_by_doi:
            match = epmc_by_doi[doi]
            s2r["in_pmc"] = True
            s2r["matched_pmcid"] = match.get("pmcid", "")
            s2r["matched_doi"] = match.get("doi", "")
            s2r["match_method"] = "doi_exact"
            continue

        # --- Strategy 2: DOI prefix/suffix variants (some APIs include https://doi.org/) ---
        # Strip common DOI URL prefixes
        doi_clean = doi
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/"):
            if doi_clean.startswith(prefix):
                doi_clean = doi_clean[len(prefix):]
        doi_clean = doi_clean.strip("/")
        if doi_clean and doi_clean != doi and doi_clean in epmc_by_doi:
            match = epmc_by_doi[doi_clean]
            s2r["in_pmc"] = True
            s2r["matched_pmcid"] = match.get("pmcid", "")
            s2r["matched_doi"] = match.get("doi", "")
            s2r["match_method"] = "doi_clean"
            continue

        # --- Strategy 3: title fuzzy match ---
        s2_title = (s2r.get("title") or "").strip()
        if s2_title and epmc_titles:
            best_score = 0.0
            best_match = None
            for ep in epmc_titles:
                ep_title = (ep.get("title") or "").strip()
                score = _titles_overlap(s2_title, ep_title)
                if score > best_score:
                    best_score = score
                    best_match = ep
            if best_score >= _TITLE_SIMILARITY_THRESHOLD and best_match:
                s2r["in_pmc"] = True
                s2r["matched_pmcid"] = best_match.get("pmcid", "")
                s2r["matched_doi"] = best_match.get("doi", "")
                s2r["match_method"] = f"title_fuzzy_{best_score:.2f}"


def run_coverage_diagnostic(
    query: str,
    max_results: int = 20,
) -> Dict[str, Any]:
    """Search Europe PMC and Semantic Scholar with the same query and compare coverage.

    Args:
        query: The search query (used for both APIs).
        max_results: Max results per API.

    Returns:
        Dict with:
          - s2_total: number of S2 results
          - epmc_total: number of Europe PMC OA results
          - matched: number of S2 papers with PMC full text
          - coverage_pct: matched / s2_total * 100
          - s2_oa_count: how many S2 results have open access PDF
          - epmc_results: list of PMC result dicts (simplified)
          - s2_coverage_detail: per-S2-paper match status
          - query: the original query
    """
    from src.retrieval.europe_pmc import EuropePMCClient
    from src.retrieval.semantic_scholar import SemanticScholarClient

    epmc = EuropePMCClient()
    s2 = SemanticScholarClient()

    logger.info("Coverage diagnostic: searching both APIs for '%s'...", query[:80])

    # Search Europe PMC (OA only for full-text papers)
    epmc_results = epmc.search(query, oa_only=True, max_results=max_results)
    epmc_total = len(epmc_results)
    logger.info("  Europe PMC: %d OA papers found", epmc_total)
    for i, r in enumerate(epmc_results):
        logger.debug("    EPMC [%d] DOI=%s  title=%s",
                     i + 1, r.get("doi", "-"), (r.get("title", "") or "")[:80])

    # Search Semantic Scholar
    s2_results = s2.search(query, limit=max_results)
    s2_total = len(s2_results)
    s2_oa_count = sum(1 for r in s2_results if r.get("open_access_pdf"))
    logger.info("  Semantic Scholar: %d results (%d with OA PDF)", s2_total, s2_oa_count)
    for i, r in enumerate(s2_results):
        logger.debug("    S2 [%d] DOI=%s  title=%s",
                     i + 1, r.get("doi", "-"), (r.get("title", "") or "")[:80])

    # Match: which S2 results also exist in PMC?
    _match_pmcids_to_s2_results(s2_results, epmc_results)
    matched = sum(1 for r in s2_results if r.get("in_pmc"))
    coverage_pct = (matched / s2_total * 100) if s2_total else 0.0

    logger.info(
        "  Coverage: %d/%d papers (%.1f%%) have PMC full text",
        matched, s2_total, coverage_pct,
    )

    # Per-paper detail
    for i, r in enumerate(s2_results):
        method = r.get("match_method", "none")
        status = "PMC" if r.get("in_pmc") else "---"
        logger.debug(
            "  [%d] %s via %s | %s",
            i + 1, status, method, (r.get("title", "") or "")[:70],
        )

    return {
        "query": query,
        "s2_total": s2_total,
        "epmc_total": epmc_total,
        "s2_oa_count": s2_oa_count,
        "matched": matched,
        "coverage_pct": round(coverage_pct, 1),
        "epmc_results": [
            {
                "pmcid": r.get("pmcid", ""),
                "doi": r.get("doi", ""),
                "title": (r.get("title", "") or "")[:100],
                "year": r.get("year"),
                "journal": r.get("journal", ""),
            }
            for r in epmc_results
        ],
        "s2_coverage_detail": [
            {
                "title": (r.get("title", "") or "")[:100],
                "doi": r.get("doi", ""),
                "in_pmc": r.get("in_pmc", False),
                "matched_pmcid": r.get("matched_pmcid", ""),
                "match_method": r.get("match_method", ""),
            }
            for r in s2_results
        ],
    }
