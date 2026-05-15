"""
Gap resolver — closes the gap-analysis loop by parsing gap output into
structured search queries and feeding results into the Europe PMC pipeline.

The orchestrator (Phase 10) runs the survey pipeline, identifies research
gaps, then calls this module to:
  1. Parse gap-analysis text into structured search queries
  2. Search Europe PMC + Semantic Scholar for relevant papers
  3. Ingest new papers into ChromaDB + BM25 + KG
  4. Return the expanded evidence base for re-synthesis

Usage::

    from src.agents.gap_resolver import GapResolver

    resolver = GapResolver()
    new_papers = resolver.resolve_gaps(
        gap_analysis_text="No osteoblast data in obese Ti models...",
        max_papers_per_gap=5,
    )
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_PAPERS_PER_GAP = 5
MAX_GAPS = 3


def _is_false_positive_gap(text: str) -> bool:
    """Exclude common phrases that look like gaps but are actually findings."""
    false_positives = [
        r'\bno\s+significant\s+difference\b',
        r'\bno\s+statistically\s+significant\b',
        r'\bno\s+significant\s+change\b',
        r'\bno\s+difference\s+was\s+observed\b',
        r'\bno\s+effect\s+was\s+observed\b',
        r'\bno\s+evidence\s+of\s+(toxicity|harm|adverse)\b',
    ]
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in false_positives)


def _parse_gaps_to_queries(gap_text: str) -> List[Dict[str, str]]:
    """Parse unstructured gap-analysis text into structured search queries.

    Handles two common formats:
      1. Numbered/bulleted gaps: "1. Gap: ..." or "- Gap: ..."
      2. Sentence-level gaps: each sentence mentioning "no data", "missing",
         "lacking", "not studied", "unexplored" is treated as a gap.

    Includes false-positive filtering to exclude statements of null findings
    (e.g., "no significant difference was observed") which are not gaps.
    """
    if not gap_text or not gap_text.strip():
        return []

    queries: List[Dict[str, str]] = []

    # Split on numbered items (1., 2), 3-), bullet markers (-, •, *) at the
    # start of lines, and "Gap:" / "gap:" markers.
    # IMPORTANT: only split on hyphens that appear as bullet markers
    # (at start of line or after newline, followed by a space/letter),
    # not hyphens inside compound words like "IL-17A" or "Ti-6Al-4V".
    gap_blocks = re.split(
        r'(?:^|\n)\s*(?:\d+[\.\)]|[-•*])\s+',
        gap_text, flags=re.MULTILINE,
    )
    # Also try splitting on "Gap:" markers
    if len(gap_blocks) <= 1:
        gap_blocks = re.split(r'\bGap\s*:\s*', gap_text, flags=re.IGNORECASE)

    # Remove leading empty/whitespace block
    gap_blocks = [b.strip() for b in gap_blocks if b.strip()]

    for block in gap_blocks:
        block = block.strip()
        if not block or len(block) < 15:
            continue

        # Extract a concise query from the gap description
        sentences = re.split(r'(?<=[.!?])\s+', block)
        gap_title = sentences[0].strip().rstrip(".:,")
        gap_context = " ".join(sentences[1:]).strip() if len(sentences) > 1 else ""

        # Skip false positives (null findings, not real gaps)
        # Only check the first sentence — trailing text may contain non-gap statements
        if _is_false_positive_gap(gap_title):
            continue

        # Gap keyword check with word-boundary awareness
        gap_patterns = [
            r'\bno\s+[\w\s-]{0,40}?\bdata\b',   # "no osteoblast activity data" or "no data on X"
            r'\bno\s+study\b',                 # "no study has examined"
            r'\bmissing\b',                    # "missing data on X"
            r'\black(s|ing)\s+(data|evidence|study|research)\b',
            r'\bunexplored\b',
            r'\bunderstudied\b',
            r'\bnot\s+studied\b',
            r'\bunknown\b',                    # "the role of X is unknown"
            r'\bunclear\b',
            r'\binsufficient\s+(data|evidence|research)\b',
            r'\blimited\s+(data|evidence|understanding)\b',
            r'\bneeded\b',                     # "further research is needed"
            r'\bwarranted\b',                  # "further study is warranted"
            r'\bknowledge\s+gap\b',
            r'\bresearch\s+gap\b',
        ]
        is_gap = any(re.search(pat, block.lower()) for pat in gap_patterns)
        if not is_gap:
            continue

        # Build a search query: strip common gap-prefix phrases
        for prefix in ["gap:", "there is no", "no study has", "research is needed on",
                        "we lack", "there is a lack of", "insufficient data on",
                        "further research is needed on", "further study is needed on"]:
            if gap_title.lower().startswith(prefix):
                gap_title = gap_title[len(prefix):].strip()
                if gap_title.lower().startswith("the "):
                    gap_title = gap_title[4:].strip()

        query = gap_title[:200]
        queries.append({"query": query, "context": gap_context[:500]})

        if len(queries) >= MAX_GAPS:
            break

    return queries


class GapResolver:
    """Resolves research gaps by searching external APIs and ingesting results.

    The gap-analysis loop for Phase 10 feeding new literature into the
    KG at each cycle, enabling the background daemon to autonomously
    expand the knowledge base.
    """

    def __init__(self, max_papers_per_gap: int = MAX_PAPERS_PER_GAP):
        self.max_papers_per_gap = max_papers_per_gap

    def parse_gaps(self, gap_text: str) -> List[Dict[str, str]]:
        """Parse unstructured gap analysis text into structured queries."""
        return _parse_gaps_to_queries(gap_text)

    def resolve_gaps(
        self,
        gap_analysis_text: str,
        *,
        graph_storage: Any = None,
        ingest: bool = False,
    ) -> Dict[str, Any]:
        """Parse gap analysis, search for relevant papers, optionally ingest.

        Args:
            gap_analysis_text: Raw gap analysis output from the survey pipeline.
            graph_storage: Optional KG backend to update with new entities.
            ingest: If True, ingest new papers into ChromaDB + BM25 + KG.

        Returns:
            Dict with:
              - gaps_found: number of gaps parsed
              - queries: list of structured queries
              - new_papers: papers discovered across all gaps
              - total_ingested: papers ingested (if ingest=True)
        """
        from src.retrieval.europe_pmc import EuropePMCClient
        from src.ingestion.pmc_xml_parser import PMCXMLParser
        from src.utils.ingest_progress import IngestProgress

        gap_queries = _parse_gaps_to_queries(gap_analysis_text)

        result: Dict[str, Any] = {
            "gaps_found": len(gap_queries),
            "queries": gap_queries,
            "new_papers": [],
            "total_ingested": 0,
        }

        if not gap_queries:
            logger.info("GapResolver: no structured gaps found in analysis text")
            return result

        logger.info("GapResolver: %d gaps parsed", len(gap_queries))

        epmc = EuropePMCClient()
        parser = PMCXMLParser()
        progress = IngestProgress() if ingest else None
        seen_pmcids: set = set()

        for i, gq in enumerate(gap_queries):
            query = gq["query"]
            logger.info("  Gap %d/%d: searching '%s'", i + 1, len(gap_queries), query[:80])

            papers = epmc.search(query, oa_only=True,
                                 max_results=self.max_papers_per_gap)
            if not papers:
                logger.info("    No results")
                continue

            # Fetch full text for new papers
            pmcids = [p["pmcid"] for p in papers if p.get("pmcid")
                      and p["pmcid"] not in seen_pmcids]
            if not pmcids:
                continue

            xml_docs = epmc.full_text_xml_batch(pmcids)
            for p in papers:
                pmcid = p.get("pmcid", "")
                if not pmcid or pmcid in seen_pmcids:
                    continue
                xml = xml_docs.get(pmcid)
                if not xml:
                    continue

                chunks = parser.parse(xml, pmcid=pmcid, doi=p.get("doi", ""))
                paper_info = {
                    "pmcid": pmcid,
                    "doi": p.get("doi", ""),
                    "title": (p.get("title") or "")[:100],
                    "chunks": len(chunks),
                    "gap_index": i,
                    "gap_query": query,
                }
                result["new_papers"].append(paper_info)
                seen_pmcids.add(pmcid)

                # Ingest if requested
                if ingest and chunks and progress:
                    if not progress.is_completed(pmcid):
                        try:
                            from src.retrieval.chroma_client import ChromaClient
                            from src.retrieval.bm25_index import BM25Index
                            from src.retrieval.hybrid_retriever import HybridRetriever
                            from pathlib import Path

                            chroma = ChromaClient(
                                collection_name="public_corpus",
                                persist_directory="projects/default/chroma_data",
                            )
                            bm25 = BM25Index(
                                persist_dir=Path("projects/default/bm25_index"),
                            )
                            bm25.load()
                            retriever = HybridRetriever(
                                chroma_client=chroma, bm25_index=bm25,
                            )
                            retriever.ingest(chunks)
                            bm25.save()
                            progress.checkpoint(pmcid)
                            result["total_ingested"] += 1

                            # Wire PreExtractor for KG updates
                            if graph_storage:
                                try:
                                    from src.ingestion.pre_extractor import PreExtractor
                                    pre = PreExtractor()
                                    pre.extract_paper(
                                        paper_id=pmcid,
                                        chunks=chunks,
                                        graph_storage=graph_storage,
                                    )
                                    logger.debug("    PreExtractor ran for %s", pmcid)
                                except Exception as e:
                                    logger.debug("    PreExtractor failed: %s", e)

                            logger.info("    Ingested %s: %d chunks", pmcid, len(chunks))
                        except Exception as e:
                            logger.warning("    Ingest failed for %s: %s", pmcid, e)

            if ingest and progress:
                progress.finalize()

        logger.info("GapResolver: %d new papers across %d gaps",
                     len(result["new_papers"]), len(gap_queries))
        return result
