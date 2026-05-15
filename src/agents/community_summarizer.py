"""Community summarizer — generates LLM summaries for each detected community.

Uses a cheap model (OLLAMA_SMALL_MODEL / gemma4:e4b) to produce one-paragraph
summaries per research community. Results are cached to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm import get_chat_model, resolve_model
from src.graph.base_graph import BaseGraphStorage
from src.graph.community_detection import detect_communities, get_community_entities
from src.cache.llm_cache import get_cache
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

_DEFAULT_SUMMARY_CACHE = Path("projects/default/community_summaries.json")

_COMMUNITY_SUMMARY_SYSTEM = """You are a biomedical research analyst. Given a list of entities \
from a knowledge graph, write a 2-4 sentence paragraph describing the research area these \
entities represent. Focus on the scientific themes, methods, and findings. \
Use plain technical language. Output ONLY the paragraph, no other text."""


class CommunitySummarizer:
    """Generates LLM summaries for each KG community."""

    def __init__(
        self,
        model: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ):
        self.model = resolve_model(model or "small")
        self.cache_path = cache_path or _DEFAULT_SUMMARY_CACHE

    def summarize(
        self,
        graph_storage: BaseGraphStorage,
        *,
        community_data: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """Generate summaries for all communities.

        Args:
            graph_storage: The knowledge graph backend.
            community_data: Pre-computed community data from ``detect_communities()``.
                If None, runs detection first.
            force_recompute: Skip loading from cache.

        Returns:
            {community_id: {"name": "...", "summary": "...", "n_entities": N, "top_papers": [...], "entity_types": [...]}}
        """
        if community_data is None:
            community_data = detect_communities(graph_storage)

        if not force_recompute and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached:
                    conv = {int(k): v for k, v in cached.items()}
                    logger.info("Loaded %d community summaries from cache", len(conv))
                    return conv
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("Community summary cache corrupted, recomputing: %s", e)

        community_entities = get_community_entities(community_data, graph_storage)
        community_papers = _get_community_papers(community_data, graph_storage)

        summaries: Dict[int, Dict[str, Any]] = {}
        n_communities = community_data.get("n_communities", 0)

        for cid in range(n_communities):
            entities = community_entities.get(cid, [])
            papers = community_papers.get(cid, [])

            if not entities:
                summaries[cid] = {
                    "name": f"Community {cid}",
                    "summary": "No entities in this community.",
                    "n_entities": 0,
                    "top_papers": [],
                    "entity_types": [],
                }
                continue

            entity_text = self._format_entity_list(entities)
            summary_text = self._generate_summary(entity_text, cid)

            entity_types = sorted({e.get("node_type", "unknown") for e in entities})

            summaries[cid] = {
                "name": f"Community {cid}",
                "summary": summary_text,
                "n_entities": len(entities),
                "top_papers": papers[:10],
                "entity_types": entity_types,
            }

            logger.info("Community %d summary: %d entities, %d papers, types=%s",
                          cid, len(entities), len(papers), entity_types[:5])

        self._save_cache(summaries)

        return summaries

    def _format_entity_list(self, entities: List[Dict[str, Any]]) -> str:
        """Format entity list for the LLM prompt."""
        lines: List[str] = []
        for ent in entities[:50]:
            node_id = ent.get("node_id", "?")
            node_type = ent.get("node_type", "?")
            evidence = ent.get("evidence", "")[:120]
            source = ent.get("source_paper", "") or ""
            parts = [f"  - [{node_type}] {node_id}"]
            if evidence:
                parts.append(f"    evidence: {evidence}")
            if source:
                parts.append(f"    paper: {source}")
            lines.append("  ".join(parts[:1]))
        return "\n".join(lines)

    def _generate_summary(self, entity_text: str, cid: int) -> str:
        """Generate a one-paragraph summary via LLM."""
        if not entity_text.strip():
            return f"No description available for community {cid}."

        user_prompt = (
            f"Knowledge graph entities in this research cluster:\n\n"
            f"{entity_text}\n\n"
            f"Write a 2-4 sentence paragraph summarizing the research area."
        )

        cache = get_cache()
        cached = cache.get(_COMMUNITY_SUMMARY_SYSTEM, user_prompt, model=self.model)
        if cached is not None:
            return scrub_unicode(cached)

        try:
            llm = get_chat_model(self.model, temperature=0.0, max_tokens=300)
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=_COMMUNITY_SUMMARY_SYSTEM),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            result = scrub_unicode((response.content or "").strip())
            cache.set(_COMMUNITY_SUMMARY_SYSTEM, user_prompt, result, model=self.model)
            return result
        except Exception as e:
            logger.warning("Community %d summary generation failed: %s", cid, e)
            return f"Summary generation failed for community {cid}."

    def _save_cache(self, summaries: Dict[int, Dict[str, Any]]) -> None:
        """Persist community summaries to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in summaries.items()}, f, indent=2, ensure_ascii=False)
        logger.info("Community summaries cached to %s", self.cache_path)


def _get_community_papers(
    community_data: Dict[str, Any],
    graph_storage: BaseGraphStorage,
) -> Dict[int, List[str]]:
    """Extract source_paper from nodes in each community."""
    from src.graph.community_detection import get_community_papers as _gcp
    return _gcp(community_data, graph_storage)
