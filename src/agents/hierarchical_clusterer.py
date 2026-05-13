"""
Hierarchical two-level clustering for publication-scale corpora.

Level 1: Broad topic assignment via embedding similarity (reuses
         ``ThematicClusterer`` with sentence-transformers).
Level 2: Fine-grained theme discovery per broad topic via LLM
         (category‑discovery-style prompt on paper summaries).

Avoids O(N) context explosion at 100+ papers by screening at Level 1
before invoking the LLM at Level 2.

Usage::

    from src.agents.hierarchical_clusterer import HierarchicalClusterer
    from src.agents.thematic_clusterer import ThematicClusterer

    base = ThematicClusterer()
    hc = HierarchicalClusterer(base_clusterer=base)
    result = hc.cluster(papers, themes, n_subthemes=5)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.agents.thematic_clusterer import ThematicClusterer
from src.llm import resolve_model, get_chat_model

logger = logging.getLogger(__name__)

LLM_LEVEL2_PROMPT = """You are analyzing a collection of biomedical research papers \
grouped under the broad topic "{broad_topic}".

Below are paper summaries for {paper_count} papers in this topic. \
Identify {n_subthemes} to {n_subthemes_high} fine-grained sub-themes that capture \
distinct research directions, methods, or findings.

For each sub-theme:
- Give a short name (5-8 words)
- Describe what papers in this sub-theme investigate
- List 3-5 example findings or methods from the papers

Output as JSON:
{{"subthemes": [{{"name": "...", "description": "...", "examples": ["..."]}}]}}

Papers:
{paper_summaries}
"""


class HierarchicalClusterer:
    """Two-level clustering for large corpora.

    Level 1 uses embedding similarity (fast, deterministic) to group
    papers into broad topics.  Level 2 invokes an LLM per topic to
    discover fine-grained sub-themes.

    Papers per topic stay small enough that the LLM sees all summaries
    without context-window overflow, even at 1000+ papers.
    """

    MAX_PAPERS_PER_TOPIC_FOR_LLM = 50
    LEVEL1_SIMILARITY_THRESHOLD = 0.35

    def __init__(
        self,
        base_clusterer: ThematicClusterer | None = None,
        llm_model: str | None = None,
    ):
        """
        Args:
            base_clusterer: Pre-configured ThematicClusterer.  Created fresh
                           if None (uses default sentence-transformer).
            llm_model: Model name for Level 2 LLM sub-theme discovery.
                      Defaults to OLLAMA_SMALL_MODEL (gemma4:e4b).
        """
        self.base_clusterer = base_clusterer or ThematicClusterer(use_embeddings=True)
        self.llm_model = llm_model or resolve_model("small")

    def cluster(
        self,
        papers: Dict[str, str],
        themes: List[Dict[str, str]],
        n_subthemes: int = 5,
    ) -> Dict[str, Any]:
        """Run two-level hierarchical clustering.

        Args:
            papers: {paper_id: summary_text} mapping.
            themes: List of {{"name": "...", "description": "..."}} themes
                    discovered by query decomposition.
            n_subthemes: Target sub-themes per broad topic for Level 2.

        Returns:
            {
                "level1_clusters": {theme_name: [paper_ids]},
                "level2_subthemes": {  # per broad topic
                    theme_name: [{"name": ..., "description": ..., "examples": [...]}]
                },
                "paper_assignments": {paper_id: [subtheme_name, ...]},
                "unassigned": [paper_id, ...]
            }
        """
        # ── Level 1: Embedding-based broad topic assignment ──
        logger.info("Level 1 clustering: %d papers → %d topics",
                     len(papers), len(themes))
        l1_result = self.base_clusterer.cluster(papers, themes)
        level1_clusters = l1_result.get("clusters", {})
        unassigned = l1_result.get("unassigned", [])

        # ── Level 2: LLM sub-theme discovery per broad topic ──
        level2_subthemes: Dict[str, List[Dict]] = {}
        paper_to_subthemes: Dict[str, List[str]] = {}

        for topic, paper_ids in level1_clusters.items():
            if len(paper_ids) == 0:
                continue

            # Sample papers if over LLM limit
            if len(paper_ids) > self.MAX_PAPERS_PER_TOPIC_FOR_LLM:
                logger.info("Topic '%s': sampling %d/%d papers for LLM",
                             topic, self.MAX_PAPERS_PER_TOPIC_FOR_LLM, len(paper_ids))
                sampled_ids = sorted(paper_ids)[:self.MAX_PAPERS_PER_TOPIC_FOR_LLM]
            else:
                sampled_ids = paper_ids

            summaries_text = self._build_summary_block(
                {pid: papers.get(pid, "") for pid in sampled_ids}
            )

            subthemes = self._discover_subthemes(
                broad_topic=topic,
                paper_count=len(paper_ids),
                paper_summaries=summaries_text,
                n_subthemes=n_subthemes,
            )
            level2_subthemes[topic] = subthemes

            # Assign papers to subthemes via embedding similarity
            for pid in paper_ids:
                summary = papers.get(pid, "")
                assigned = self._assign_to_subthemes(pid, summary, subthemes)
                if assigned:
                    paper_to_subthemes[pid] = assigned

        return {
            "level1_clusters": level1_clusters,
            "level2_subthemes": level2_subthemes,
            "paper_assignments": paper_to_subthemes,
            "unassigned": unassigned,
        }

    def _build_summary_block(self, papers: Dict[str, str]) -> str:
        """Format paper summaries for LLM prompt."""
        lines = []
        for i, (pid, summary) in enumerate(papers.items()):
            text = summary[:300] if summary else f"Paper {pid}"
            lines.append(f"{i+1}. [{pid}] {text}")
        return "\n".join(lines)

    def _discover_subthemes(
        self,
        broad_topic: str,
        paper_count: int,
        paper_summaries: str,
        n_subthemes: int,
    ) -> List[Dict[str, Any]]:
        """Use LLM to discover fine-grained sub-themes within a broad topic."""
        try:
            llm = get_chat_model(self.llm_model, temperature=0.0)
            prompt = LLM_LEVEL2_PROMPT.format(
                broad_topic=broad_topic,
                paper_count=paper_count,
                n_subthemes=n_subthemes,
                n_subthemes_high=n_subthemes + 2,
                paper_summaries=paper_summaries,
            )
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))

            import json, re
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("subthemes", [])
        except Exception as e:
            logger.warning("Level 2 sub-theme discovery failed for '%s': %s",
                           broad_topic, e)
        return []

    def _assign_to_subthemes(
        self,
        paper_id: str,
        summary: str,
        subthemes: List[Dict[str, Any]],
    ) -> List[str]:
        """Assign a paper to relevant subthemes via embedding similarity."""
        if not subthemes or not summary:
            return []
        assigned = []
        try:
            sub_descs = [s.get("description", "") for s in subthemes]
            sub_names = [s.get("name", "") for s in subthemes]
            paper_emb = self.base_clusterer._encode(summary)
            for i, desc in enumerate(sub_descs):
                desc_emb = self.base_clusterer._encode(desc)
                sim = float(
                    (paper_emb @ desc_emb.T) /
                    (float(paper_emb.norm()) * float(desc_emb.norm()) + 1e-9)
                )
                if sim >= self.LEVEL1_SIMILARITY_THRESHOLD:
                    assigned.append(sub_names[i])
        except Exception as e:
            logger.debug("Sub-theme assignment failed for %s: %s", paper_id, e)
        return assigned
