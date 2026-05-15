"""Progressive disclosure tiers for hierarchical knowledge access (Phase 11).

Three-tier architecture:
  - Tier 1 (system): High-level overview of all research communities
  - Tier 2 (community): Detailed summary + key entities + paper list for one community
  - Tier 3 (paper): Individual paper entities and evidence (existing retrieval)

Used by Survey Mode to present scalable, hierarchical access to the knowledge
graph without context-window overflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.graph.base_graph import BaseGraphStorage
from src.graph.community_detection import detect_communities, get_community_entities, get_community_papers

logger = logging.getLogger(__name__)


class ProgressiveDisclosure:
    """Three-tier hierarchical access to the knowledge graph.

    Tier 1: System-level overview (all communities)
    Tier 2: Community-level detail (one community)
    Tier 3: Paper-level evidence (individual paper entities)
    """

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        community_data: Optional[Dict[str, Any]] = None,
        community_summaries: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        self.graph_storage = graph_storage
        self.community_data = community_data or detect_communities(graph_storage)
        self.community_summaries = community_summaries or {}
        self._community_entities = get_community_entities(self.community_data, graph_storage)
        self._community_papers = get_community_papers(self.community_data, graph_storage)

    def get_system_overview(
        self,
        relevant_communities: Optional[List[int]] = None,
    ) -> str:
        """Tier 1: High-level overview of all (or relevant) communities.

        Returns a concise markdown summary listing each community's name,
        entity count, and one-line description.
        """
        if relevant_communities is None:
            cids = sorted(self._community_papers.keys(), key=str)
        elif not relevant_communities:
            return "No research communities detected in the knowledge graph."
        else:
            cids = relevant_communities

        n_nodes = self.community_data.get("n_nodes", 0)
        n_communities = self.community_data.get("n_communities", 0)
        lines = [f"# Research Communities Overview\n"]
        lines.append(f"{n_nodes} entities organized into {n_communities} research communities.\n")

        for cid in sorted(cids, key=str):
            info = self.community_summaries.get(cid, {})
            summary = info.get("summary", "No summary available.")
            papers = self._community_papers.get(cid, [])
            entities = self._community_entities.get(cid, [])
            entity_types = info.get("entity_types", [])

            lines.append(f"## Community {cid}")
            lines.append(f"- **Entities:** {len(entities)}")
            lines.append(f"- **Papers:** {len(papers)}")
            lines.append(f"- **Entity types:** {', '.join(entity_types[:8])}")
            lines.append(f"- **Summary:** {summary[:300]}")
            lines.append("")

        return "\n".join(lines)

    def get_community_detail(
        self,
        community_id: int,
    ) -> str:
        """Tier 2: Detailed view of one community.

        Includes the LLM summary, key entities with evidence, paper list,
        and entity-type breakdown.
        """
        info = self.community_summaries.get(community_id, {})
        summary = info.get("summary", "No summary available.")
        papers = self._community_papers.get(community_id, [])
        entities = self._community_entities.get(community_id, [])

        lines = [
            f"# Community {community_id}",
            "",
            "## Summary",
            summary,
            "",
            f"## Key Entities ({len(entities)})",
        ]

        top_entities = sorted(
            entities, key=lambda e: len(str(e.get("evidence", ""))), reverse=True
        )[:15]
        for ent in top_entities:
            node_id = ent.get("node_id", "?")
            node_type = ent.get("node_type", "?")
            evidence = ent.get("evidence", "")[:200]
            source = ent.get("source_paper", "") or ""
            lines.append(f"- **[{node_type}]** {node_id}")
            if evidence:
                lines.append(f"  > {evidence}")
            if source:
                lines.append(f"  *Source: {source}*")

        lines.append("")
        lines.append(f"## Papers ({len(papers)})")
        for paper in sorted(papers, key=str)[:20]:
            lines.append(f"- {paper}")

        return "\n".join(lines)

    def get_paper_entities(
        self,
        paper_id: str,
    ) -> List[Dict[str, Any]]:
        """Tier 3: All KG entities associated with a specific paper."""
        matching = []
        for cid, entities in self._community_entities.items():
            for ent in entities:
                source_paper = ent.get("source_paper", "")
                if paper_id in source_paper or source_paper == paper_id:
                    matching.append({**ent, "community_id": cid})
        return matching

    def build_disclosure_map(
        self,
        relevant_communities: Optional[List[int]] = None,
        query: str = "",
    ) -> Dict[str, Any]:
        """Build the full disclosure map for a query.

        Returns a structured dict with all three tiers for use in prompts
        or UI rendering.
        """
        cids = relevant_communities if relevant_communities is not None else sorted(self._community_papers.keys(), key=str)

        tier1 = self.get_system_overview(relevant_communities=cids)
        tier2: Dict[int, str] = {}
        for cid in cids[:6]:
            tier2[cid] = self.get_community_detail(cid)

        paper_map: Dict[str, List[str]] = {}
        for cid in cids:
            for paper in self._community_papers.get(cid, []):
                paper_map.setdefault(paper, []).append(str(cid))

        return {
            "query": query,
            "tier1_system_overview": tier1,
            "tier2_community_details": tier2,
            "tier3_paper_community_map": paper_map,
            "n_communities": len(cids),
            "n_papers": len(paper_map),
        }
