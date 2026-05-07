"""
Graph Builder – constructs/updates the persistent knowledge graph
from structured extraction results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.graph.base_graph import BaseGraphStorage

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a knowledge graph from extracted entities and chunk evidence."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _parse_source_index(source: Any) -> Optional[int]:
        if isinstance(source, int):
            return source
        if isinstance(source, str):
            s = source.strip()
            if s.lower().startswith("chunk "):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        return None
            try:
                return int(s)
            except ValueError:
                return None
        return None

    def build(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        chunks: List[Dict[str, Any]],
        graph_storage: BaseGraphStorage,
    ) -> None:
        """Create/update nodes and edges in *graph_storage*."""

        chunk_meta_map: Dict[int, Dict[str, Any]] = {
            idx: (ch.get("metadata", {}) or {}) for idx, ch in enumerate(chunks)
        }

        # Index entities by chunk for co-occurrence edges.
        chunk_nodes: Dict[int, List[Tuple[str, str]]] = {}
        # Each entry is (node_id, evidence_phrase) so edge evidence can be entity-level if desired.

        # Step 1: nodes
        for category, entity_list in (entities or {}).items():
            if not entity_list:
                continue
            for ent in entity_list:
                if not isinstance(ent, dict):
                    continue
                entity_text = ent.get("entity")
                if not entity_text:
                    continue

                node_id = f"{category}:{entity_text}"
                props = {k: v for k, v in ent.items() if k not in ("entity", "evidence", "source")}

                source_idx = self._parse_source_index(ent.get("source"))
                if source_idx is not None and 0 <= source_idx < len(chunks):
                    source_paper = chunk_meta_map[source_idx].get("source", "unknown")
                    chunk_nodes.setdefault(source_idx, []).append((node_id, str(ent.get("evidence", "") or "")))
                else:
                    source_paper = "unknown"

                props["source_paper"] = source_paper
                props["evidence"] = ent.get("evidence", "")

                graph_storage.add_node(node_id=node_id, node_type=category, properties=props)

        # Step 2: co-occurrence edges
        now_iso = datetime.now(timezone.utc).isoformat()
        for idx, node_list in chunk_nodes.items():
            if len(node_list) < 2:
                continue

            source_paper = chunk_meta_map[idx].get("source", "unknown")
            chunk_text = str((chunks[idx] or {}).get("text", "") or "")

            # Link every pair of nodes in this chunk (directed both ways for easier querying in DiGraph)
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    a_id, a_ev = node_list[i]
                    b_id, b_ev = node_list[j]

                    evidence_phrase = chunk_text or a_ev or b_ev
                    edge_props = {
                        "extracted_at": now_iso,
                        "source_paper": source_paper,
                        "evidence_phrase": evidence_phrase,
                    }
                    graph_storage.add_edge(a_id, b_id, "co_occurs_with", edge_props)
                    graph_storage.add_edge(b_id, a_id, "co_occurs_with", edge_props)

        graph_storage.save()
        logger.info("Knowledge graph updated and saved.")


def build_graph(hybrid_retriever: Any, graph_storage: BaseGraphStorage):
    """Construct the full Deep Mode LangGraph state graph (README §9.2)."""

    # Local imports to avoid circular deps:
    # `src.graph.nodes` imports `GraphBuilder` from this module.
    from langgraph.graph import END, StateGraph

    from src.state import AgentState
    from src.graph.nodes import (
        anchoring_check_node,
        arbiter_node,
        category_discovery_node,
        critic_node,
        drafter_node,
        extraction_node,
        human_gate_node,
        input_router_node,
        kg_builder_node,
        retrieve_node,
        scrub_node,
    )

    workflow = StateGraph(AgentState)

    workflow.add_node("input_router", input_router_node)
    workflow.add_node("retrieve", lambda state: retrieve_node(state, hybrid_retriever))
    workflow.add_node("category_discovery", category_discovery_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("kg_builder", lambda state: kg_builder_node(state, graph_storage))
    workflow.add_node("drafter", drafter_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("arbiter", arbiter_node)
    # Second-pass arbiter uses identical logic but must be a distinct node id
    # to satisfy langgraph's single-outgoing-edge constraint for non-conditional nodes.
    workflow.add_node("arbiter_pass2", arbiter_node)
    workflow.add_node("anchoring_check_pass1", lambda state: anchoring_check_node(state, pass2_flag=False))
    workflow.add_node("anchoring_check_pass2", lambda state: anchoring_check_node(state, pass2_flag=True))
    workflow.add_node("scrub", scrub_node)
    workflow.add_node("human_gate", human_gate_node)

    workflow.set_entry_point("input_router")
    workflow.add_edge("input_router", "retrieve")
    workflow.add_edge("retrieve", "category_discovery")
    workflow.add_edge("category_discovery", "extraction")
    workflow.add_edge("extraction", "kg_builder")
    workflow.add_edge("kg_builder", "drafter")
    workflow.add_edge("drafter", "critic")

    def critic_router(state: AgentState) -> str:
        feedback = state.get("critic_feedback", "") or ""
        if feedback.startswith("NO_CRITIQUE"):
            return "anchoring_check_pass1"
        return "arbiter"

    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "arbiter": "arbiter",
            "anchoring_check_pass1": "anchoring_check_pass1",
        },
    )

    workflow.add_edge("arbiter", "anchoring_check_pass1")

    def anchoring_pass1_router(state: AgentState) -> str:
        score = float(state.get("anchoring_score", 0.0) or 0.0)
        if score >= 0.85:
            return "scrub"
        return "arbiter_pass2"

    workflow.add_conditional_edges(
        "anchoring_check_pass1",
        anchoring_pass1_router,
        {
            "scrub": "scrub",
            "arbiter_pass2": "arbiter_pass2",
        },
    )

    workflow.add_edge("arbiter_pass2", "anchoring_check_pass2")

    def anchoring_pass2_router(state: AgentState) -> str:
        score = float(state.get("anchoring_score", 0.0) or 0.0)
        if score >= 0.85:
            return "scrub"
        return "human_gate"

    workflow.add_conditional_edges(
        "anchoring_check_pass2",
        anchoring_pass2_router,
        {
            "scrub": "scrub",
            "human_gate": "human_gate",
        },
    )

    workflow.add_edge("human_gate", "scrub")
    workflow.add_edge("scrub", END)

    # Interrupt semantics in langgraph==0.2.0 are handled at compile-time.
    return workflow.compile(interrupt_before=["human_gate"])
