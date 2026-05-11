"""
Phase 7b: Sectioned Survey Graph — builds the LangGraph state machine for
multi-turn section writing with claim/citation ledger and figure integration.

Graph structure (8 nodes):
  sectioned_init → sectioned_retrieve → sectioned_draft_section
    → sectioned_review → [sectioned_route → sectioned_retrieve | sectioned_assemble]
    → sectioned_scrub → END

Human-in-the-loop: interrupt_before=["sectioned_review"] allows approval or
edit-with-feedback at each section boundary.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.graph.sectioned_survey_nodes import SECTIONED_NODE_MAP
from src.retrieval.hybrid_retriever import HybridRetriever
from src.state import AgentState

logger = logging.getLogger(__name__)


def build_sectioned_survey_graph(
    hybrid_retriever: HybridRetriever,
    checkpointer=None,
) -> StateGraph:
    """Build the sectioned survey LangGraph state machine.

    Args:
        hybrid_retriever: The retriever for evidence + figure lookup.
        checkpointer: Optional LangGraph checkpointer for human-in-the-loop.

    Returns:
        A compiled StateGraph.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    for node_name, (node_func, _) in SECTIONED_NODE_MAP.items():
        if node_name == "sectioned_retrieve":
            graph.add_node(
                node_name,
                lambda s, r=hybrid_retriever: SECTIONED_NODE_MAP[node_name][0](s, r),
            )
        else:
            graph.add_node(node_name, node_func)

    # ── Edges ──
    graph.add_edge("sectioned_init", "sectioned_retrieve")
    graph.add_edge("sectioned_retrieve", "sectioned_draft_section")
    graph.add_edge("sectioned_draft_section", "sectioned_review")

    # Conditional routing from review
    graph.add_conditional_edges(
        "sectioned_review",
        SECTIONED_NODE_MAP["sectioned_review"][1],
        {
            "route": "sectioned_route",
            "assemble": "sectioned_assemble",
        },
    )

    # Conditional routing from route
    graph.add_conditional_edges(
        "sectioned_route",
        SECTIONED_NODE_MAP["sectioned_route"][1],
        {
            "retrieve": "sectioned_retrieve",
            "assemble": "sectioned_assemble",
        },
    )

    graph.add_edge("sectioned_assemble", "sectioned_scrub")
    graph.add_edge("sectioned_scrub", END)

    # Entry point
    graph.set_entry_point("sectioned_init")

    # Compile
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Sectioned survey graph compiled: 8 nodes")
    return compiled
