"""LangGraph node functions for the Deep Mode pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.state import AgentState
from src.agents.extraction_agent import ExtractionAgent
from src.agents.synthesis_drafter import SynthesisDrafter
from src.agents.socratic_critic import SocraticCritic
from src.agents.arbiter import Arbiter
from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims
from src.graph.base_graph import BaseGraphStorage
from src.graph.graph_builder import GraphBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.scrubber import final_scrub

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Node 1: Input Router
# ---------------------------------------------------------------------------
def input_router_node(state: AgentState) -> Dict[str, Any]:
    """Determines retrieval scope and sets mode.
    Since we have nothing to change, return a no‑op update.
    """

    return {"routes": state.get("routes", {})}   # write back existing routes (or empty dict)


# ---------------------------------------------------------------------------
#  Node 2: Hybrid Retriever
# ---------------------------------------------------------------------------
def retrieve_node(state: AgentState, hybrid_retriever: HybridRetriever) -> Dict[str, Any]:
    """Query the hybrid retriever, filtering references."""

    query = state["user_query"]
    scope = state["query_scope"]

    chunks = hybrid_retriever.query(query, n_results=10, filter_references=True)

    updates: Dict[str, Any] = {}
    if scope in ("public", "both"):
        updates["public_context"] = chunks
    if scope in ("secure", "both"):
        updates["secure_context"] = chunks
    return updates


# ---------------------------------------------------------------------------
#  Node 3: Category Discovery
# ---------------------------------------------------------------------------
def category_discovery_node(state: AgentState) -> Dict[str, Any]:
    """Run category discovery via the ExtractionAgent."""

    chunks = state.get("public_context") or state.get("secure_context", [])
    callback = state.get("callback")
    agent = ExtractionAgent(
        num_ctx=int(state.get("num_ctx", 8192) or 8192),
        client_kwargs=state.get("client_kwargs"),
        callback=callback,
    )
    categories = agent.discover_categories(chunks, state["user_query"])
    return {"discovered_categories": categories}


# ---------------------------------------------------------------------------
#  Node 4: Entity Extraction
# ---------------------------------------------------------------------------
def extraction_node(state: AgentState) -> Dict[str, Any]:
    """Run entity extraction with evidence grounding."""

    chunks = state.get("public_context") or state.get("secure_context", [])
    categories = state.get("discovered_categories", {})
    callback = state.get("callback")
    agent = ExtractionAgent(
        num_ctx=int(state.get("num_ctx", 8192) or 8192),
        client_kwargs=state.get("client_kwargs"),
        callback=callback,
    )
    entities = agent.extract_entities(chunks, categories, state["user_query"])
    return {"extracted_entities": entities}


# ---------------------------------------------------------------------------
#  Node 5: KG Builder
# ---------------------------------------------------------------------------
def kg_builder_node(state: AgentState, graph_storage: BaseGraphStorage) -> Dict[str, Any]:
    """Construct/update the persistent knowledge graph."""

    entities = state.get("extracted_entities", {})
    chunks = state.get("public_context") or state.get("secure_context", [])
    GraphBuilder().build(entities, chunks, graph_storage)

    node_ids = [
        f"{cat}:{ent['entity']}"
        for cat, ent_list in (entities or {}).items()
        for ent in (ent_list or [])
        if isinstance(ent, dict) and "entity" in ent
    ]
    subgraph = graph_storage.get_subgraph(node_ids, depth=1) if node_ids else {}
    return {"knowledge_graph_snapshot": subgraph}


# ---------------------------------------------------------------------------
#  Node 6: Drafter
# ---------------------------------------------------------------------------
def drafter_node(state: AgentState) -> Dict[str, Any]:
    """First synthesis draft."""

    entities = state.get("extracted_entities", {})
    chunks = state.get("public_context") or state.get("secure_context", [])
    citations = list({(ch.get("metadata", {}) or {}).get("source", "unknown") for ch in chunks})
    kg_snapshot = state.get("knowledge_graph_snapshot", {})
    callback = state.get("callback")

    drafter = SynthesisDrafter(
        num_ctx=int(state.get("num_ctx", 8192) or 8192),
        client_kwargs=state.get("client_kwargs"),
        callback=callback,
    )
    draft = drafter.draft(
        query=state["user_query"],
        entities=entities,
        chunks=chunks,
        citations=citations,
        kg_context=kg_snapshot,
    )
    return {"synthesis_draft": draft, "citations_used": citations}


# ---------------------------------------------------------------------------
#  Node 7: Socratic Critic
# ---------------------------------------------------------------------------
def critic_node(state: AgentState) -> Dict[str, Any]:
    """Critique the draft."""

    draft = state.get("synthesis_draft", "")
    chunks = state.get("public_context") or state.get("secure_context", [])
    entities = state.get("extracted_entities", {})
    callback = state.get("callback")
    critic = SocraticCritic(
        num_ctx=int(state.get("num_ctx", 8192) or 8192),
        client_kwargs=state.get("client_kwargs"),
        callback=callback,
    )
    feedback = critic.critique(draft, chunks, entities)
    return {"critic_feedback": feedback}


# ---------------------------------------------------------------------------
#  Node 8: Arbiter (Revision)
# ---------------------------------------------------------------------------
def arbiter_node(state: AgentState) -> Dict[str, Any]:
    """Revise the draft based on critique."""

    draft = state.get("synthesis_draft", "")
    critique = state.get("critic_feedback", "")
    chunks = state.get("public_context") or state.get("secure_context", [])
    callback = state.get("callback")
    arbiter = Arbiter(
        num_ctx=int(state.get("num_ctx", 8192) or 8192),
        client_kwargs=state.get("client_kwargs"),
        callback=callback,
    )
    revised = arbiter.revise(draft, critique, chunks)
    return {"synthesis_revised": revised}


# ---------------------------------------------------------------------------
#  Node 9: Evidence Anchoring Check
# ---------------------------------------------------------------------------
def anchoring_check_node(state: AgentState, pass2_flag: bool = False) -> Dict[str, Any]:
    """Programmatic anchoring score calculation."""

    _ = pass2_flag  # kept for parity with architecture; not required for computation
    draft = state.get("synthesis_revised") or state.get("synthesis_draft", "")
    claims = decompose_claims(draft)
    chunks = state.get("public_context") or state.get("secure_context", [])
    score, ungrounded = compute_anchoring_score(claims, chunks)
    return {"anchoring_score": score, "ungrounded_claims": ungrounded}


# ---------------------------------------------------------------------------
#  Node 10: Final Scrubber
# ---------------------------------------------------------------------------
def scrub_node(state: AgentState) -> Dict[str, Any]:
    """Apply final ASCII scrub to the chosen synthesis."""

    final_text = state.get("synthesis_revised") or state.get("synthesis_draft", "")
    return {"final_output": final_scrub(final_text)}


# ---------------------------------------------------------------------------
#  Human gate node (placeholder)
# ---------------------------------------------------------------------------
def human_gate_node(state: AgentState) -> Dict[str, Any]:
    """Placeholder for the human‑in‑the‑loop gate.
    The interrupt is placed before this node; if reached we return a dummy update.
    """

    return {"human_approved": state.get("human_approved", False)}

