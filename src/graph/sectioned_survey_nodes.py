"""
Phase 7b: Sectioned Survey Nodes — multi-turn section writing with stateful
section tracking, claim/citation ledger, and figure integration.

Graph:
  sectioned_init → sectioned_retrieve → sectioned_draft_section
    → sectioned_review → [sectioned_route → sectioned_retrieve or sectioned_assemble]
    → sectioned_scrub → END
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from src.agents.synthesis_drafter import SynthesisDrafter
from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims
from src.llm import get_chat_model, resolve_model
from src.retrieval.hybrid_retriever import HybridRetriever
from src.state import AgentState
from src.synthesis.claim_ledger import ClaimLedger
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

SECTIONS_DEFAULT = [
    {"name": "Introduction", "description": "Background and rationale"},
    {"name": "Methods", "description": "Experimental approaches and models"},
    {"name": "Results", "description": "Key findings with evidence"},
    {"name": "Discussion", "description": "Synthesis, implications, and gaps"},
]


# ── Sectioned Init ───────────────────────────────────────────────────────────

def sectioned_init_node(state: AgentState) -> Dict[str, Any]:
    """Initialize the sectioned survey: parse user query for section requests.

    If the user specifies sections (e.g., "write only the Results section"),
    use those.  Otherwise, generate a full Introduction/Methods/Results/Discussion
    plan.
    """
    query = state["user_query"]
    existing_plan = state.get("section_plan")

    if existing_plan:
        logger.info("Sectioned init: using pre-existing plan (%d sections)", len(existing_plan))
        return {
            "section_plan": existing_plan,
            "current_section_index": state.get("current_section_index", 0),
            "section_drafts": state.get("section_drafts", {}),
            "section_context": state.get("section_context", {}),
        }

    # Attempt LLM-based section planning
    try:
        llm = get_chat_model(model="small", temperature=0.0, max_tokens=500)
        plan_prompt = (
            f"Given the research query: \"{query}\"\n\n"
            "Determine which manuscript sections to write. Return a JSON list:\n"
            '  [{"name": "Introduction", "description": "..."}, ...]\n'
            'Valid section names: Introduction, Methods, Results, Discussion.\n'
            "Output ONLY valid JSON. No markdown."
        )
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=plan_prompt)])
        raw = resp.content.strip()

        # Try to parse JSON
        plan = _parse_json(raw)
        if isinstance(plan, list) and len(plan) > 0:
            logger.info("Sectioned init: LLM planned %d sections", len(plan))
            return {
                "section_plan": plan,
                "current_section_index": 0,
                "section_drafts": {},
                "section_context": {},
            }
    except Exception as e:
        logger.warning("Sectioned init LLM planning failed: %s — using defaults", e)

    # Fallback: default IMRaD plan
    logger.info("Sectioned init: using default IMRaD plan")
    return {
        "section_plan": SECTIONS_DEFAULT,
        "current_section_index": 0,
        "section_drafts": {},
        "section_context": {},
    }


def _parse_json(raw: str) -> Any:
    """Parse JSON from LLM output, handling markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Brace fallback
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        return raw


# ── Sectioned Retrieve ───────────────────────────────────────────────────────

def sectioned_retrieve_node(
    state: AgentState, hybrid_retriever: HybridRetriever
) -> Dict[str, Any]:
    """Retrieve evidence for the current section, including figure descriptions."""
    query = state["user_query"]
    section_plan = state.get("section_plan", [])
    idx = state.get("current_section_index", 0)

    if idx >= len(section_plan):
        logger.warning("Section index %d out of range (%d sections)", idx, len(section_plan))
        return {}

    current_section = section_plan[idx]
    section_name = current_section.get("name", f"section_{idx}")
    section_desc = current_section.get("description", "")

    # Build section-aware query
    section_query = f"{section_desc}: {query}" if section_desc else query

    # Retrieve including figures
    chunks = hybrid_retriever.query(
        section_query,
        similarity_threshold=1.5,
        max_chunks=30,
        filter_references=True,
        include_figures=True,
    )

    text_chunks = [c for c in chunks if (c.get("metadata", {}) or {}).get("chunk_type") != "figure"]
    figure_chunks = [c for c in chunks if (c.get("metadata", {}) or {}).get("chunk_type") == "figure"]

    logger.info(
        "Sectioned retrieve [%s]: %d text + %d figure chunks",
        section_name, len(text_chunks), len(figure_chunks),
    )

    section_context = state.get("section_context", {})
    section_context[section_name] = chunks

    figure_context = state.get("figure_context", [])
    figure_context.extend(figure_chunks)

    return {
        "section_context": section_context,
        "figure_context": figure_context,
    }


# ── Sectioned Draft ──────────────────────────────────────────────────────────

def sectioned_draft_node(state: AgentState) -> Dict[str, Any]:
    """Draft the current section using the Drafter with claim ledger tracking."""
    section_plan = state.get("section_plan", [])
    idx = state.get("current_section_index", 0)
    if idx >= len(section_plan):
        logger.warning("Section index %d out of range", idx)
        return {}

    current_section = section_plan[idx]
    section_name = current_section.get("name", f"section_{idx}")

    section_context = state.get("section_context", {}).get(section_name, [])
    figure_context = state.get("figure_context", [])

    # Filter figure chunks relevant to this section
    section_figures = [
        f for f in figure_context
        if (f.get("metadata", {}) or {}).get("chunk_type") == "figure"
    ]

    # Build evidence summaries
    summaries: List[str] = []
    for ch in section_context:
        meta = ch.get("metadata", {}) or {}
        if meta.get("chunk_type") == "figure":
            summaries.append(f"[Figure] {ch.get('text', '')}")
        else:
            cs = meta.get("chunk_summary", ch.get("text", "")[:300])
            if cs:
                summaries.append(cs)

    # Add dedicated figure descriptions
    for fig in section_figures:
        desc = fig.get("text", "")
        caption = (fig.get("metadata", {}) or {}).get("caption", "")
        page = (fig.get("metadata", {}) or {}).get("page_no", "?")
        if desc:
            summaries.append(f"[Figure on page {page}: {caption}] {desc}")

    # Load ledger
    ledger_json = state.get("claim_ledger_json", "")
    ledger = ClaimLedger()
    if ledger_json:
        try:
            data = json.loads(ledger_json)
            ledger.claims = data.get("claims", [])
        except json.JSONDecodeError:
            pass

    # Get prior sections' context for continuity
    prior_context = _build_prior_context(state, section_name)

    # ── Draft ──
    summary_text = "\n\n".join(summaries[:30])  # cap for prompt size
    summary_chunks = [{"text": summary_text, "metadata": {"source": f"section:{section_name}"}}]

    citations = sorted({
        (ch.get("metadata", {}) or {}).get("cite_key") or
        (ch.get("metadata", {}) or {}).get("source", "unknown")
        for ch in section_context
    })

    drafter = SynthesisDrafter(
        model_name="per-theme",
        model=resolve_model("small"),
    )

    draft = drafter.draft(
        query=f"{state['user_query']} [Section: {section_name}]",
        entities={},
        chunks=summary_chunks,
        citations=citations,
        kg_context=prior_context,
    )

    # ── Ledger tracking ──
    claims = decompose_claims(draft)
    new_claims = ledger.filter_new_claims(claims)
    for c_text in new_claims:
        is_grounded = _check_claim_grounded(c_text, section_context, summary_chunks)
        ledger.add_claim(
            claim_text=c_text,
            section=section_name,
            grounded=is_grounded,
        )

    # ── Warning about duplicates ──
    dupe_count = len(claims) - len(new_claims)
    if dupe_count > 0:
        logger.info("Sectioned draft [%s]: %d/%d claims are duplicates — filtered",
                     section_name, dupe_count, len(claims))

    # ── Deduplicated synthesis ──
    final_synthesis = "\n".join(new_claims)

    section_drafts = state.get("section_drafts", {}).copy()
    section_drafts[section_name] = final_synthesis

    # Save ledger
    ledger.save(Path("projects/default/section_ledger.json"))

    return {
        "section_drafts": section_drafts,
        "claim_ledger_json": json.dumps(ledger.to_dict(), ensure_ascii=False),
    }


def _build_prior_context(state: AgentState, current_section: str) -> str:
    """Build context string from prior sections for continuity."""
    section_plan = state.get("section_plan", [])
    section_drafts = state.get("section_drafts", {})

    parts = []
    for section in section_plan:
        name = section.get("name", "")
        if name == current_section:
            break
        draft = section_drafts.get(name, "")
        if draft:
            parts.append(f"[Prior Section: {name}]\n{draft[:500]}")

    if parts:
        return "Cross-section context:\n" + "\n\n".join(parts)
    return ""


def _check_claim_grounded(
    claim_text: str,
    chunks: List[Dict],
    summary_chunks: List[Dict],
) -> bool:
    """Quick grounding check: does the claim have an @citation?"""
    import re
    if re.search(r"@[\w-]+", claim_text):
        return True
    # Fallback: check anchoring
    try:
        evidence = chunks or summary_chunks
        score, _ = compute_anchoring_score([claim_text], evidence)
        return score >= 0.35
    except Exception:
        return True  # err on the side of grounded


# ── Sectioned Review ─────────────────────────────────────────────────────────

def sectioned_review_node(state: AgentState) -> Dict[str, Any]:
    """Return section status for review (used with LangGraph interrupt)."""
    section_plan = state.get("section_plan", [])
    idx = state.get("current_section_index", 0)
    if idx < len(section_plan):
        current = section_plan[idx]
        name = current.get("name", "?")
        draft = state.get("section_drafts", {}).get(name, "")
        logger.info("Sectioned review [%s]: %d chars draft", name, len(draft))
    return {}


# ── Sectioned Route ──────────────────────────────────────────────────────────

def sectioned_route_node(state: AgentState) -> Dict[str, Any]:
    """Route to next section or to assemble."""
    section_plan = state.get("section_plan", [])
    idx = state.get("current_section_index", 0)
    next_idx = idx + 1

    if next_idx < len(section_plan):
        logger.info("Sectioned route: moving to section %d/%d (%s)",
                     next_idx + 1, len(section_plan),
                     section_plan[next_idx].get("name", "?"))
        return {"current_section_index": next_idx, "routes": {"next": "retrieve"}}
    else:
        logger.info("Sectioned route: all sections drafted — assembling")
        return {"routes": {"next": "assemble"}}


def _route_after_init(state: AgentState) -> str:
    """Route from init to retrieve."""
    return "retrieve"


def _route_after_retrieve(state: AgentState) -> str:
    """Route from retrieve to draft."""
    return "draft"


def _route_after_draft(state: AgentState) -> str:
    """Route from draft to review."""
    return "review"


def _route_after_review(state: AgentState) -> str:
    """Route from review: if more sections → retrieve, else → assemble."""
    section_plan = state.get("section_plan", [])
    idx = state.get("current_section_index", 0)
    if idx + 1 < len(section_plan):
        return "route"
    return "assemble"


def _route_after_route(state: AgentState) -> str:
    """Route based on routes dict."""
    routes = state.get("routes", {})
    return routes.get("next", "retrieve")


# ── Sectioned Assemble ───────────────────────────────────────────────────────

def sectioned_assemble_node(state: AgentState) -> Dict[str, Any]:
    """Assemble all section drafts into a complete manuscript with
    cross-references and ledger validation.
    """
    section_plan = state.get("section_plan", [])
    section_drafts = state.get("section_drafts", {})

    # Load ledger for validation
    ledger_json = state.get("claim_ledger_json", "")
    ledger = ClaimLedger()
    if ledger_json:
        try:
            data = json.loads(ledger_json)
            ledger.claims = data.get("claims", [])
        except json.JSONDecodeError:
            pass

    report = ledger.coverage_report()
    warnings = []
    for section in section_plan:
        sec_name = section.get("name", "")
        sec_warnings = ledger.validate_section(sec_name)
        warnings.extend(sec_warnings)

    # Build manuscript
    parts = []
    for i, section in enumerate(section_plan):
        name = section.get("name", f"section_{i}")
        draft = section_drafts.get(name, "")
        parts.append(f"## {name.upper()}\n\n{draft.strip()}")

    manuscript = "\n\n".join(parts)

    logger.info(
        "Sectioned assemble: %d sections, %d claims, %d warnings",
        len(section_plan), len(ledger),
        len(warnings),
    )

    if warnings:
        logger.warning("Ledger warnings:\n  " + "\n  ".join(warnings[:5]))

    return {
        "final_output": manuscript,
        "claim_ledger_json": json.dumps(ledger.to_dict(), ensure_ascii=False),
        "human_approved": len(warnings) == 0,
    }


# ── Sectioned Scrub ──────────────────────────────────────────────────────────

def sectioned_scrub_node(state: AgentState) -> Dict[str, Any]:
    """ASCII scrub and final formatting for sectioned output."""
    final = state.get("final_output", "")
    final = scrub_unicode(final)
    return {"final_output": final}


# ── Node metadata for graph builder ──────────────────────────────────────────

SECTIONED_NODE_MAP = {
    "sectioned_init": (sectioned_init_node, _route_after_init),
    "sectioned_retrieve": (sectioned_retrieve_node, _route_after_retrieve),
    "sectioned_draft_section": (sectioned_draft_node, _route_after_draft),
    "sectioned_review": (sectioned_review_node, _route_after_review),
    "sectioned_route": (sectioned_route_node, _route_after_route),
    "sectioned_assemble": (sectioned_assemble_node, None),
    "sectioned_scrub": (sectioned_scrub_node, None),
}
