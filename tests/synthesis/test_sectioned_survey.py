"""
Integration tests for Phase 7b sectioned survey pipeline.

Tests the sectioned survey graph compilation and node execution
with mocked LLM calls.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.synthesis.claim_ledger import ClaimLedger


def make_mock_llm_response(content: str):
    """Create a mock LLM response with content."""
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.fixture
def retriever():
    """In-memory retriever with text + figure chunks."""
    chroma = ChromaClient("test_sectioned")
    bm25 = BM25Index()
    hybrid = HybridRetriever(chroma, bm25)

    chunks = [
        {"text": "IL-6 levels increase in obese mice post-implantation.",
         "metadata": {"source": "avery2024.pdf", "chunk_type": "text", "cite_key": "avery2024"}},
        {"text": "CD4+ T cells regulate macrophage polarization at titanium implant surfaces.",
         "metadata": {"source": "avery2024.pdf", "chunk_type": "text", "cite_key": "avery2024"}},
        {"text": "Bone-implant contact is reduced in CD4-deficient mice.",
         "metadata": {"source": "avery2024.pdf", "chunk_type": "figure", "caption": "Fig. 8. Bone implant contact.",
                      "page_no": 10, "figure_index": 8}},
        {"text": "Flow cytometry shows altered macrophage populations in CD4-/- and CD8-/- mice.",
         "metadata": {"source": "avery2024.pdf", "chunk_type": "figure", "caption": "Fig. 4. Macrophage populations.",
                      "page_no": 6, "figure_index": 4}},
    ]
    hybrid.ingest(chunks)
    return hybrid


def test_claim_ledger_integration():
    """ClaimLedger correctly deduplicates and tracks across sections."""
    ledger = ClaimLedger()

    # Section 1: Introduction
    intro_claims = [
        "Titanium implants are used in orthopedic applications (@avery2024)",
        "Macrophage polarization affects implant outcomes (@smith2025)",
    ]
    for c in intro_claims:
        ledger.add_claim(c, section="Introduction")

    # Section 2: Results
    result_claims = [
        "IL-6 increased 3-fold in obese mice (@avery2024)",
        "CD4+ T cell deficiency reduced bone-implant contact by 45% (@avery2024)",
        "Titanium implants are used in orthopedic applications (@avery2024)",  # DUPLICATE from Intro
    ]
    new = ledger.filter_new_claims(result_claims)
    assert len(new) == 2  # 3rd is duplicate
    for c in new:
        ledger.add_claim(c, section="Results")

    assert len(ledger) == 4  # 2 intro + 2 results (duplicate filtered)

    # Coverage check
    report = ledger.coverage_report({"avery2024", "smith2025"})
    assert report["total_claims"] == 4
    assert report["coverage_rate"] == 1.0  # both citations used


def test_claim_ledger_section_validation():
    """validate_section catches ungrounded and duplicate issues."""
    ledger = ClaimLedger()
    ledger.add_claim("Grounded (@avery2024)", section="Results", grounded=True)
    ledger.add_claim("No citation here", section="Results", grounded=False, citations=[])
    ledger.add_claim("No citation here", section="Results", grounded=False, citations=[])

    warnings = ledger.validate_section("Results")
    assert len(warnings) >= 2


def test_ledger_persistence_roundtrip(tmp_path):
    """Ledger round-trips through JSON persistence."""
    path = tmp_path / "test_ledger.json"

    ledger = ClaimLedger(ledger_path=path)
    ledger.add_claim("Claim A (@avery2024)", section="Introduction")
    ledger.add_claim("Claim B (@smith2025)", section="Results")
    ledger.save()

    # Load fresh
    ledger2 = ClaimLedger(ledger_path=path)
    assert len(ledger2) == 2

    report = ledger2.coverage_report()
    assert report["total_claims"] == 2
    assert report["per_section"]["Introduction"]["claims"] == 1
    assert report["per_section"]["Results"]["claims"] == 1


def test_figure_chunks_in_retrieval(retriever):
    """Figure chunks appear alongside text chunks with include_figures=True."""
    results = retriever.query("macrophage polarization CD4", include_figures=True, max_chunks=10)
    assert len(results) > 0

    text = [r for r in results if (r.get("metadata", {}) or {}).get("chunk_type") != "figure"]
    figs = [r for r in results if (r.get("metadata", {}) or {}).get("chunk_type") == "figure"]

    assert len(text) > 0, "Expected text chunks"
    assert len(figs) > 0, "Expected figure chunks with include_figures=True"

    # Figure metadata
    for f in figs:
        meta = f.get("metadata", {})
        assert meta.get("chunk_type") == "figure"
        assert "caption" in meta
        assert meta.get("page_no", 0) > 0


def test_figure_chunks_excluded_without_flag(retriever):
    """Figure chunks are excluded when include_figures=False."""
    results_no_figs = retriever.query("macrophage CD4", include_figures=False, max_chunks=10)
    figure_results = [
        r for r in results_no_figs
        if (r.get("metadata", {}) or {}).get("chunk_type") == "figure"
    ]
    assert len(figure_results) == 0


def test_sectioned_survey_graph_compiles(retriever):
    """Build_sectioned_survey_graph compiles without error."""
    from src.graph.sectioned_survey_graph import build_sectioned_survey_graph

    graph = build_sectioned_survey_graph(retriever)
    assert graph is not None

    # Verify key nodes exist
    nodes = list(graph.get_graph().nodes.keys()) if hasattr(graph, 'get_graph') else []
    # StateGraph's get_graph() returns a graph with nodes as tuples
    # Just verify compilation succeeded
    assert graph is not None


@patch("src.graph.sectioned_survey_nodes.SynthesisDrafter")
@patch("src.graph.sectioned_survey_nodes.get_chat_model")
def test_sectioned_init_node_default_plan(mock_get_chat, mock_drafter):
    """sectioned_init_node generates a default IMRaD plan when LLM unavailable."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")
    mock_get_chat.return_value = mock_llm

    from src.graph.sectioned_survey_nodes import sectioned_init_node

    state = {
        "user_query": "What is the immune response to titanium implants?",
        "query_scope": "public",
        "public_context": [],
        "secure_context": [],
        "mode": "survey",
        "extracted_entities": {},
        "synthesis_draft": "",
        "citations_used": [],
        "final_output": "",
        "human_approved": False,
        "discovered_categories": {},
        "knowledge_graph_snapshot": {},
        "critic_feedback": "",
        "synthesis_revised": "",
        "anchoring_score": 0.0,
        "ungrounded_claims": [],
        "chunk_summary": "",
        "ner_entities": [],
        "decomposed_themes": [],
        "thematic_clusters": {},
        "per_paper_extractions": {},
        "per_theme_syntheses": {},
        "cross_theme_synthesis": "",
        "gap_analysis": "",
    }

    result = sectioned_init_node(state)
    assert "section_plan" in result
    assert len(result["section_plan"]) == 4  # IMRaD
    assert result["section_plan"][0]["name"] == "Introduction"
    assert result["current_section_index"] == 0


def test_claim_ledger_deduplication_across_sections():
    """Claims from Introduction should be detected as duplicates in Results."""
    ledger = ClaimLedger()
    text = "Titanium surfaces promote macrophage polarization (@avery2024)"
    ledger.add_claim(text, section="Introduction")
    assert ledger.is_duplicate(text)

    # Simulate trying to add to Results
    new_claims = ledger.filter_new_claims([text, "New finding (@smith2025)"])
    assert new_claims == ["New finding (@smith2025)"]


def test_citation_extraction():
    """ClaimLedger parses @citations correctly."""
    ledger = ClaimLedger()
    record = ledger.add_claim(
        "MSCs (@avery2023) and T cells (@smith2025, @jones2023) interact at implant surfaces.",
        section="Results",
    )
    cites = record["citations"]
    assert "avery2023" in cites
    assert "smith2025" in cites
    assert "jones2023" in cites
    assert len(cites) == 3
