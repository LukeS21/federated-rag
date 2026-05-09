"""Integration tests for Survey Mode graph (Phase 4)."""

from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
#  _fit_summaries_to_context
# ---------------------------------------------------------------------------
def test_fit_summaries_empty():
    from src.graph.survey_nodes import _fit_summaries_to_context
    assert _fit_summaries_to_context([], 16384) == []


def test_fit_summaries_all_fit():
    from src.graph.survey_nodes import _fit_summaries_to_context
    summaries = ["short summary"] * 5
    result = _fit_summaries_to_context(summaries, 16384)
    assert len(result) == 5


def test_fit_summaries_capped_by_context():
    from src.graph.survey_nodes import _fit_summaries_to_context
    summaries = ["x" * 1000] * 100  # ~250 tokens each = 25000 tokens total
    result = _fit_summaries_to_context(summaries, num_ctx=4000, max_ratio=0.7, overhead_tokens=500)
    assert len(result) < 100
    assert len(result) >= 1


def test_fit_summaries_respects_overhead():
    from src.graph.survey_nodes import _fit_summaries_to_context
    summaries = ["x" * 1000] * 100
    low_oh = _fit_summaries_to_context(summaries, num_ctx=4000, max_ratio=0.7, overhead_tokens=500)
    high_oh = _fit_summaries_to_context(summaries, num_ctx=4000, max_ratio=0.7, overhead_tokens=2000)
    assert len(low_oh) > len(high_oh)


def test_fit_summaries_large_context_returns_all():
    from src.graph.survey_nodes import _fit_summaries_to_context
    summaries = ["summary %d" % i for i in range(50)]
    result = _fit_summaries_to_context(summaries, num_ctx=128000)
    assert len(result) == 50


# ---------------------------------------------------------------------------
#  _run_debate_for_theme — conditional Critic (EGSR pattern)
# ---------------------------------------------------------------------------
def _make_chunk(text="evidence text", source="paper1", chunk_summary="summary text"):
    return {"text": text, "metadata": {"source": source, "chunk_summary": chunk_summary}}


def _make_state_chunks():
    return [_make_chunk()]


@patch("src.graph.survey_nodes.compute_anchoring_score")
@patch("src.graph.survey_nodes.decompose_claims")
@patch("src.graph.survey_nodes.load_theme_synthesis", return_value=None)
@patch("src.graph.survey_nodes._get_drafter")
@patch("src.graph.survey_nodes._get_critic")
@patch("src.graph.survey_nodes._get_arbiter")
def test_run_debate_single_paper_formats_entities(
    mock_get_arbiter, mock_get_critic, mock_get_drafter,
    mock_load_theme, mock_decompose, mock_anchoring,
):
    """Single-paper themes format entities directly, no Drafter call."""
    from src.graph.survey_nodes import _run_debate_for_theme, _clear_agent_caches
    _clear_agent_caches()

    mock_decompose.return_value = ["Key findings from paper1"]
    mock_anchoring.return_value = (0.4, [])

    result = _run_debate_for_theme(
        "theme1", _make_state_chunks(), {}, "test query",
        num_ctx=16384, client_kwargs=None, num_papers=1,
    )
    assert result["num_papers"] == 1
    assert "Key findings from paper1" in result["synthesis"]
    assert "theme1" in result["synthesis"]
    mock_get_drafter.assert_not_called()


@patch("src.graph.survey_nodes.compute_anchoring_score")
@patch("src.graph.survey_nodes.decompose_claims")
@patch("src.graph.survey_nodes.load_theme_synthesis", return_value=None)
@patch("src.graph.survey_nodes._get_drafter")
@patch("src.graph.survey_nodes._get_critic")
@patch("src.graph.survey_nodes._get_arbiter")
def test_run_debate_well_grounded_skips_critic(
    mock_get_arbiter, mock_get_critic, mock_get_drafter,
    mock_load_theme, mock_decompose, mock_anchoring,
):
    """Draft with anchoring >= 0.85 skips Critic/Arbiter (well-grounded)."""
    from src.graph.survey_nodes import _run_debate_for_theme, _clear_agent_caches
    _clear_agent_caches()

    mock_drafter = MagicMock()
    mock_drafter.draft.return_value = "Well-grounded draft."
    mock_get_drafter.return_value = mock_drafter
    mock_decompose.return_value = ["Well-grounded draft."]
    mock_anchoring.return_value = (0.92, [])

    result = _run_debate_for_theme(
        "theme1", _make_state_chunks(), {}, "test query",
        num_ctx=16384, client_kwargs=None, num_papers=3,
    )
    assert result["anchoring_score"] == 0.92
    assert result["synthesis"] == "Well-grounded draft."
    mock_get_critic.assert_not_called()
    mock_get_arbiter.assert_not_called()


@patch("src.graph.survey_nodes.compute_anchoring_score")
@patch("src.graph.survey_nodes.decompose_claims")
@patch("src.graph.survey_nodes.load_theme_synthesis", return_value=None)
@patch("src.graph.survey_nodes._get_drafter")
@patch("src.graph.survey_nodes._get_critic")
@patch("src.graph.survey_nodes._get_arbiter")
def test_run_debate_moderately_grounded_skips_critic(
    mock_get_arbiter, mock_get_critic, mock_get_drafter,
    mock_load_theme, mock_decompose, mock_anchoring,
):
    """Draft with anchoring >= 0.50 but < 0.85 skips Critic (moderately grounded)."""
    from src.graph.survey_nodes import _run_debate_for_theme, _clear_agent_caches
    _clear_agent_caches()

    mock_drafter = MagicMock()
    mock_drafter.draft.return_value = "Moderately grounded draft."
    mock_get_drafter.return_value = mock_drafter
    mock_decompose.return_value = ["Moderately grounded draft."]
    score = 0.55
    mock_anchoring.return_value = (score, [{"claim": "test", "best_evidence_sentence": "ev", "similarity": score}])

    result = _run_debate_for_theme(
        "theme1", _make_state_chunks(), {}, "test query",
        num_ctx=16384, client_kwargs=None, num_papers=3,
    )
    assert result["anchoring_score"] == score
    assert result["synthesis"] == "Moderately grounded draft."
    mock_get_critic.assert_not_called()
    mock_get_arbiter.assert_not_called()


@patch("src.graph.survey_nodes.compute_anchoring_score")
@patch("src.graph.survey_nodes.decompose_claims")
@patch("src.graph.survey_nodes.load_theme_synthesis", return_value=None)
@patch("src.graph.survey_nodes._get_drafter")
@patch("src.graph.survey_nodes._get_critic")
@patch("src.graph.survey_nodes._get_arbiter")
def test_run_debate_poorly_grounded_invokes_critic(
    mock_get_arbiter, mock_get_critic, mock_get_drafter,
    mock_load_theme, mock_decompose, mock_anchoring,
):
    """Draft with anchoring < CONDITIONAL_CRITIC_THRESHOLD invokes Critic."""
    from src.graph.survey_nodes import _run_debate_for_theme, _clear_agent_caches
    _clear_agent_caches()

    mock_drafter = MagicMock()
    mock_drafter.draft.return_value = "Poorly grounded draft."
    mock_get_drafter.return_value = mock_drafter
    mock_decompose.return_value = ["Poorly grounded draft."]

    poor_score = 0.25
    mock_anchoring.return_value = (poor_score, [{"claim": "test", "best_evidence_sentence": "", "similarity": poor_score}])

    mock_critic = MagicMock()
    mock_critic.critique.return_value = "NO_CRITIQUE: All claims are evidence-grounded."
    mock_get_critic.return_value = mock_critic
    mock_get_arbiter.return_value = MagicMock()

    result = _run_debate_for_theme(
        "theme1", _make_state_chunks(), {}, "test query",
        num_ctx=16384, client_kwargs=None, num_papers=3,
    )
    mock_get_critic.assert_called_once()
    mock_get_arbiter.return_value.revise.assert_not_called()
    assert result["anchoring_score"] == poor_score


@patch("src.graph.survey_nodes.compute_anchoring_score")
@patch("src.graph.survey_nodes.decompose_claims")
@patch("src.graph.survey_nodes.load_theme_synthesis", return_value=None)
@patch("src.graph.survey_nodes._get_drafter")
@patch("src.graph.survey_nodes._get_critic")
@patch("src.graph.survey_nodes._get_arbiter")
def test_run_debate_poorly_grounded_with_critique_invokes_arbiter(
    mock_get_arbiter, mock_get_critic, mock_get_drafter,
    mock_load_theme, mock_decompose, mock_anchoring,
):
    """Poorly grounded draft with actual critique proceeds to Arbiter."""
    from src.graph.survey_nodes import _run_debate_for_theme, _clear_agent_caches
    _clear_agent_caches()

    mock_drafter = MagicMock()
    mock_drafter.draft.return_value = "Poorly grounded draft."
    mock_get_drafter.return_value = mock_drafter
    mock_decompose.return_value = ["Poorly grounded draft."]

    poor_score = 0.25
    call_count = [0]

    def anchoring_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            return (poor_score, [{"claim": "test", "best_evidence_sentence": "", "similarity": poor_score}])
        return (0.9, [])

    mock_anchoring.side_effect = anchoring_side_effect

    mock_critic = MagicMock()
    mock_critic.critique.return_value = "Claim is unsupported."
    mock_get_critic.return_value = mock_critic

    mock_arbiter = MagicMock()
    mock_arbiter.revise.return_value = "Revised draft."
    mock_get_arbiter.return_value = mock_arbiter

    result = _run_debate_for_theme(
        "theme1", _make_state_chunks(), {}, "test query",
        num_ctx=16384, client_kwargs=None, num_papers=3,
    )
    mock_get_critic.assert_called()
    mock_get_arbiter.assert_called_once()
    assert result["synthesis"] == "Revised draft."


@patch("src.graph.survey_nodes.compute_anchoring_score")
@patch("src.graph.survey_nodes.decompose_claims")
@patch("src.graph.survey_nodes.load_theme_synthesis", return_value=None)
@patch("src.graph.survey_nodes._get_drafter")
@patch("src.graph.survey_nodes._get_critic")
@patch("src.graph.survey_nodes._get_arbiter")
def test_run_debate_regression_guard_keeps_draft(
    mock_get_arbiter, mock_get_critic, mock_get_drafter,
    mock_load_theme, mock_decompose, mock_anchoring,
):
    """When debate makes anchoring worse, regression guard keeps the draft."""
    from src.graph.survey_nodes import _run_debate_for_theme, _clear_agent_caches
    _clear_agent_caches()

    mock_drafter = MagicMock()
    mock_drafter.draft.return_value = "Good draft."
    mock_get_drafter.return_value = mock_drafter
    mock_decompose.return_value = ["Good draft."]

    call_count = [0]

    def anchoring_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return (0.25, [{"claim": "test", "best_evidence_sentence": "", "similarity": 0.2}])
        return (0.15, [{"claim": "test", "best_evidence_sentence": "", "similarity": 0.1}])

    mock_anchoring.side_effect = anchoring_side_effect

    mock_critic = MagicMock()
    mock_critic.critique.return_value = "Claim is unsupported."
    mock_get_critic.return_value = mock_critic

    mock_arbiter = MagicMock()
    mock_arbiter.revise.return_value = "Revised draft that is worse."
    mock_get_arbiter.return_value = mock_arbiter

    result = _run_debate_for_theme(
        "theme1", _make_state_chunks(), {}, "test query",
        num_ctx=16384, client_kwargs=None, num_papers=3,
    )
    assert result["synthesis"] == "Good draft."
    assert result["anchoring_score"] == 0.25


def test_survey_nodes_import():
    from src.graph.survey_nodes import (
        survey_cross_theme_synthesize_node,
        survey_per_document_extract_node,
        survey_per_theme_synthesize_node,
        survey_query_decompose_node,
        survey_retrieve_node,
        survey_scrub_node,
        survey_thematic_cluster_node,
    )
    assert callable(survey_query_decompose_node)
    assert callable(survey_retrieve_node)
    assert callable(survey_thematic_cluster_node)
    assert callable(survey_per_document_extract_node)
    assert callable(survey_per_theme_synthesize_node)
    assert callable(survey_cross_theme_synthesize_node)
    assert callable(survey_scrub_node)


@patch("src.graph.survey_nodes.ExtractionAgent")
@patch("src.graph.survey_nodes.GraphBuilder")
def test_per_document_extract_empty_state(mock_gb, mock_ea):
    """Per-document extract with no chunks returns empty dicts."""
    from src.graph.survey_nodes import survey_per_document_extract_node
    from src.graph.networkx_json_storage import NetworkXJSONStorage
    import tempfile, os

    state = {
        "user_query": "test",
        "public_context": [],
        "thematic_clusters": {},
    }
    graph_path = os.path.join(tempfile.gettempdir(), "test_survey_graph.json")
    storage = NetworkXJSONStorage(graph_path)
    result = survey_per_document_extract_node(state, storage)
    assert result["per_paper_extractions"] == {}
    assert result["extracted_entities"] == {}


@patch("src.graph.survey_nodes.QueryDecomposer")
def test_query_decompose_node(mock_qd):
    mock_qd.return_value.decompose.return_value = {
        "original_query": "test query",
        "themes": [
            {"theme": "Theme A", "sub_query": "sub a", "rationale": "r1"},
            {"theme": "Theme B", "sub_query": "sub b", "rationale": "r2"},
        ],
        "cross_cutting_themes": ["shared"],
    }
    from src.graph.survey_nodes import survey_query_decompose_node
    state = {"user_query": "test query"}
    result = survey_query_decompose_node(state)
    assert len(result["decomposed_themes"]) == 2
    assert result["decomposed_themes"][0]["theme"] == "Theme A"


@patch("src.graph.survey_nodes.ThematicClusterer")
def test_thematic_cluster_node(mock_tc):
    mock_tc.return_value.cluster.return_value = {
        "clusters": {"Theme A": ["paper1", "paper2"], "Theme B": ["paper1"]},
        "paper_assignments": {},
        "unassigned": [],
    }
    from src.graph.survey_nodes import survey_thematic_cluster_node
    state = {
        "user_query": "test",
        "decomposed_themes": [
            {"theme": "Theme A", "sub_query": "a"},
            {"theme": "Theme B", "sub_query": "b"},
        ],
        "public_context": [
            {"text": "content", "metadata": {"source": "paper1", "chunk_summary": "summary1"}},
            {"text": "content", "metadata": {"source": "paper2", "chunk_summary": "summary2"}},
        ],
    }
    result = survey_thematic_cluster_node(state)
    assert "Theme A" in result["thematic_clusters"]
    assert result["thematic_clusters"]["Theme A"] == ["paper1", "paper2"]


def test_survey_graph_compiles():
    """Verify the survey graph builds without errors."""
    from unittest.mock import MagicMock
    from src.graph.networkx_json_storage import NetworkXJSONStorage
    from src.graph.graph_builder import build_survey_graph
    import tempfile, os

    mock_retriever = MagicMock()
    graph_path = os.path.join(tempfile.gettempdir(), "test_survey_graph.json")
    storage = NetworkXJSONStorage(graph_path)
    graph = build_survey_graph(mock_retriever, storage)
    assert graph is not None
    # Should have 7 user nodes + __start__
    nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    assert len(nodes) >= 7


@patch("src.graph.survey_nodes.SynthesisDrafter")
def test_cross_theme_synthesize_empty(mock_drafter):
    """Cross-theme synthesis with no per-theme syntheses returns fallback."""
    from src.graph.survey_nodes import survey_cross_theme_synthesize_node
    state = {
        "user_query": "test",
        "per_theme_syntheses": {},
    }
    result = survey_cross_theme_synthesize_node(state)
    assert "No per-theme syntheses" in result["cross_theme_synthesis"]
    assert result["gap_analysis"] == ""


def test_survey_scrub_node():
    """Survey scrub node produces final_output from state."""
    from src.graph.survey_nodes import survey_scrub_node
    state = {
        "cross_theme_synthesis": "Cross synthesis text.",
        "gap_analysis": "Gap 1: missing evidence.",
        "per_theme_syntheses": {
            "Theme A": {"synthesis": "Theme A text.", "anchoring_score": 0.9},
        },
    }
    result = survey_scrub_node(state)
    assert "SURVEY SYNTHESIS" in result["final_output"]
    assert "Cross synthesis text" in result["final_output"]
    assert "RESEARCH GAPS" in result["final_output"]
    assert "Gap 1" in result["final_output"]
    assert "PER-THEME DETAILS" in result["final_output"]
    assert "Theme A" in result["final_output"]
