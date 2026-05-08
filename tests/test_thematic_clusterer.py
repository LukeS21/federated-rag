"""Unit tests for ThematicClusterer (DeepSeek API calls mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.thematic_clusterer import ThematicClusterer


def _mock_llm_response(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


@pytest.fixture
def mock_invoke():
    """Mock langchain_openai.ChatOpenAI.invoke at the class level."""
    with patch("langchain_openai.ChatOpenAI.invoke") as m:
        yield m


@pytest.fixture(autouse=True)
def _bypass_cache():
    with patch("src.agents.thematic_clusterer.get_cache") as mock_cache:
        mock_cache.return_value.get.return_value = None
        yield


SAMPLE_PAPERS = [
    {"id": "paper_a", "title": "Paper A on T cells", "summary": "T cell activation at implants."},
    {"id": "paper_b", "title": "Paper B on macrophages", "summary": "Macrophage polarization in bone."},
    {"id": "paper_c", "title": "Paper C on both", "summary": "T cells and macrophages at titanium surfaces."},
]

SAMPLE_THEMES = [
    {"theme": "T cell response", "sub_query": "How do T cells respond to biomaterials?"},
    {"theme": "Macrophage biology", "sub_query": "What drives macrophage polarization at implants?"},
]


# ---------------------------------------------------------------------------
#  Edge cases (no API call needed)
# ---------------------------------------------------------------------------

def test_cluster_empty_papers():
    result = ThematicClusterer().cluster([], SAMPLE_THEMES)
    assert result == {"clusters": {}, "paper_assignments": {}, "unassigned": []}


def test_cluster_empty_themes():
    result = ThematicClusterer().cluster(SAMPLE_PAPERS, [])
    assert "unthemed" in result["clusters"]
    assert result["clusters"]["unthemed"] == ["paper_a", "paper_b", "paper_c"]
    assert result["paper_assignments"]["paper_a"] == ["unthemed"]
    assert result["unassigned"] == []


def test_cluster_empty_both():
    result = ThematicClusterer().cluster([], [])
    assert result == {"clusters": {}, "paper_assignments": {}, "unassigned": []}


# ---------------------------------------------------------------------------
#  JSON parsing and cluster output (mocked API)
# ---------------------------------------------------------------------------

def test_cluster_parses_valid_json(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"assignments": {"T cell response": ["paper_a", "paper_c"], '
        '"Macrophage biology": ["paper_b", "paper_c"]}, "unassigned": []}'
    )
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(SAMPLE_PAPERS, SAMPLE_THEMES)

    assert "clusters" in result
    assert "paper_assignments" in result
    assert "unassigned" in result

    clusters = result["clusters"]
    assert "paper_a" in clusters["T cell response"]
    assert "paper_c" in clusters["T cell response"]
    assert "paper_b" in clusters["Macrophage biology"]
    assert "paper_c" in clusters["Macrophage biology"]


def test_cluster_paper_multi_theme_assignment(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"assignments": {"T cell response": ["paper_a", "paper_c"], '
        '"Macrophage biology": ["paper_b", "paper_c"]}, "unassigned": []}'
    )
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(SAMPLE_PAPERS, SAMPLE_THEMES)
    pa = result["paper_assignments"]
    assert set(pa["paper_c"]) == {"T cell response", "Macrophage biology"}
    assert pa["paper_a"] == ["T cell response"]
    assert pa["paper_b"] == ["Macrophage biology"]


def test_cluster_unassigned_papers(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"assignments": {"T cell response": ["paper_a"]}, "unassigned": ["paper_b", "paper_c"]}'
    )
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(SAMPLE_PAPERS, SAMPLE_THEMES)
    assert result["unassigned"] == ["paper_b", "paper_c"]


def test_cluster_paper_in_unassigned_has_empty_assignments(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"assignments": {"T cell response": ["paper_a"]}, "unassigned": ["paper_b"]}'
    )
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(SAMPLE_PAPERS, SAMPLE_THEMES)
    assert result["paper_assignments"]["paper_b"] == []


# ---------------------------------------------------------------------------
#  JSON parsing edge cases
# ---------------------------------------------------------------------------

def test_parse_json_markdown_fence(mock_invoke):
    mock_invoke.return_value = _mock_llm_response("""Results:
```json
{"assignments": {"T cell response": ["paper_a"]}, "unassigned": ["paper_b", "paper_c"]}
```
""")
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(SAMPLE_PAPERS, SAMPLE_THEMES)
    assert "paper_a" in result["clusters"]["T cell response"]


def test_parse_json_brace_fallback(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        'Here you go — {"assignments": {"Macrophage biology": ["paper_b"]}, '
        '"unassigned": ["paper_a", "paper_c"]}'
    )
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(SAMPLE_PAPERS, SAMPLE_THEMES)
    assert "paper_b" in result["clusters"]["Macrophage biology"]


# ---------------------------------------------------------------------------
#  Model configurability
# ---------------------------------------------------------------------------

def test_default_model_is_chat():
    agent = ThematicClusterer()
    assert agent.model == "deepseek-chat"


def test_model_override():
    agent = ThematicClusterer(model="deepseek-v4-pro")
    assert agent.model == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
#  Paper ID fallbacks (when "id" is missing)
# ---------------------------------------------------------------------------

def test_cluster_paper_without_id_uses_title(mock_invoke):
    papers_no_id = [
        {"title": "Paper X", "summary": "Content about T cells."},
    ]
    themes = SAMPLE_THEMES[:1]
    mock_invoke.return_value = _mock_llm_response(
        '{"assignments": {"T cell response": ["Paper X"]}, "unassigned": []}'
    )
    agent = ThematicClusterer(use_embeddings=False)
    result = agent.cluster(papers_no_id, themes)
    # When no 'id' field, title is used as the paper ID in the prompt
    assert "Paper X" in result["paper_assignments"]


def test_cluster_empty_themes_fallback_ids():
    """When no id or title, fall back to str(index) as paper identifiers."""
    papers = [{"summary": "stuff"}, {"summary": "more stuff"}]
    result = ThematicClusterer().cluster(papers, [])
    assert "unthemed" in result["clusters"]
    assert result["clusters"]["unthemed"] == ["0", "1"]
