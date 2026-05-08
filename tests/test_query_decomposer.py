"""Unit tests for QueryDecomposer (DeepSeek API calls mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.query_decomposer import QueryDecomposer


def _mock_llm_response(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


@pytest.fixture
def mock_invoke():
    """Mock langchain_openai.ChatOpenAI.invoke at the class level."""
    with patch("langchain_openai.ChatOpenAI.invoke") as m:
        yield m


# Also bypass cache so tests don't read stale disk entries
@pytest.fixture(autouse=True)
def _bypass_cache():
    with patch("src.agents.query_decomposer.get_cache") as mock_cache:
        mock_cache.return_value.get.return_value = None
        yield


# ---------------------------------------------------------------------------
#  JSON parsing (via decompose)
# ---------------------------------------------------------------------------

def test_decompose_parses_valid_json(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"themes": [{"theme": "T cell biology", "sub_query": "How do T cells respond to implants?", '
        '"rationale": "Core immune axis"}], "cross_cutting_themes": ["inflammation"]}'
    )
    agent = QueryDecomposer()
    result = agent.decompose("Test query about implants and T cells")
    assert "original_query" in result
    assert result["original_query"] == "Test query about implants and T cells"
    assert len(result["themes"]) == 1
    assert result["themes"][0]["theme"] == "T cell biology"
    assert result["cross_cutting_themes"] == ["inflammation"]


def test_decompose_includes_original_query(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"themes": [], "cross_cutting_themes": []}'
    )
    agent = QueryDecomposer()
    result = agent.decompose("some unique query")
    assert result["original_query"] == "some unique query"


def test_decompose_multiple_themes(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        '{"themes": ['
        '{"theme": "A", "sub_query": "a?", "rationale": "r1"},'
        '{"theme": "B", "sub_query": "b?", "rationale": "r2"},'
        '{"theme": "C", "sub_query": "c?", "rationale": "r3"}'
        '], "cross_cutting_themes": ["shared method"]}'
    )
    agent = QueryDecomposer()
    result = agent.decompose("broad query")
    assert len(result["themes"]) == 3
    assert all("theme" in t and "sub_query" in t and "rationale" in t for t in result["themes"])


def test_parse_json_markdown_fence(mock_invoke):
    mock_invoke.return_value = _mock_llm_response("""Here is the decomposition:
```json
{"themes": [{"theme": "X", "sub_query": "y", "rationale": "z"}], "cross_cutting_themes": []}
```
Done.""")
    agent = QueryDecomposer()
    result = agent.decompose("q")
    assert result["themes"][0]["theme"] == "X"


def test_parse_json_brace_fallback(mock_invoke):
    mock_invoke.return_value = _mock_llm_response(
        'Sure, here you go — {"themes": [{"theme": "Brace", "sub_query": "bq", '
        '"rationale": "br"}], "cross_cutting_themes": ["ct"]} and that is all.'
    )
    agent = QueryDecomposer()
    result = agent.decompose("q")
    assert result["themes"][0]["theme"] == "Brace"


def test_decompose_scrubs_non_ascii(mock_invoke):
    """Non-ASCII characters in LLM output are scrubbed before JSON parse."""
    mock_invoke.return_value = _mock_llm_response(
        '{"themes": [{"theme": "\u03b1\u03b2 T cells", "sub_query": "\u03bc\u03b1", '
        '"rationale": "\u03b3\u03b4"}], "cross_cutting_themes": []}'
    )
    agent = QueryDecomposer()
    result = agent.decompose("q")
    assert result["themes"][0]["theme"] == "alphabeta T cells"


# ---------------------------------------------------------------------------
#  Model configurability
# ---------------------------------------------------------------------------

def test_default_model_is_chat():
    agent = QueryDecomposer()
    assert agent.model == "deepseek-chat"


def test_model_override():
    agent = QueryDecomposer(model="deepseek-v4-pro")
    assert agent.model == "deepseek-v4-pro"
