"""Unit tests for ExtractionAgent (Ollama calls mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.extraction_agent import ExtractionAgent


@pytest.fixture
def mock_llm_invoke():
    with patch.object(ExtractionAgent, "_call_llm") as m:
        yield m


def test_discover_categories_parses_json(mock_llm_invoke):
    mock_llm_invoke.return_value = (
        '{"discovered_categories": [], "key_variables": [], "experimental_methods": []}'
    )
    agent = ExtractionAgent()
    out = agent.discover_categories([], "test query")
    assert out == {"discovered_categories": [], "key_variables": [], "experimental_methods": []}


def test_parse_json_with_markdown_fence(mock_llm_invoke):
    mock_llm_invoke.return_value = """Here is the result:
```json
{"discovered_categories": [{"name": "x", "description": "d", "examples_found": ["a"]}], "key_variables": [], "experimental_methods": []}
```
"""
    agent = ExtractionAgent()
    out = agent.discover_categories([], "q")
    assert out["discovered_categories"][0]["name"] == "x"


def test_parse_json_brace_fallback_after_preamble(mock_llm_invoke):
    mock_llm_invoke.return_value = (
        'Sure — {"animal_models": [{"entity": "mice", "evidence": "text", "source": "Chunk 0"}]}'
    )
    agent = ExtractionAgent()
    out = agent.extract_entities([{"text": "text"}], {"discovered_categories": []}, "q")
    assert "animal_models" in out
    assert out["animal_models"][0]["entity"] == "mice"


def test_format_chunks_for_prompt():
    agent = ExtractionAgent()
    text = agent._format_chunks_for_prompt(
        [
            {"text": "  a\nb  ", "metadata": {}},
            {"text": "c", "metadata": {}},
        ]
    )
    assert "[Chunk 0] a b" in text
    assert "[Chunk 1] c" in text
