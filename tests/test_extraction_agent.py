"""Unit tests for ExtractionAgent (Ollama calls mocked)."""

from unittest.mock import patch

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


def test_parse_line_tagged_extraction(mock_llm_invoke):
    """extract_entities now uses line‑tagged format (Phase 10)."""
    mock_llm_invoke.return_value = (
        "TYPE: animal_models\n"
        "ENTITY: C57BL/6J mice\n"
        "EVIDENCE: 20-week-old male C57BL/6J mice were fed a diet containing 45 kcal% fat\n"
        "SOURCE: Chunk 0 | europe_pmc_xml_PMC5506916\n"
        "CONDITIONS: HFD-induced obesity\n"
        "\n"
        "TYPE: animal_models\n"
        "ENTITY: Sprague-Dawley rats\n"
        "EVIDENCE: Sprague-Dawley rats were used for the implant study\n"
        "SOURCE: Chunk 2 | europe_pmc_xml_PMC5512621\n"
    )
    agent = ExtractionAgent()
    out = agent.extract_entities(
        [{"text": "text"}],
        {"discovered_categories": [{"name": "animal_models", "description": "Animal models used"}],
         "key_variables": [], "experimental_methods": []},
        "test query",
    )
    assert "animal_models" in out
    assert len(out["animal_models"]) == 2
    assert out["animal_models"][0]["entity"] == "C57BL/6J mice"
    assert out["animal_models"][0]["evidence"] == (
        "20-week-old male C57BL/6J mice were fed a diet containing 45 kcal% fat"
    )
    assert out["animal_models"][0]["conditions"] == "HFD-induced obesity"
    assert out["animal_models"][1]["entity"] == "Sprague-Dawley rats"


def test_parse_line_tagged_empty_result_logs_warning(mock_llm_invoke):
    """Empty line‑tagged output should return an empty dict with a warning."""
    mock_llm_invoke.return_value = "Some irrelevant text without entity blocks."
    agent = ExtractionAgent()
    out = agent.extract_entities(
        [{"text": "text"}],
        {"discovered_categories": [{"name": "cytokines", "description": "..."}],
         "key_variables": [], "experimental_methods": []},
        "q",
    )
    assert out == {}


def test_format_chunks_for_prompt():
    agent = ExtractionAgent()
    text = agent._format_chunks_for_prompt(
        [
            {"text": "  a\nb  ", "metadata": {}},
            {"text": "c", "metadata": {}},
        ]
    )
    assert "[Chunk 0 | ?] a b" in text
    assert "[Chunk 1 | ?] c" in text


def test_categories_to_line_tagged():
    """_categories_to_line_tagged converts Pass 1 JSON to compact prompt text."""
    categories = {
        "discovered_categories": [
            {"name": "cytokines", "description": "Signaling molecules", "examples_found": ["IL-6"]},
            {"name": "materials", "description": "Implant materials"},
        ],
        "key_variables": ["cytokine levels", "bone formation"],
        "experimental_methods": ["ELISA", "microCT"],
    }
    text = ExtractionAgent._categories_to_line_tagged(categories)
    assert "CATEGORY: cytokines" in text
    assert "DESCRIPTION: Signaling molecules" in text
    assert "CATEGORY: materials" in text
    assert "KEY_VARIABLES: cytokine levels, bone formation" in text
    assert "METHODS: ELISA, microCT" in text
    assert "examples_found" not in text
    assert "IL-6" not in text


def test_parse_line_tagged_ignores_markdown_fence(mock_llm_invoke):
    """Markdown fences in output should be stripped before parsing."""
    mock_llm_invoke.return_value = (
        "```\n"
        "TYPE: cytokine\n"
        "ENTITY: IL-6\n"
        "EVIDENCE: IL-6 was elevated\n"
        "SOURCE: Chunk 0\n"
        "```"
    )
    agent = ExtractionAgent()
    out = agent.extract_entities(
        [{"text": "IL-6 was elevated"}],
        {"discovered_categories": [{"name": "cytokine", "description": "Cytokines"}],
         "key_variables": [], "experimental_methods": []},
        "q",
    )
    assert "cytokine" in out
    assert out["cytokine"][0]["entity"] == "IL-6"


def test_parse_line_tagged_handles_thinking_block(mock_llm_invoke):
    """<think> blocks should be stripped from output."""
    mock_llm_invoke.return_value = (
        "<think>Hmm, let me think about cytokines...</think>\n"
        "TYPE: cytokine\n"
        "ENTITY: TNF-alpha\n"
        "EVIDENCE: TNF-alpha was measured via ELISA\n"
        "SOURCE: Chunk 1 | test.pdf\n"
    )
    agent = ExtractionAgent()
    out = agent.extract_entities(
        [{"text": "TNF-alpha measured"}],
        {"discovered_categories": [{"name": "cytokine", "description": "..."}],
         "key_variables": [], "experimental_methods": []},
        "q",
    )
    assert "cytokine" in out
    assert out["cytokine"][0]["entity"] == "TNF-alpha"
