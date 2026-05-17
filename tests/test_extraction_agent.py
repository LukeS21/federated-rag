"""Unit tests for ExtractionAgent (Ollama calls mocked)."""

from unittest.mock import patch

import pytest

from src.agents.extraction_agent import ExtractionAgent


@pytest.fixture
def mock_llm_invoke():
    with patch.object(ExtractionAgent, "_call_llm") as m:
        yield m


@pytest.fixture
def mock_extraction_llm():
    with patch.object(ExtractionAgent, "_call_llm_with_detection") as m:
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


def test_parse_line_tagged_extraction(mock_extraction_llm):
    """extract_entities now uses line‑tagged format (Phase 10)."""
    mock_extraction_llm.return_value = (
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


def test_parse_line_tagged_empty_result_logs_warning(mock_extraction_llm):
    """Empty line‑tagged output should return an empty dict with a warning."""
    mock_extraction_llm.return_value = "Some irrelevant text without entity blocks."
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


def test_parse_line_tagged_ignores_markdown_fence(mock_extraction_llm):
    """Markdown fences in output should be stripped before parsing."""
    mock_extraction_llm.return_value = (
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


def test_parse_line_tagged_handles_thinking_block(mock_extraction_llm):
    """<think> blocks should be stripped from output."""
    mock_extraction_llm.return_value = (
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


def test_parse_markdown_fallback_keywords_list():
    """Parse markdown keyword lists that Qwen outputs when ignoring line‑tagged format."""
    text = (
        "**Keywords:**\n\n"
        "* **Materials:** Titanium, Titanium Alloy, Stainless Steel, Polymer\n"
        "* **Methods:** ELISA, flow cytometry, microCT, PCR\n\n"
        "**Cytokines**\n"
        "* **IL-6:** elevated in obese mice post-implantation\n"
        "* **TNF-alpha:** correlated with macrophage activation\n"
    )
    result = ExtractionAgent._parse_markdown_fallback(text)
    assert "Materials" in result
    assert len(result["Materials"]) == 4
    assert result["Materials"][0]["entity"] == "Titanium"
    assert result["Materials"][1]["entity"] == "Titanium Alloy"
    assert "Methods" in result
    assert len(result["Methods"]) == 4
    assert "Cytokines" in result
    assert result["Cytokines"][0]["entity"] == "IL-6"
    assert "elevated in obese mice" in result["Cytokines"][0]["evidence"]


def test_parse_markdown_fallback_bold_headers():
    """Parse markdown with **bold** section headers and bullet entities."""
    text = (
        "**Bioactive Materials and Composites**\n\n"
        "* **Bioactive Glass:** Mentioned in the context of bone regeneration\n"
        "* **Bioactive Ceramic:** General category for materials used in bone regeneration\n"
        "* **Hydroxyapatite:** HA coatings showed improved osseointegration\n"
    )
    result = ExtractionAgent._parse_markdown_fallback(text)
    assert "Bioactive Materials and Composites" in result
    assert len(result["Bioactive Materials and Composites"]) == 3
    assert result["Bioactive Materials and Composites"][0]["entity"] == "Bioactive Glass"


def test_parse_markdown_fallback_empty():
    """Non-markdown text should produce empty result."""
    result = ExtractionAgent._parse_markdown_fallback("Some plain text without formatting.")
    assert result == {}


def test_extract_entities_falls_back_to_markdown(mock_extraction_llm):
    """When line‑tagged parsing fails, the fallback should try markdown parsing."""
    mock_extraction_llm.return_value = (
        "**Materials**\n\n"
        "* **Titanium:** Ti-6Al-4V alloy for implants\n"
        "* **Hydroxyapatite:** HA coatings for bone\n"
    )
    agent = ExtractionAgent()
    out = agent.extract_entities(
        [{"text": "Ti-6Al-4V implant study"}],
        {"discovered_categories": [{"name": "materials", "description": "Test"}],
         "key_variables": [], "experimental_methods": []},
        "q",
    )
    assert "Materials" in out or "materials" in out
    entities = out.get("Materials", out.get("materials", []))
    assert len(entities) >= 1
    assert entities[0]["entity"] == "Titanium"
