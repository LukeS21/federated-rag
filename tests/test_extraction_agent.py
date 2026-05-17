"""Unit tests for ExtractionAgent (Ollama calls mocked)."""

from unittest.mock import patch

import pytest

from src.agents.extraction_agent import ExtractionAgent
from src.streaming_handler import ModelDegradedException, TokenStreamHandler


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


# ── Compression‑ratio degradation detection ──────────────────────────

def test_parse_line_tagged_detects_repeated_evidence_lines():
    """Repeated EVIDENCE lines (no entities) → ModelDegradedException."""
    line = "EVIDENCE: The thermoelectric materials |"
    text = "\n".join([line] * 80)
    with pytest.raises(ModelDegradedException) as exc_info:
        ExtractionAgent._parse_line_tagged(text)
    assert "Compression ratio" in str(exc_info.value)


def test_parse_line_tagged_normal_output_no_false_positive():
    """Normal varied output should NOT trigger compression detection."""
    text = (
        "TYPE: material\n"
        "ENTITY: titanium\n"
        "EVIDENCE: Titanium alloys were used for implant fabrication and showed excellent biocompatibility\n"
        "SOURCE: Chunk 0 | paper.pdf\n"
        "CONDITIONS: in vivo implantation for 12 weeks\n"
        "\n"
        "TYPE: cytokine\n"
        "ENTITY: IL-6\n"
        "EVIDENCE: IL-6 levels were significantly elevated in the treated group compared to controls\n"
        "SOURCE: Chunk 1 | paper.pdf\n"
        "CONDITIONS: post-implantation day 7\n"
    )
    result = ExtractionAgent._parse_line_tagged(text)
    assert "material" in result
    assert "cytokine" in result
    assert len(result["material"]) == 1
    assert len(result["cytokine"]) == 1


def test_parse_line_tagged_repeated_evidence_preserves_salvage():
    """Good entities before repetition should be salvaged."""
    text = (
        "TYPE: material\n"
        "ENTITY: titanium\n"
        "EVIDENCE: Titanium alloys were used for implants\n"
        "SOURCE: Chunk 0 | paper.pdf\n"
        "\n"
        + "\n".join(["EVIDENCE: The thermoelectric materials |"] * 50)
    )
    result = ExtractionAgent._parse_line_tagged(text)
    assert "material" in result
    assert result["material"][0]["entity"] == "titanium"


def test_compression_stream_handler_detects_repeated_lines():
    """TokenStreamHandler detects repeated lines via compression ratio."""
    handler = TokenStreamHandler()
    line = "EVIDENCE: The thermoelectric materials |\n"
    handler.current_text = line * 60
    handler._check_degradation()
    assert handler.degraded
    assert "Compression ratio" in handler.degraded_reason


def test_compression_stream_handler_accepts_normal_text():
    """TokenStreamHandler does NOT flag normal varied text."""
    handler = TokenStreamHandler()
    handler.current_text = (
        "TYPE: material\n"
        "ENTITY: titanium\n"
        "EVIDENCE: Titanium alloys were used for implant fabrication and showed excellent biocompatibility\n"
        "SOURCE: Chunk 0 | paper.pdf\n"
        "CONDITIONS: in vivo implantation for 12 weeks\n"
        "\n"
        "TYPE: cytokine\n"
        "ENTITY: IL-6\n"
        "EVIDENCE: IL-6 levels were significantly elevated in the treated group compared to controls\n"
        "SOURCE: Chunk 1 | paper.pdf\n"
        "CONDITIONS: post-implantation day 7\n"
    )
    handler._check_degradation()
    assert not handler.degraded


def test_compression_stream_handler_detects_word_spam():
    """Word-level spam still detected by compression (universal fallback)."""
    handler = TokenStreamHandler(word_repeat_threshold=999)  # disable word check
    handler.current_text = ("energy " * 200) + "\n"
    handler._check_degradation()
    assert handler.degraded
    assert "Compression ratio" in handler.degraded_reason


# ── Token‑budgeted batching ──────────────────────────────────────────

def test_format_chunk_text():
    """_format_chunk_text produces the same format _format_chunks_for_prompt uses."""
    chunk = {"text": "  hello   world  ", "metadata": {"source": "test.pdf"}}
    formatted = ExtractionAgent._format_chunk_text(5, chunk)
    assert formatted == "[Chunk 5 | test.pdf] hello world"


def test_calculate_chunk_budget_positive():
    """Budget should be positive for defaults (boundary_lower=2500, ratio=0.50)."""
    agent = ExtractionAgent()
    system_prompt = ExtractionAgent._build_all_entities_prompt()
    budget = agent._calculate_chunk_budget(system_prompt)
    assert budget > 0
    # boundary_lower=2500, system=919, overhead=350, ratio=0.50
    # → (2375 - 1269) / 1.50 = 737
    assert 500 < budget < 1500


def test_pack_chunks_into_batches_respects_budget():
    """Chunks split when their token count would exceed budget."""
    agent = ExtractionAgent()
    chunks = [
        {"text": "Short chunk.", "metadata": {"source": "a.pdf"}},
        {"text": "Another short chunk.", "metadata": {"source": "a.pdf"}},
        {"text": "Short.", "metadata": {"source": "a.pdf"}},
    ]
    # Each chunk is ~4 tokens → budget of 8 should split after 2
    batches = agent._pack_chunks_into_batches(chunks, chunk_budget=8)
    assert len(batches) >= 2


def test_pack_chunks_into_batches_single_if_fits():
    """All chunks go in one batch if budget is large."""
    agent = ExtractionAgent()
    chunks = [
        {"text": "a", "metadata": {"source": "x"}},
        {"text": "b", "metadata": {"source": "x"}},
    ]
    batches = agent._pack_chunks_into_batches(chunks, chunk_budget=500)
    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_pack_chunks_into_batches_never_loses_chunks():
    """All chunks appear in some batch, total matches input."""
    agent = ExtractionAgent()
    chunks = [
        {"text": f"Chunk number {i} with some extra words to make it bigger.", "metadata": {}}
        for i in range(20)
    ]
    batches = agent._pack_chunks_into_batches(chunks, chunk_budget=30)
    total = sum(len(b) for b in batches)
    assert total == 20


def test_pack_chunks_empty_input():
    """Empty chunk list returns empty batch list."""
    agent = ExtractionAgent()
    assert agent._pack_chunks_into_batches([], chunk_budget=100) == []


def test_calculate_max_workers_respects_ollama_parallel(monkeypatch):
    """Workers capped by OLLAMA_NUM_PARALLEL."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
    workers = ExtractionAgent._calculate_max_workers(num_ctx=16384, total_batches=100)
    assert workers == 2


def test_calculate_max_workers_respects_batch_count(monkeypatch):
    """Workers cannot exceed number of batches."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "16")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
    workers = ExtractionAgent._calculate_max_workers(num_ctx=16384, total_batches=3)
    assert workers == 3


def test_calculate_max_workers_respects_override(monkeypatch):
    """EXTRACTION_MAX_WORKERS forces a cap."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "8")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
    monkeypatch.setenv("EXTRACTION_MAX_WORKERS", "3")
    workers = ExtractionAgent._calculate_max_workers(num_ctx=16384, total_batches=100)
    assert workers == 3


def test_calculate_max_workers_minimum_one(monkeypatch):
    """Always returns at least 1 worker."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "1")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
    workers = ExtractionAgent._calculate_max_workers(num_ctx=16384, total_batches=1)
    assert workers == 1


# ── Pulsed‑wave extraction ────────────────────────────────────────────

def test_try_extract_once_success():
    """Returns (entities, False, {}, output_tokens) on clean extraction."""
    agent = ExtractionAgent()
    fake_entities = {"material": [{"entity": "Ti", "evidence": "Used for implants"}]}
    with patch.object(agent, "extract_entities", return_value=fake_entities):
        entities, degraded, salvage, ot = agent._try_extract_once(
            [{"text": "test", "metadata": {}}], {}, "test query",
        )
    assert entities == fake_entities
    assert degraded is False
    assert salvage == {}
    assert isinstance(ot, int)


def test_try_extract_once_degradation():
    """Returns ({}, True, salvage, output_tokens) on degradation, no crash."""
    agent = ExtractionAgent()
    exc = ModelDegradedException("test degrade", text="partial output")
    exc.parsed = {"material": [{"entity": "salvaged"}]}
    with patch.object(agent, "extract_entities", side_effect=exc):
        entities, degraded, salvage, ot = agent._try_extract_once(
            [{"text": "test", "metadata": {}}], {}, "test query",
        )
    assert entities == {}
    assert degraded is True
    assert "material" in salvage
    assert isinstance(ot, int)


def test_try_extract_once_unexpected_error():
    """Network/timeout errors still return degraded=True, no crash."""
    agent = ExtractionAgent()
    with patch.object(agent, "extract_entities", side_effect=RuntimeError("boom")):
        entities, degraded, salvage, ot = agent._try_extract_once(
            [{"text": "test", "metadata": {}}], {}, "test query",
        )
    assert entities == {}
    assert degraded is True
    assert salvage == {}
    assert ot == 0


def test_extract_paper_recursive_waves(monkeypatch, tmp_path):
    """Pulsed‑wave: GPU restart once per wave, parallel execution, correct merge."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)

    agent = ExtractionAgent()
    chunks = [
        {"text": f"Chunk {i} about biomedical research topic number {i}.", "metadata": {"source": "p.pdf"}}
        for i in range(12)
    ]
    categories = {"discovered_categories": [{"name": "material", "description": "Materials"}],
                  "key_variables": [], "experimental_methods": []}

    restart_count = [0]

    def fake_restart(*args, **kwargs):
        restart_count[0] += 1

    call_idx = [0]

    def fake_extract_once(chunks_list, cats, q, ner_entities=None, **kwargs):
        call_idx[0] += 1
        n = len(chunks_list)
        entities = {"material": [
            {"entity": f"item-batch-{call_idx[0]}", "evidence": f"from {n} chunks"}]}
        return entities, False, {}, 42

    with patch.object(agent, "_try_extract_once", side_effect=fake_extract_once), \
         patch("src.ingestion.pre_extractor.PreExtractor._restart_ollama_process",
               side_effect=fake_restart):
        result = agent.extract_paper_recursive(chunks, categories, "test query")

    assert "material" in result
    assert restart_count[0] >= 1
    assert len(result["material"]) >= 1


def test_extract_paper_recursive_degradation_splits_and_salvages(monkeypatch, tmp_path):
    """Degraded batches split in half and re‑queued, salvage preserved."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)

    agent = ExtractionAgent()
    chunks = [
        {"text": "Chunk about materials.", "metadata": {"source": "p.pdf"}}
        for _ in range(8)
    ]
    categories = {"discovered_categories": [{"name": "material", "description": "M"}],
                  "key_variables": [], "experimental_methods": []}

    call_count = [0]

    def fake_restart(*args, **kwargs):
        pass

    def fake_extract_once(chunks_list, cats, q, ner_entities=None, **kwargs):
        call_count[0] += 1
        n = len(chunks_list)
        if n == 8:
            return {}, True, {"material": [{"entity": "salvaged-item", "evidence": "e"}]}, 0
        elif n == 4:
            return {"material": [{"entity": f"item-{call_count[0]}", "evidence": "e"}]}, False, {}, 50
        elif n <= 2:
            return {"material": [{"entity": f"small-{call_count[0]}", "evidence": "e"}]}, False, {}, 25
        return {}, True, {}, 0

    with patch.object(agent, "_try_extract_once", side_effect=fake_extract_once), \
         patch("src.ingestion.pre_extractor.PreExtractor._restart_ollama_process",
               side_effect=fake_restart):
        result = agent.extract_paper_recursive(chunks, categories, "test query")

    assert "material" in result
    entities = [e["entity"] for e in result["material"]]
    assert "salvaged-item" in entities


def test_extract_paper_recursive_empty_chunks():
    """Empty chunk list returns empty dict."""
    agent = ExtractionAgent()
    result = agent.extract_paper_recursive([], {}, "test query")
    assert result == {}


def test_extract_paper_recursive_bad_chunk_isolation(monkeypatch, tmp_path):
    """Known‑bad chunks are pre‑emptively isolated into single‑chunk batches."""
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
    monkeypatch.setenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")

    agent = ExtractionAgent()
    # Simulate known bad chunk via persistence
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)
    agent._record_bad_chunk("PMC123", 5)
    agent._record_bad_chunk("PMC123", 5)
    agent._record_bad_chunk("PMC123", 5)

    chunks = [
        {"text": f"Chunk {i} about stuff.", "metadata": {"source": "p.pdf", "chunk_index": i, "pmcid": "PMC123"}}
        for i in range(10)
    ]
    categories = {"discovered_categories": [{"name": "material", "description": "M"}],
                  "key_variables": [], "experimental_methods": []}

    def fake_restart(*args, **kwargs):
        pass

    def fake_extract_once(chunks_list, cats, q, ner_entities=None, **kwargs):
        return {"material": [{"entity": "x", "evidence": "e"}]}, False, {}, 10

    with patch.object(agent, "_try_extract_once", side_effect=fake_extract_once), \
         patch("src.ingestion.pre_extractor.PreExtractor._restart_ollama_process",
               side_effect=fake_restart):
        result = agent.extract_paper_recursive(chunks, categories, "test query")

    assert "material" in result
    # Chunk index 5 was isolated — it should still be processed (not lost)
    assert len(result["material"]) >= 1


def test_output_ratio_persistence(monkeypatch, tmp_path):
    """Output ratio updates persist across ExtractionAgent instances."""
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)

    # First update
    ExtractionAgent._update_output_ratio("test-model", chunk_tokens=500, output_tokens=200)
    stats = ExtractionAgent._load_extraction_stats("test-model")
    assert stats["total_chunk_tokens"] == 500
    assert stats["total_output_tokens"] == 200
    # ratio = 0.80 * 0.50 + 0.20 * (200/500) = 0.40 + 0.08 = 0.48

    # Second update
    ratio = ExtractionAgent._update_output_ratio("test-model", chunk_tokens=500, output_tokens=500)
    assert 0.50 < ratio < 0.60  # converging toward 1.0 output ratio

    # Third update
    ratio = ExtractionAgent._update_output_ratio("test-model", chunk_tokens=1000, output_tokens=1000)
    assert ratio > 0.65  # still converging


def test_boundary_defaults(monkeypatch, tmp_path):
    """Fresh stats have boundary_lower=2500, boundary_upper=16384."""
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    stats = ExtractionAgent._load_extraction_stats("new-model")
    assert stats["boundary_lower"] == 2500
    assert stats["boundary_upper"] == 16384
    assert stats["output_ratio"] == 0.50


def test_boundary_update_pass_raises_lower(monkeypatch, tmp_path):
    """A passing batch raises boundary_lower."""
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)
    ExtractionAgent._update_boundary("m", actual_total=4000, passed=True)
    stats = ExtractionAgent._load_extraction_stats("m")
    assert stats["boundary_lower"] == 4000  # rose from 2500
    assert stats["boundary_upper"] == 16384  # unchanged

    ExtractionAgent._update_boundary("m", actual_total=3500, passed=True)
    stats = ExtractionAgent._load_extraction_stats("m")
    assert stats["boundary_lower"] == 4000  # max(4000, 3500) = 4000


def test_boundary_update_degrade_lowers_upper(monkeypatch, tmp_path):
    """A degrading batch lowers boundary_upper."""
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)
    ExtractionAgent._update_boundary("m", actual_total=6000, passed=False)
    stats = ExtractionAgent._load_extraction_stats("m")
    assert stats["boundary_upper"] == 6000  # fell from 16384
    assert stats["boundary_lower"] == 2500  # unchanged

    ExtractionAgent._update_boundary("m", actual_total=7000, passed=False)
    stats = ExtractionAgent._load_extraction_stats("m")
    assert stats["boundary_upper"] == 6000  # min(6000, 7000) = 6000


def test_calculate_chunk_budget_calibrated(monkeypatch, tmp_path):
    """Budget grows as boundary_lower rises from pass data."""
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)
    agent = ExtractionAgent()
    sys_prompt = ExtractionAgent._build_all_entities_prompt()

    # Default: boundary_lower=2500, ratio=0.50
    b1 = agent._calculate_chunk_budget(sys_prompt)
    assert 500 < b1 < 1500  # ~737

    # Simulate accumulated passes → boundary_lower rises to 5000
    ExtractionAgent._update_boundary(agent.model, 4500, passed=True)
    ExtractionAgent._update_boundary(agent.model, 5000, passed=True)
    b2 = agent._calculate_chunk_budget(sys_prompt)
    # (5000*0.95 - 919 - 350) / 1.50 = (4750-1269)/1.50 = 2320
    assert b2 > b1
    assert b2 > 2000

    # Also update ratio → budget should adjust further
    ExtractionAgent._update_output_ratio(agent.model, 5000, 2000)
    b3 = agent._calculate_chunk_budget(sys_prompt)
    # ratio now ~0.48 (from 0.40+0.08), same boundary
    # (4750-1269)/1.48 = 2351
    assert b3 > 2000


def test_boundary_persistence_survives_ratio_update(monkeypatch, tmp_path):
    """_update_output_ratio preserves boundary fields."""
    monkeypatch.setattr(ExtractionAgent, "_STATS_DIR",
                        tmp_path / "projects" / "default")
    (tmp_path / "projects" / "default").mkdir(parents=True)

    ExtractionAgent._update_boundary("m", 5000, passed=True)
    ExtractionAgent._update_output_ratio("m", 1000, 400)

    stats = ExtractionAgent._load_extraction_stats("m")
    assert stats["boundary_lower"] == 5000  # preserved
    assert stats["boundary_upper"] == 16384  # preserved
    assert stats["output_ratio"] != 0.50  # updated
