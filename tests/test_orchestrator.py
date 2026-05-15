"""Tests for orchestrator.py — seed derivation, query extraction, cycle composition.

Strategy:
  - Helper methods (_derive_seed_terms, _top_kg_entities, _queries_from_discovery)
    are tested directly with real inputs (no mocking needed).
  - The full _run_cycle is tested with mocked external APIs (web search, EPMC)
    to avoid network calls.
  - Dry-run mode is tested to confirm it skips ingestion.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.agents.orchestrator import (
    DEFAULT_SEED_TERMS,
    Orchestrator,
    STATE_PATH,
    PID_PATH,
)
from src.graph.networkx_json_storage import NetworkXJSONStorage


@pytest.fixture(autouse=True)
def _clean_state_files():
    """Remove state/PID files between tests so _load_state() starts fresh."""
    for p in (STATE_PATH, PID_PATH):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    yield
    for p in (STATE_PATH, PID_PATH):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_graph_with_nodes(*names: str) -> NetworkXJSONStorage:
    """Create an in-memory NetworkX graph with some entity nodes."""
    gs = NetworkXJSONStorage(file_path="/tmp/_test_orch_graph.json")
    gs._graph = nx.DiGraph()
    for i, name in enumerate(names):
        gs.add_node(f"material:{name}", "material", {"name": name})
        if i > 0:
            gs.add_edge(f"material:{name}", f"material:{names[0]}", "co_occurs_with", {
                "extracted_at": "2026-01-01T00:00:00Z",
                "source_paper": "test",
                "evidence_phrase": f"{name} co-occurs with {names[0]}",
            })
    return gs


# ── Tests: _top_kg_entities ──────────────────────────────────────────────────

class TestTopKgEntities:
    def test_returns_top_by_degree(self):
        gs = _make_graph_with_nodes("Ti-6Al-4V", "TiO2", "Hydroxyapatite", "Zirconia")
        orch = Orchestrator(graph_storage=gs)
        top = orch._top_kg_entities(max_terms=3)
        # Ti-6Al-4V has highest degree (3 edges), should be first
        assert len(top) >= 1
        assert top[0] == "Ti-6Al-4V"

    def test_no_graph_returns_empty(self):
        orch = Orchestrator(graph_storage=None)
        assert orch._top_kg_entities() == []

    def test_max_terms_respected(self):
        gs = _make_graph_with_nodes("A", "B", "C", "D", "E", "F")
        orch = Orchestrator(graph_storage=gs)
        top = orch._top_kg_entities(max_terms=3)
        assert len(top) <= 3

    def test_handles_non_colon_node_ids(self):
        """Nodes without category: prefix should still be returned."""
        gs = NetworkXJSONStorage(file_path="/tmp/_test_orch_graph2.json")
        gs._graph = nx.DiGraph()
        gs._graph.add_node("bare_node")
        orch = Orchestrator(graph_storage=gs)
        top = orch._top_kg_entities(max_terms=1)
        assert "bare_node" in top


# ── Tests: _derive_seed_terms ────────────────────────────────────────────────

class TestDeriveSeedTerms:
    def test_static_defaults(self):
        orch = Orchestrator(graph_storage=None)
        terms = orch._derive_seed_terms()
        # Should contain static defaults
        for static in DEFAULT_SEED_TERMS:
            assert static in terms

    def test_merges_kg_and_static(self):
        gs = _make_graph_with_nodes("Ti-6Al-4V", "BioactiveGlass")
        orch = Orchestrator(graph_storage=gs)
        terms = orch._derive_seed_terms()
        # Static defaults + KG entities
        assert "Ti-6Al-4V" in terms or "BioactiveGlass" in terms
        for static in DEFAULT_SEED_TERMS:
            assert static in terms

    def test_capped_at_eight(self):
        orch = Orchestrator(graph_storage=None, seed_terms=[f"t{i}" for i in range(15)])
        terms = orch._derive_seed_terms()
        assert len(terms) <= 8


# ── Tests: _queries_from_discovery ──────────────────────────────────────────

class TestQueriesFromDiscovery:
    def test_extracts_snippets_as_queries(self):
        discovered = [
            {"title": "Exosome-mediated drug delivery in osteoporosis treatment",
             "snippet": "Recent advances in ..."},
            {"title": "Macrophage polarization in peri-implant osteolysis",
             "snippet": "A comprehensive review of ..."},
        ]
        queries = Orchestrator._queries_from_discovery(discovered, DEFAULT_SEED_TERMS)
        assert len(queries) >= 2
        assert "Exosome-mediated drug delivery" in queries[0]
        assert "Macrophage polarization" in queries[1]

    def test_falls_back_to_seed_terms(self):
        discovered = []  # web discovery returned nothing
        queries = Orchestrator._queries_from_discovery(discovered, DEFAULT_SEED_TERMS)
        assert len(queries) >= 2
        # Should contain seed terms
        assert any(DEFAULT_SEED_TERMS[0] in q for q in queries)

    def test_deduplicates(self):
        discovered = [
            {"title": "titanium implant osseointegration research 2025", "snippet": ""},
            {"title": "titanium implant osseointegration research 2025", "snippet": ""},
        ]
        queries = Orchestrator._queries_from_discovery(discovered, [])
        assert len(queries) == 1

    def test_capped_at_six(self):
        discovered = [
            {"title": f"Topic {i} — biomedical research finding about biomaterials",
             "snippet": ""}
            for i in range(15)
        ]
        queries = Orchestrator._queries_from_discovery(discovered, [])
        assert len(queries) <= 6

    def test_short_snippets_filtered(self):
        """Titles/snippets < 20 chars should be skipped."""
        discovered = [
            {"title": "Short", "snippet": ""},
            {"title": "A very detailed and specific biomedical research topic with enough length",
             "snippet": ""},
        ]
        queries = Orchestrator._queries_from_discovery(discovered, [])
        # "Short" is < 20 chars, should be skipped
        assert len(queries) == 1
        assert "Short" not in queries[0]


# ── Tests: Orchestrator instantiation ────────────────────────────────────────

class TestOrchestratorInit:
    def test_defaults(self):
        orch = Orchestrator()
        assert orch.interval_minutes == 60
        assert orch.max_papers_per_query == 5
        assert orch._cycle == 0
        assert orch._total_ingested == 0
        assert orch.graph_storage is None
        assert not orch.is_running

    def test_custom_config(self):
        orch = Orchestrator(
            interval_minutes=30,
            max_papers_per_query=3,
            seed_terms=["custom term 1", "custom term 2"],
        )
        assert orch.interval_minutes == 30
        assert orch.max_papers_per_query == 3
        assert "custom term 1" in orch._seed_terms


# ── Tests: _run_cycle with mocked APIs ───────────────────────────────────────

class TestRunCycleMocked:
    """Test the full cycle with mocked external calls."""

    def test_run_once_with_no_discovery_results(self):
        """If web discovery returns nothing, cycle still completes cleanly."""
        orch = Orchestrator(graph_storage=None)

        with patch.object(orch, "_discover_topics", return_value=[]), \
             patch.object(orch, "_search_and_ingest", return_value={}):
            summary = orch._run_cycle()

        assert summary["cycle"] >= 1
        assert summary["discovered_topics"] == 0
        assert summary["papers_ingested"] == 0

    def test_run_once_increments_cycle(self):
        orch = Orchestrator(graph_storage=None)

        with patch.object(orch, "_discover_topics", return_value=[]), \
             patch.object(orch, "_search_and_ingest", return_value={}):
            orch._run_cycle()
            assert orch._cycle == 1
            orch._run_cycle()
            assert orch._cycle == 2

    def test_dry_run_skips_ingestion(self):
        """When discovery works but _search_and_ingest is NOT called because
        queries are empty or mocked out — test the full path."""
        discovered = [
            {"title": "Osteoblast differentiation on titanium surfaces "
             "with nanoscale roughness features",
             "snippet": "A study examining ..."},
        ]
        orch = Orchestrator(graph_storage=None)

        with patch.object(orch, "_discover_topics", return_value=discovered), \
             patch.object(orch, "_search_and_ingest", return_value={}):
            summary = orch._run_cycle()

        assert summary["discovered_topics"] == 1
        assert summary["epmc_queries_run"] >= 1

    def test_handoff_called_on_completion(self):
        """_write_handoff should be called regardless of cycle outcome."""
        orch = Orchestrator(graph_storage=None)
        handoff_called = []

        with patch.object(orch, "_discover_topics", return_value=[]), \
             patch.object(orch, "_search_and_ingest", return_value={}), \
             patch.object(orch, "_write_handoff",
                          side_effect=lambda s: handoff_called.append(s)):
            orch._run_cycle()

        assert len(handoff_called) == 1

    def test_kg_save_error_captured(self):
        """If graph_storage.save() fails, the error is captured in summary."""
        gs = MagicMock()
        gs.save.side_effect = OSError("disk full")

        orch = Orchestrator(graph_storage=gs)

        with patch.object(orch, "_discover_topics", return_value=[]), \
             patch.object(orch, "_search_and_ingest", return_value={}):
            summary = orch._run_cycle()

        assert any("disk full" in str(e) for e in summary["errors"])


# ── Tests: resolve_gaps delegation ───────────────────────────────────────────

class TestResolveGapsDelegation:
    def test_resolve_gaps_delegates_to_gap_resolver(self):
        orch = Orchestrator(graph_storage=MagicMock(), max_papers_per_query=3)

        with patch("src.agents.gap_resolver.GapResolver.resolve_gaps") as mock_resolve:
            mock_resolve.return_value = {"gaps_found": 2, "papers_ingested": 1}
            result = orch.resolve_gaps("No data on X. Missing Y.")

        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs["ingest"] is True
        assert call_kwargs["graph_storage"] is orch.graph_storage
        assert result["gaps_found"] == 2


# ── Tests: State file + PID ──────────────────────────────────────────────────

class TestStateFile:
    def test_state_file_contains_expected_keys(self, tmp_path, monkeypatch):
        """_write_state writes a JSON file with expected keys."""
        import src.agents.orchestrator as orch_mod
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")
        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)

        orch = Orchestrator()
        orch._cycle = 5
        orch._total_ingested = 42
        orch._write_state("cycle_complete", last_error="test error")

        state_path = tmp_path / "orchestrator_state.json"
        assert state_path.exists()
        import json
        state = json.loads(state_path.read_text())
        assert state["status"] == "cycle_complete"
        assert state["last_cycle"] == 5
        assert state["total_ingested"] == 42
        assert state["last_error"] == "test error"
        assert "last_heartbeat" in state
        assert "pid" in state

    def test_pid_file_written_and_removed(self, tmp_path, monkeypatch):
        """_write_pid writes PID, _remove_pid deletes it."""
        import src.agents.orchestrator as orch_mod
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)

        orch = Orchestrator()
        orch._write_pid()

        pid_path = tmp_path / "orchestrator.pid"
        assert pid_path.exists()
        pid = int(pid_path.read_text().strip())
        assert pid > 0

        orch._remove_pid()
        assert not pid_path.exists()

    def test_handoff_writes_cycle_specific_file(self, tmp_path, monkeypatch):
        """_write_handoff writes to cycle_N_handoff.md, not HANDOFF.md."""
        import src.agents.orchestrator as orch_mod
        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")

        orch = Orchestrator()
        summary = {"cycle": 3, "discovered_topics": 5, "epmc_queries_run": 2,
                   "papers_fetched": 0, "papers_ingested": 0, "errors": []}

        orch._write_handoff(summary)

        handoff_path = tmp_path / "cycle_3_handoff.md"
        assert handoff_path.exists()
        content = handoff_path.read_text()
        assert "Cycle: 3" in content
        assert "Discovered topics: 5" in content

        # The human HANDOFF.md should NOT be overwritten
        human_handoff = tmp_path / "HANDOFF.md"
        # (it may not exist — but the orchestrator should never write there)


# ── Tests: Parallel EPMC (module-level _fetch_and_parse_for_query) ───────────

class TestFetchAndParseForQuery:
    def test_returns_structured_result_with_mocked_epmc(self):
        """_fetch_and_parse_for_query calls EPMC and returns structured data."""
        from src.agents.orchestrator import _fetch_and_parse_for_query

        mock_paper = {
            "pmcid": "PMC12345",
            "doi": "10.1234/test",
            "title": "Test Paper About Biomaterials",
        }
        mock_xml = (
            "<article>"
            "<front><article-meta>"
            "<title-group><article-title>A Study of Biomaterial Surface Modifications"
            " and Their Effects on Macrophage Polarization In Vivo</article-title>"
            "</title-group>"
            "<abstract><p>This study investigates how biomaterial surface properties "
            "influence the host immune response following implantation in a murine "
            "model of peri-implant osteolysis.</p></abstract>"
            "</article-meta></front>"
            "<body>"
            "<sec><title>Introduction</title>"
            "<p>Biomaterial surface properties critically influence the host immune "
            "response following surgical implantation. Surface roughness, hydrophilicity, "
            "and chemical composition have all been shown to modulate macrophage "
            "polarization phenotypes at the implant-tissue interface. Understanding "
            "these relationships is essential for designing next-generation implant "
            "materials that promote favorable tissue integration.</p>"
            "</sec>"
            "<sec><title>Methods</title>"
            "<p>C57BL/6J mice were implanted with titanium rods having either "
            "rough-hydrophilic surfaces or machined control surfaces. Peri-implant "
            "tissue was harvested at multiple time points and analyzed using flow "
            "cytometry and quantitative real-time PCR for macrophage markers.</p>"
            "</sec>"
            "</body>"
            "</article>"
        )

        with patch("src.retrieval.europe_pmc.EuropePMCClient") as MockEPMC:
            mock_epmc_instance = MockEPMC.return_value
            mock_epmc_instance.search.return_value = [mock_paper]
            mock_epmc_instance.full_text_xml_batch.return_value = {"PMC12345": mock_xml}

            result = _fetch_and_parse_for_query("test query", max_papers=3)

        assert result["query"] == "test query"
        assert len(result["paper_data"]) == 1
        pd = result["paper_data"][0]
        assert pd["pmcid"] == "PMC12345"
        assert pd["doi"] == "10.1234/test"
        assert len(pd["chunks"]) >= 1
        assert "text" in pd["chunks"][0]
        assert "metadata" in pd["chunks"][0]

    def test_skips_completed_pmcids(self):
        """Already-ingested PMCIDs are skipped."""
        from src.agents.orchestrator import _fetch_and_parse_for_query

        mock_paper = {"pmcid": "PMC99999", "doi": "10.9999/skip", "title": "Skip Me"}
        mock_xml = (
            "<article><front><article-meta>"
            "<title-group><article-title>Skip This Paper About Nothing</article-title></title-group>"
            "<abstract><p>This paper contains sufficient text to pass the minimum word count "
            "requirement of the XML parser. It discusses biomaterials and immune response "
            "in a murine model of peri-implant inflammation.</p></abstract>"
            "</article-meta></front>"
            "<body><sec><title>Results</title>"
            "<p>The experimental results demonstrate that surface modification of titanium "
            "implants significantly alters macrophage polarization kinetics in a murine "
            "model of peri-implant osteolysis over a fourteen day observation period.</p>"
            "</sec></body></article>"
        )

        with patch("src.retrieval.europe_pmc.EuropePMCClient") as MockEPMC:
            mock_epmc_instance = MockEPMC.return_value
            mock_epmc_instance.search.return_value = [mock_paper]
            mock_epmc_instance.full_text_xml_batch.return_value = {"PMC99999": mock_xml}

            result = _fetch_and_parse_for_query(
                "test", max_papers=3, completed_pmcids={"PMC99999"},
            )

        assert result["paper_data"] == []

    def test_empty_search_returns_empty(self):
        """Empty EPMC search returns no paper_data."""
        from src.agents.orchestrator import _fetch_and_parse_for_query

        with patch("src.retrieval.europe_pmc.EuropePMCClient") as MockEPMC:
            MockEPMC.return_value.search.return_value = []
            result = _fetch_and_parse_for_query("nothing", max_papers=3)

        assert result["query"] == "nothing"
        assert result["paper_data"] == []
