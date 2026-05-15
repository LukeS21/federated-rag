"""Integration test — full orchestrator cycle with mocked external APIs.

Verifies end-to-end data flow from web discovery through batch ingest
to handoff generation, without hitting real APIs.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.orchestrator import Orchestrator


@pytest.fixture
def mock_web_discovery():
    """Return fake discovery results simulating DuckDuckGo output."""
    return [
        {"title": "Exosome-mediated drug delivery in osteoporosis treatment "
         "with nano-biomaterials — a comprehensive review",
         "snippet": "Recent advances in exosome drug delivery for bone regeneration.",
         "source_type": "discovery"},
        {"title": "Macrophage polarization in peri-implant osteolysis "
         "modulated by surface chemistry",
         "snippet": "A review of M1/M2 polarization dynamics.",
         "source_type": "discovery"},
    ]


@pytest.fixture
def mock_epmc_search():
    """Return fake EPMC search results for two queries."""
    return [
        {
            "pmcid": "PMC12345",
            "doi": "10.1000/test.1",
            "title": "Exosome-Mediated Drug Delivery for Osteoporosis: A Review",
        },
        {
            "pmcid": "PMC67890",
            "doi": "10.1000/test.2",
            "title": "Macrophage Polarization Modulated by Titanium Surface Chemistry",
        },
    ]


SAMPLE_XML = """<article>
  <front>
    <article-meta>
      <title-group><article-title>Test Paper</article-title></title-group>
      <abstract><p>This study investigates biomaterial surface modifications
      and their effects on macrophage polarization in a murine model of
      peri-implant osteolysis. Titanium implants with rough-hydrophilic
      surfaces significantly increased M2 macrophage markers including
      IL-4 and IL-10 while decreasing M1 markers such as TNF-alpha and IL-6.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title>
      <p>Biomaterial surface properties critically influence the host immune
      response. Surface roughness, hydrophilicity, and chemical composition
      all modulate macrophage polarization phenotypes at the implant-tissue
      interface.</p>
    </sec>
    <sec><title>Methods</title>
      <p>C57BL/6J mice (n=24, 8 weeks old) received titanium implants with
      either rough-hydrophilic (modSLA) or machined surfaces in the
      proximal tibia. Peri-implant tissue was harvested at days 3, 7, and 14
      post-implantation for flow cytometry and RT-qPCR analysis.</p>
    </sec>
    <sec><title>Results</title>
      <p>Rough-hydrophilic surfaces significantly increased M2 macrophage
      markers (IL-4, IL-10, CD206) at all time points compared to machined
      surfaces (p&lt;0.01). M1 markers (TNF-alpha, IL-6, iNOS) were
      significantly decreased at day 7 and 14 (p&lt;0.05).</p>
    </sec>
  </body>
</article>"""


class TestOrchestratorIntegration:
    """Full-cycle integration tests with mocked external APIs."""

    def test_full_cycle_web_discovery_to_handoff(
        self, tmp_path, monkeypatch, mock_web_discovery,
    ):
        """Run a complete cycle: discovery → queries → ingest → handoff."""
        import src.agents.orchestrator as orch_mod

        # Redirect all file output to tmp_path
        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")

        orch = Orchestrator(max_papers_per_query=2)

        # Simulate _search_and_ingest returning 2 papers ingested per query
        mock_ingest_result = {
            "query1": {"fetched": 2, "ingested": 2},
            "query2": {"fetched": 1, "ingested": 1},
        }

        with patch.object(orch, "_discover_topics", return_value=mock_web_discovery), \
             patch.object(orch, "_search_and_ingest", return_value=mock_ingest_result):
            summary = orch.run_once()

        # ── Verify cycle summary ──
        assert summary["cycle"] == 1
        assert summary["mode"] == "live"
        assert summary["discovered_topics"] == 2
        assert summary["epmc_queries_run"] >= 1
        assert summary["papers_fetched"] == 3
        assert summary["papers_ingested"] == 3

        # ── Verify cycle handoff was written ──
        handoff_path = tmp_path / "cycle_1_handoff.md"
        assert handoff_path.exists()
        content = handoff_path.read_text()
        assert "Cycle: 1" in content
        assert "Discovered topics: 2" in content
        assert "Papers fetched: 3" in content

        # ── Verify state file ──
        state_path = tmp_path / "orchestrator_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["last_cycle"] == 1
        assert state["total_ingested"] == 3
        assert state["status"] == "cycle_complete"

    def test_dry_run_skips_epmc_and_ingest(
        self, tmp_path, monkeypatch, mock_web_discovery,
    ):
        """Dry run discovers topics and builds queries but skips all ingestion."""
        import src.agents.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")

        orch = Orchestrator(dry_run=True)

        with patch.object(orch, "_discover_topics", return_value=mock_web_discovery):
            summary = orch.run_once()

        assert summary["mode"] == "dry_run"
        assert summary["discovered_topics"] == 2
        assert summary["epmc_queries_run"] >= 1
        assert summary["papers_fetched"] == 0
        assert summary["papers_ingested"] == 0
        assert len(summary["would_have_queries"]) >= 1

    def test_no_discovery_clean_exit(self, tmp_path, monkeypatch):
        """Zero discovery results should produce a clean summary, no crash."""
        import src.agents.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")

        orch = Orchestrator()

        with patch.object(orch, "_discover_topics", return_value=[]):
            summary = orch.run_once()

        assert summary["cycle"] == 1
        assert summary["discovered_topics"] == 0
        assert summary["papers_ingested"] == 0
        assert summary["errors"] == []

    def test_cycle_increments_across_runs(self, tmp_path, monkeypatch):
        """Each call to run_once increments the cycle counter."""
        import src.agents.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "PROJECT_DIR", tmp_path)
        monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path / "orchestrator_state.json")
        monkeypatch.setattr(orch_mod, "PID_PATH", tmp_path / "orchestrator.pid")

        orch = Orchestrator()

        with patch.object(orch, "_discover_topics", return_value=[]):
            s1 = orch.run_once()
            s2 = orch.run_once()
            s3 = orch.run_once()

        assert s1["cycle"] == 1
        assert s2["cycle"] == 2
        assert s3["cycle"] == 3
        assert orch._cycle == 3
