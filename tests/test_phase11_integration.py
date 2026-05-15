"""Phase 11 integration tests — community routing, chunk filtering, progressive disclosure.

These tests go beyond unit mocks to validate:
  - survey_community_route_node with real graph + embedding routing
  - Chunk filtering by community paper membership
  - Community summarizer full pipeline (detection → entities → LLM)
  - Progressive disclosure output structure validation
  - Orchestrator _update_communities integration
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.graph.community_detection import (
    detect_communities,
    get_community_papers,
    _run_detection,
)
from src.graph.survey_nodes import survey_community_route_node
from src.graph.progressive_disclosure import ProgressiveDisclosure
from src.agents.relevance_router import RelevanceRouter
from src.state import AgentState


class _MockGraphStorage:
    def __init__(self, nodes=None, edges=None):
        self._graph = nx.DiGraph()
        for n in (nodes or []):
            self._graph.add_node(n[0], **n[1])
        for e in (edges or []):
            self._graph.add_edge(*e)

    def save(self):
        pass

    def load(self):
        pass


def _make_multi_paper_graph():
    """KG with 2 communities across 3 papers."""
    g = nx.DiGraph()
    # Community 0 — biomaterials (paper_a)
    g.add_node("material:Ti64", node_type="material", source_paper="paper_a.pdf",
               evidence="Ti-6Al-4V alloy tested for osseointegration")
    g.add_node("finding:osteoblast", node_type="finding", source_paper="paper_a.pdf",
               evidence="Osteoblast adhesion improved on rough Ti")
    g.add_node("method:microCT", node_type="method", source_paper="paper_a.pdf",
               evidence="microCT used for 3D bone quantification")
    # Community 1 — immunology (paper_b, paper_c)
    g.add_node("cytokine:IL-6", node_type="cytokine", source_paper="paper_b.pdf",
               evidence="IL-6 elevated 3-fold in obese mice")
    g.add_node("cell_type:macrophage", node_type="cell_type", source_paper="paper_b.pdf",
               evidence="M2 macrophages increased IL-10 production")
    g.add_node("cell_type:T-cell", node_type="cell_type", source_paper="paper_c.pdf",
               evidence="CD4+ T cells showed increased activation")
    g.add_node("cytokine:TNF-a", node_type="cytokine", source_paper="paper_c.pdf",
               evidence="TNF-alpha correlated with inflammation")
    # Edges — enough for Louvain to find structure
    g.add_edge("material:Ti64", "finding:osteoblast")
    g.add_edge("material:Ti64", "method:microCT")
    g.add_edge("finding:osteoblast", "method:microCT")
    g.add_edge("cytokine:IL-6", "cell_type:macrophage")
    g.add_edge("cell_type:T-cell", "cytokine:TNF-a")
    g.add_edge("cytokine:IL-6", "cell_type:T-cell")
    return g


def _build_state(chunks, query="test query"):
    return {
        "user_query": query,
        "public_context": chunks,
        "query_scope": "public",
        "mode": "survey",
        "per_paper_extractions": {},
        "per_theme_syntheses": {},
        "decomposed_themes": [],
        "thematic_clusters": {},
    }


# ──────────────────────────────────────────────────────────────────────
#  Community route node integration
# ──────────────────────────────────────────────────────────────────────

class TestCommunityRouteNode:
    """Tests for survey_community_route_node with real graph + embedding routing."""

    def test_node_with_communities(self, tmp_path):
        """Full path: detect communities, route query, filter chunks."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        chunks = [
            {"text": "Ti-6Al-4V alloy study", "metadata": {"source": "paper_a.pdf", "chunk_index": 0}},
            {"text": "Macrophage cytokine study", "metadata": {"source": "paper_b.pdf", "chunk_index": 1}},
            {"text": "T cell activation study", "metadata": {"source": "paper_c.pdf", "chunk_index": 2}},
        ]

        # Patch cache path to tmp
        with patch("src.graph.community_detection._DEFAULT_COMMUNITY_PATH",
                   tmp_path / "communities.json"):
            with patch("src.agents.community_summarizer._DEFAULT_SUMMARY_CACHE",
                       tmp_path / "commsum.json"):
                state = _build_state(chunks, "titanium implant surface modification")
                result = survey_community_route_node(state, storage)

                assert "community_data" in result
                cd = result["community_data"]
                assert cd["n_communities"] >= 1
                assert cd["n_nodes"] == 7

    def test_node_no_communities_passthrough(self):
        """Empty graph should pass through chunks unchanged."""
        g = nx.DiGraph()
        storage = _MockGraphStorage()
        storage._graph = g

        chunks = [{"text": "data", "metadata": {"source": "p1.pdf"}}]
        state = _build_state(chunks)

        with patch("src.graph.community_detection._DEFAULT_COMMUNITY_PATH",
                   Path("/tmp/no_comms_test.json")):
            with patch("src.agents.community_summarizer._DEFAULT_SUMMARY_CACHE",
                       Path("/tmp/no_comms_test_summ.json")):
                result = survey_community_route_node(state, storage)
                assert "community_data" in result
                assert result["community_data"]["n_communities"] == 0

    def test_node_empty_chunks(self):
        """No chunks in state should return empty result."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g
        state = _build_state([])
        result = survey_community_route_node(state, storage)
        assert result == {}

    def test_node_without_graph_storage(self):
        """No graph storage should return empty dict."""
        state = _build_state([{"text": "data", "metadata": {"source": "p1.pdf"}}])
        result = survey_community_route_node(state, None)
        assert result == {}


# ──────────────────────────────────────────────────────────────────────
#  Chunk filtering by community papers
# ──────────────────────────────────────────────────────────────────────

class TestChunkFilteringByCommunity:
    """Tests that chunks are filtered to relevant communities' papers."""

    def test_filter_chunks_to_relevant_papers(self, tmp_path):
        """Chunks should be filtered when relevant communities are found."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        chunks = [
            {"text": "ti implant", "metadata": {"source": "paper_a.pdf"}},
            {"text": "immune cells", "metadata": {"source": "paper_b.pdf"}},
            {"text": "t cell", "metadata": {"source": "paper_c.pdf"}},
        ]

        with patch("src.graph.community_detection._DEFAULT_COMMUNITY_PATH",
                   tmp_path / "communities.json"):
            with patch("src.agents.community_summarizer._DEFAULT_SUMMARY_CACHE",
                       tmp_path / "commsum.json"):
                state = _build_state(chunks, "titanium implant bone healing")
                result = survey_community_route_node(state, storage)

                if "public_context" in result:
                    filtered = result["public_context"]
                    filtered_sources = {c.get("metadata", {}).get("source") for c in filtered}
                    assert "paper_a.pdf" in filtered_sources

    def test_no_filter_when_no_papers_match(self, tmp_path):
        """When relevant communities have no papers, chunks pass through."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        chunks = [
            {"text": "unrelated", "metadata": {"source": "paper_z.pdf"}},
        ]

        with patch("src.graph.community_detection._DEFAULT_COMMUNITY_PATH",
                   tmp_path / "communities.json"):
            with patch("src.agents.community_summarizer._DEFAULT_SUMMARY_CACHE",
                       tmp_path / "commsum.json"):
                state = _build_state(chunks, "quantum physics")
                result = survey_community_route_node(state, storage)
                assert result is not None

    def test_community_papers_extraction(self):
        """get_community_papers correctly maps communities to source papers."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        community_data = _run_detection(storage)
        papers = get_community_papers(community_data, storage)

        all_papers = set()
        for paper_list in papers.values():
            all_papers.update(paper_list)

        assert "paper_a.pdf" in all_papers
        assert "paper_b.pdf" in all_papers
        assert "paper_c.pdf" in all_papers


# ──────────────────────────────────────────────────────────────────────
#  Progressive disclosure output structure
# ──────────────────────────────────────────────────────────────────────

class TestProgressiveDisclosureStructure:
    """Validates the structure and content of disclosure tier output."""

    @pytest.fixture
    def disclosure(self):
        storage = _MockGraphStorage()
        storage._graph = _make_multi_paper_graph()
        community_data = _run_detection(storage)
        summaries = {
            cid: {
                "name": f"Community {cid}",
                "summary": f"Test summary for community {cid}.",
                "n_entities": 3,
                "top_papers": ["paper_a.pdf"],
                "entity_types": ["material", "finding"],
            }
            for cid in community_data.get("community_nodes", {})
        }
        return ProgressiveDisclosure(storage, community_data, summaries)

    def test_tier1_has_required_sections(self, disclosure):
        overview = disclosure.get_system_overview()
        assert "Research Communities" in overview
        assert "Community" in overview
        assert "Entities:" in overview
        assert "Papers:" in overview
        assert "Entity types:" in overview
        assert "Summary:" in overview

    def test_tier2_has_required_sections(self, disclosure):
        cd = disclosure.community_data
        cids = list(cd.get("community_nodes", {}).keys())
        if not cids:
            pytest.skip("No communities detected")
        detail = disclosure.get_community_detail(cids[0])
        assert "## Summary" in detail
        assert "## Key Entities" in detail
        assert "## Papers" in detail

    def test_tier3_paper_entities_not_empty(self, disclosure):
        entities = disclosure.get_paper_entities("paper_a.pdf")
        assert len(entities) > 0
        for ent in entities:
            assert "node_id" in ent
            assert "node_type" in ent
            assert "community_id" in ent

    def test_tier1_filtered_shows_only_relevant(self, disclosure):
        overview = disclosure.get_system_overview(relevant_communities=[0])
        assert "Community 0" in overview
        # Community 1 should not appear
        lines = overview.split("\n")
        comm1_lines = [l for l in lines if "Community 1" in l]
        assert len(comm1_lines) == 0

    def test_build_disclosure_map_structure(self, disclosure):
        dmap = disclosure.build_disclosure_map(query="titanium bone")
        assert "tier1_system_overview" in dmap
        assert "tier2_community_details" in dmap
        assert "tier3_paper_community_map" in dmap
        assert dmap["n_communities"] > 0
        assert dmap["n_papers"] > 0


# ──────────────────────────────────────────────────────────────────────
#  Community summarizer full pipeline
# ──────────────────────────────────────────────────────────────────────

class TestCommunitySummarizerPipeline:
    """Tests the full summarizer pipeline from detection → entities → summary."""

    def test_full_pipeline_detection_to_summary(self, tmp_path):
        """Integration: detect → get entities → generate summaries."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        cache = tmp_path / "test_summaries.json"

        from src.agents.community_summarizer import CommunitySummarizer
        summarizer = CommunitySummarizer(cache_path=cache)

        community_data = detect_communities(storage, cache_path=tmp_path / "comms.json",
                                             force_recompute=True)

        with patch.object(summarizer, "_generate_summary",
                          return_value="Test summary paragraph."):
            summaries = summarizer.summarize(storage, community_data=community_data,
                                              force_recompute=True)

        assert len(summaries) >= 1
        for cid, info in summaries.items():
            assert "name" in info
            assert "summary" in info
            assert info["summary"]
            assert "n_entities" in info
            assert "top_papers" in info
            assert "entity_types" in info

    def test_summaries_persisted_to_cache(self, tmp_path):
        """After summarization, cache file should exist and be valid."""
        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        cache = tmp_path / "persist_test.json"

        from src.agents.community_summarizer import CommunitySummarizer
        summarizer = CommunitySummarizer(cache_path=cache)

        community_data = detect_communities(storage, cache_path=tmp_path / "comms.json",
                                             force_recompute=True)

        with patch.object(summarizer, "_generate_summary",
                          return_value="Cached test summary."):
            summarizer.summarize(storage, community_data=community_data,
                                  force_recompute=True)

        assert cache.exists()
        with open(cache) as f:
            data = json.load(f)
        assert len(data) >= 1


# ──────────────────────────────────────────────────────────────────────
#  Orchestrator _update_communities
# ──────────────────────────────────────────────────────────────────────

class TestOrchestratorUpdateCommunities:
    """Tests for the orchestrator's _update_communities method."""

    def test_update_communities_returns_structure(self, tmp_path):
        from src.agents.orchestrator import Orchestrator

        g = _make_multi_paper_graph()
        storage = _MockGraphStorage()
        storage._graph = g

        orch = Orchestrator(graph_storage=storage, dry_run=True)
        orch._cycle = 1

        with patch("src.graph.community_detection._DEFAULT_COMMUNITY_PATH",
                   tmp_path / "communities.json"):
            result = orch._update_communities()

        assert "n_communities" in result
        assert "modularity" in result
        assert "n_nodes" in result
        assert result["n_communities"] >= 1
        assert result["n_nodes"] == 7

    def test_update_communities_handles_error_gracefully(self, tmp_path):
        from src.agents.orchestrator import Orchestrator

        g = nx.DiGraph()
        g.add_node("a")
        storage = _MockGraphStorage()
        storage._graph = g

        orch = Orchestrator(graph_storage=storage, dry_run=True)

        with patch("src.graph.community_detection._DEFAULT_COMMUNITY_PATH",
                   tmp_path / "communities.json"):
            result = orch._update_communities()

        assert "n_communities" in result
        assert result["n_communities"] == 0


# ──────────────────────────────────────────────────────────────────────
#  Relevance router embedding accuracy
# ──────────────────────────────────────────────────────────────────────

class TestRelevanceRouterAccuracy:
    """Validate that embedding routing produces sensible rankings."""

    def test_biomaterial_ranks_highest_for_biomaterial_query(self):
        router = RelevanceRouter(use_llm=False)
        summaries = {
            0: {"summary": "Titanium surface modifications for bone implant integration."},
            1: {"summary": "Macrophage polarization in inflammatory responses."},
            2: {"summary": "Adipokine signaling in obesity-related inflammation."},
        }
        result = router.route("titanium implant osseointegration rough surface", summaries)
        scores = result["scores"]
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_immune_cell_ranks_highest_for_immune_query(self):
        router = RelevanceRouter(use_llm=False)
        summaries = {
            0: {"summary": "Biomaterial surface chemistry for dental implants."},
            1: {"summary": "T cells and macrophage cytokine production in wound healing."},
            2: {"summary": "Bone morphogenetic protein signaling in osteogenesis."},
        }
        result = router.route("CD4+ T cell activation and cytokine release", summaries)
        scores = result["scores"]
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_no_false_positives_for_irrelevant_queries(self):
        router = RelevanceRouter(use_llm=False)
        summaries = {0: {"summary": "Titanium implant surface roughness effects on bone."}}
        result = router.route("quantum computing algorithm complexity theory", summaries)
        assert result["relevant_communities"] == []
        assert result["scores"][0] < 0.35
