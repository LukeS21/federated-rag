"""Tests for progressive disclosure system (Phase 11)."""

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.graph.progressive_disclosure import ProgressiveDisclosure


class _MockGraphStorage:
    """Minimal mock of NetworkXJSONStorage."""

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


def _make_toy_storage():
    """Create a toy graph storage with 3 communities across 2 papers."""
    g = nx.DiGraph()
    g.add_node("cytokine:IL-6", node_type="cytokine", source_paper="paper_a.pdf",
               evidence="IL-6 elevated in obese mice")
    g.add_node("cell_type:macrophage", node_type="cell_type", source_paper="paper_a.pdf",
               evidence="M2 macrophages increased at implant surface")
    g.add_node("material:Ti64", node_type="material", source_paper="paper_a.pdf",
               evidence="Ti-6Al-4V alloy tested for osseointegration")
    g.add_node("cytokine:TNF-a", node_type="cytokine", source_paper="paper_b.pdf",
               evidence="TNF-alpha levels correlated with inflammation")
    g.add_node("method:ELISA", node_type="method", source_paper="paper_b.pdf",
               evidence="ELISA used to quantify cytokine levels")
    g.add_node("finding:osteoblast", node_type="finding", source_paper="paper_a.pdf",
               evidence="Osteoblast adhesion improved on rough Ti")
    g.add_edge("cytokine:IL-6", "cell_type:macrophage")
    g.add_edge("cell_type:macrophage", "material:Ti64")
    g.add_edge("cytokine:TNF-a", "method:ELISA")
    g.add_edge("material:Ti64", "finding:osteoblast")
    return _MockGraphStorage(
        nodes=[(n, dict(g.nodes[n])) for n in g.nodes()],
        edges=list(g.edges()),
    )


class TestProgressiveDisclosure:
    """Tests for the three-tier disclosure system."""

    @pytest.fixture
    def storage(self):
        return _make_toy_storage()

    @pytest.fixture
    def community_data(self):
        return {
            "algorithm": "louvain",
            "n_nodes": 6,
            "n_communities": 2,
            "modularity": 0.42,
            "community_sizes": {0: 3, 1: 3},
            "node_to_community": {
                "cytokine:IL-6": 0,
                "cell_type:macrophage": 0,
                "material:Ti64": 0,
                "cytokine:TNF-a": 1,
                "method:ELISA": 1,
                "finding:osteoblast": 0,
            },
            "community_nodes": {
                0: ["cytokine:IL-6", "cell_type:macrophage", "material:Ti64", "finding:osteoblast"],
                1: ["cytokine:TNF-a", "method:ELISA"],
            },
        }

    @pytest.fixture
    def community_summaries(self):
        return {
            0: {
                "name": "Community 0",
                "summary": "Biomaterial-immune interactions at the implant interface.",
                "n_entities": 3,
                "top_papers": ["paper_a.pdf"],
                "entity_types": ["cytokine", "cell_type", "material"],
            },
            1: {
                "name": "Community 1",
                "summary": "Cytokine quantification methods and osteoblast findings.",
                "n_entities": 3,
                "top_papers": ["paper_b.pdf"],
                "entity_types": ["cytokine", "method", "finding"],
            },
        }

    def test_get_system_overview_all(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        overview = pd.get_system_overview()
        assert "Research Communities Overview" in overview
        assert "Community 0" in overview
        assert "Community 1" in overview
        assert "paper_a.pdf" not in overview  # Tier 1 is high-level only

    def test_get_system_overview_filtered(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        overview = pd.get_system_overview(relevant_communities=[0])
        assert "Community 0" in overview
        assert "Community 1" not in overview

    def test_get_system_overview_empty(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        overview = pd.get_system_overview(relevant_communities=[])
        assert "No research communities" in overview

    def test_get_community_detail(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        detail = pd.get_community_detail(0)
        assert "Community 0" in detail
        assert "Biomaterial-immune interactions" in detail
        assert "cytokine:IL-6" in detail
        assert "paper_a.pdf" in detail

    def test_get_community_detail_missing(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        detail = pd.get_community_detail(99)
        assert "Community 99" in detail
        assert "No summary available" in detail

    def test_get_paper_entities(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        entities = pd.get_paper_entities("paper_a.pdf")
        assert len(entities) == 4  # IL-6, macrophage, Ti64, osteoblast from paper_a
        assert all(e["community_id"] == 0 for e in entities)

    def test_get_paper_entities_no_match(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        entities = pd.get_paper_entities("nonexistent.pdf")
        assert entities == []

    def test_build_disclosure_map(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        dmap = pd.build_disclosure_map(query="test query")
        assert "tier1_system_overview" in dmap
        assert "tier2_community_details" in dmap
        assert "tier3_paper_community_map" in dmap
        assert dmap["n_communities"] == 2
        assert dmap["n_papers"] > 0

    def test_build_disclosure_map_filtered(self, storage, community_data, community_summaries):
        pd = ProgressiveDisclosure(storage, community_data, community_summaries)
        dmap = pd.build_disclosure_map(relevant_communities=[0])
        assert dmap["n_communities"] == 1

    def test_disclosure_without_summaries(self, storage, community_data):
        """Should work without pre-computed community summaries."""
        pd = ProgressiveDisclosure(storage, community_data, {})
        overview = pd.get_system_overview()
        assert "No description available" in overview or "Research Communities" in overview

    def test_disclosure_no_community_data(self, storage):
        """Should gracefully handle missing community data."""
        pd = ProgressiveDisclosure(storage, None, {})
        overview = pd.get_system_overview()
        assert isinstance(overview, str)
