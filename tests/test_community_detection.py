"""Tests for community detection on the knowledge graph (Phase 11)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import networkx as nx
import pytest

from src.graph.community_detection import (
    detect_communities,
    get_community_papers,
    get_community_entities,
    _run_detection,
    _get_undirected_graph,
)


class _MockGraphStorage:
    """Minimal mock of NetworkXJSONStorage for testing."""

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


def _make_toy_graph(n_nodes=50, n_edges=80, seed=42):
    """Create a toy DiGraph with realistic entity metadata."""
    rng = __import__("random").Random(seed)
    g = nx.DiGraph()
    entity_types = ["cytokine", "cell_type", "material", "method", "finding", "model_system"]
    papers = ["paper_a.pdf", "paper_b.pdf", "paper_c.pdf", "paper_d.pdf"]

    for i in range(n_nodes):
        etype = rng.choice(entity_types)
        paper = rng.choice(papers)
        g.add_node(
            f"{etype}:entity_{i}",
            node_type=etype,
            source_paper=paper,
            evidence=f"Evidence for entity_{i} in context X",
        )

    node_ids = list(g.nodes())
    for _ in range(n_edges):
        u = rng.choice(node_ids)
        v = rng.choice(node_ids)
        if u != v and not g.has_edge(u, v):
            g.add_edge(u, v, relation="co_occurs_with", source_paper=rng.choice(papers))

    return g


class TestCommunityDetection:
    """Tests for core detection functions."""

    def test_run_detection_toy_graph(self):
        g = _make_toy_graph(50, 80)
        storage = _MockGraphStorage()
        storage._graph = g

        result = _run_detection(storage)

        assert result["algorithm"] == "louvain"
        assert result["n_nodes"] == 50
        assert result["n_communities"] >= 1
        assert result["modularity"] >= -0.5  # basic sanity
        assert len(result["node_to_community"]) == 50
        assert sum(result["community_sizes"].values()) == 50
        assert result["community_nodes"]

    def test_run_detection_small_graph(self):
        """Graph with < 3 nodes returns empty communities."""
        g = nx.DiGraph()
        g.add_node("a", node_type="test")
        g.add_node("b", node_type="test")
        g.add_edge("a", "b")
        storage = _MockGraphStorage()
        storage._graph = g

        result = _run_detection(storage)

        assert result["n_communities"] == 0
        assert result["n_nodes"] == 2
        assert result["community_sizes"] == {}
        assert result["node_to_community"] == {}

    def test_run_detection_empty_graph(self):
        g = nx.DiGraph()
        storage = _MockGraphStorage()
        storage._graph = g

        result = _run_detection(storage)

        assert result["n_nodes"] == 0
        assert result["n_communities"] == 0

    def test_detect_communities_cache_hit(self, tmp_path):
        cache_path = tmp_path / "communities.json"
        cached = {"algorithm": "louvain", "n_nodes": 100, "n_communities": 5,
                   "node_to_community": {"a": 0}, "community_sizes": {0: 1},
                   "community_nodes": {0: ["a"]}, "modularity": 0.5}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        g = _make_toy_graph(30, 40)
        storage = _MockGraphStorage()
        storage._graph = g

        result = detect_communities(storage, cache_path=cache_path, force_recompute=False)
        assert result["n_communities"] == 5
        assert result["n_nodes"] == 100

    def test_detect_communities_force_recompute(self, tmp_path):
        cache_path = tmp_path / "communities.json"
        cached = {"algorithm": "louvain", "n_nodes": 5, "n_communities": 1,
                   "node_to_community": {}, "community_sizes": {}, "community_nodes": {},
                   "modularity": 0.0}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        g = _make_toy_graph(30, 40)
        storage = _MockGraphStorage()
        storage._graph = g

        result = detect_communities(storage, cache_path=cache_path, force_recompute=True)
        assert result["n_nodes"] == 30  # recomputed, not cached

    def test_detect_communities_corrupt_cache(self, tmp_path):
        cache_path = tmp_path / "communities.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("not json!!")

        g = _make_toy_graph(30, 40)
        storage = _MockGraphStorage()
        storage._graph = g

        result = detect_communities(storage, cache_path=cache_path)
        assert result["n_nodes"] == 30  # recomputed


class TestCommunityPapers:
    """Tests for get_community_papers."""

    def test_get_community_papers(self):
        g = nx.DiGraph()
        g.add_node("cytokine:IL-6", node_type="cytokine", source_paper="paper_a.pdf")
        g.add_node("cell_type:macrophage", node_type="cell_type", source_paper="paper_a.pdf")
        g.add_node("material:TiO2", node_type="material", source_paper="paper_b.pdf")
        g.add_node("method:ELISA", node_type="method", source_paper="paper_a.pdf")
        storage = _MockGraphStorage()
        storage._graph = g

        community_data = {
            "node_to_community": {
                "cytokine:IL-6": 0,
                "cell_type:macrophage": 0,
                "material:TiO2": 1,
                "method:ELISA": 0,
            },
            "community_nodes": {0: ["cytokine:IL-6", "cell_type:macrophage", "method:ELISA"],
                                1: ["material:TiO2"]},
        }

        papers = get_community_papers(community_data, storage)
        assert papers[0] == ["paper_a.pdf"]
        assert papers[1] == ["paper_b.pdf"]

    def test_get_community_papers_no_source(self):
        g = nx.DiGraph()
        g.add_node("x:no_paper", node_type="x")
        storage = _MockGraphStorage()
        storage._graph = g

        community_data = {
            "node_to_community": {"x:no_paper": 0},
            "community_nodes": {0: ["x:no_paper"]},
        }

        papers = get_community_papers(community_data, storage)
        assert papers[0] == []


class TestCommunityEntities:
    """Tests for get_community_entities."""

    def test_get_community_entities(self):
        g = nx.DiGraph()
        g.add_node("cytokine:IL-6", node_type="cytokine", source_paper="p1",
                    evidence="IL-6 elevated in obese mice")
        g.add_node("material:TiO2", node_type="material", source_paper="p2",
                    evidence="TiO2 coating increased osteoblast adhesion")
        storage = _MockGraphStorage()
        storage._graph = g

        community_data = {
            "node_to_community": {"cytokine:IL-6": 0, "material:TiO2": 1},
        }

        entities = get_community_entities(community_data, storage)
        assert len(entities[0]) == 1
        assert entities[0][0]["node_id"] == "cytokine:IL-6"
        assert entities[0][0]["node_type"] == "cytokine"
        assert entities[0][0]["source_paper"] == "p1"
        assert entities[0][0]["evidence"] == "IL-6 elevated in obese mice"
        assert len(entities[1]) == 1

    def test_get_community_entities_empty(self):
        storage = _MockGraphStorage()
        community_data = {"node_to_community": {}}
        entities = get_community_entities(community_data, storage)
        assert entities == {}


class TestGetUndirectedGraph:
    """Tests for _get_undirected_graph."""

    def test_converts_directed_to_undirected(self):
        g = nx.DiGraph()
        g.add_node("a", node_type="test")
        g.add_node("b", node_type="test")
        g.add_edge("a", "b")
        storage = _MockGraphStorage()
        storage._graph = g

        ug = _get_undirected_graph(storage)
        assert isinstance(ug, nx.Graph)
        assert ug.has_edge("a", "b") or ug.has_edge("b", "a")
