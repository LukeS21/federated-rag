"""Tests for community summarizer (Phase 11)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.community_summarizer import CommunitySummarizer


class _MockGraphStorage:
    def __init__(self, graph=None):
        import networkx as nx
        self._graph = graph or nx.DiGraph()

    def save(self):
        pass

    def load(self):
        pass


def _make_toy_graph():
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("cytokine:IL-6", node_type="cytokine", source_paper="p1.pdf",
               evidence="IL-6 increased 3-fold in obese mice")
    g.add_node("cell_type:macrophage", node_type="cell_type", source_paper="p1.pdf",
               evidence="M2 macrophages showed increased IL-10 production")
    g.add_node("material:TiO2", node_type="material", source_paper="p2.pdf",
               evidence="TiO2 nanotubes enhanced osteoblast differentiation")
    g.add_node("method:microCT", node_type="method", source_paper="p2.pdf",
               evidence="microCT used for 3D bone volume quantification")
    g.add_edge("cytokine:IL-6", "cell_type:macrophage")
    g.add_edge("material:TiO2", "method:microCT")
    return _MockGraphStorage(g)


class TestCommunitySummarizer:
    """Tests for CommunitySummarizer."""

    @pytest.fixture
    def community_data(self):
        return {
            "algorithm": "louvain",
            "n_nodes": 4,
            "n_communities": 2,
            "modularity": 0.35,
            "community_sizes": {0: 2, 1: 2},
            "node_to_community": {
                "cytokine:IL-6": 0,
                "cell_type:macrophage": 0,
                "material:TiO2": 1,
                "method:microCT": 1,
            },
            "community_nodes": {
                0: ["cytokine:IL-6", "cell_type:macrophage"],
                1: ["material:TiO2", "method:microCT"],
            },
        }

    def test_summarize_no_communities(self):
        summarizer = CommunitySummarizer(cache_path=Path("/tmp/test_empty_comm.json"))
        storage = _MockGraphStorage()

        community_data = {
            "algorithm": "louvain",
            "n_nodes": 0,
            "n_communities": 0,
            "community_sizes": {},
            "node_to_community": {},
            "community_nodes": {},
        }

        summaries = summarizer.summarize(storage, community_data=community_data,
                                          force_recompute=True)
        assert summaries == {}

    def test_summarize_with_mock_llm(self, community_data):
        summarizer = CommunitySummarizer(cache_path=Path("/tmp/test_community_summaries.json"))

        storage = _make_toy_graph()

        with patch.object(summarizer, "_generate_summary") as mock_gen:
            mock_gen.return_value = "Test summary paragraph about biomaterials."

            summaries = summarizer.summarize(storage, community_data=community_data,
                                              force_recompute=True)
            assert len(summaries) == 2
            assert 0 in summaries
            assert 1 in summaries
            assert summaries[0]["name"] == "Community 0"
            assert "summary" in summaries[0]
            assert summaries[0]["n_entities"] == 2
            assert "entity_types" in summaries[0]
            assert "top_papers" in summaries[0]

    def test_summarize_empty_community(self, community_data):
        summarizer = CommunitySummarizer()

        storage = _make_toy_graph()
        # Clear nodes from one community
        community_data["community_nodes"][0] = []
        community_data["node_to_community"] = {
            "material:TiO2": 1,
            "method:microCT": 1,
        }

        summaries = summarizer.summarize(storage, community_data=community_data,
                                          force_recompute=True)
        assert "No entities" in summaries[0]["summary"] or summaries[0]["n_entities"] == 0

    def test_format_entity_list(self, community_data):
        summarizer = CommunitySummarizer()
        entities = [
            {"node_id": "cytokine:IL-6", "node_type": "cytokine",
             "evidence": "IL-6 increased", "source_paper": "p1.pdf"},
            {"node_id": "cell_type:macrophage", "node_type": "cell_type",
             "evidence": "M2 macrophages", "source_paper": "p1.pdf"},
        ]
        formatted = summarizer._format_entity_list(entities)
        assert "cytokine:IL-6" in formatted or "IL-6" in formatted
        assert "macrophage" in formatted.lower()

    def test_summarize_cache_hit(self, community_data, tmp_path):
        cache_path = tmp_path / "community_summaries.json"
        cached = {
            "0": {"name": "C0", "summary": "Cached summary.", "n_entities": 2,
                  "top_papers": [], "entity_types": []},
            "1": {"name": "C1", "summary": "Cached summary 2.", "n_entities": 2,
                  "top_papers": [], "entity_types": []},
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        summarizer = CommunitySummarizer(cache_path=cache_path)
        storage = _make_toy_graph()

        summaries = summarizer.summarize(storage, community_data=community_data)
        assert summaries[0]["summary"] == "Cached summary."
        assert summaries[1]["summary"] == "Cached summary 2."

    def test_summarize_force_recompute(self, community_data, tmp_path):
        cache_path = tmp_path / "force_comms.json"
        cached = {
            "0": {"name": "C0", "summary": "Old summary", "n_entities": 0,
                  "top_papers": [], "entity_types": []},
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        summarizer = CommunitySummarizer(cache_path=cache_path)
        storage = _make_toy_graph()

        with patch.object(summarizer, "_generate_summary") as mock_gen:
            mock_gen.return_value = "Fresh summary."
            summaries = summarizer.summarize(storage, community_data=community_data,
                                              force_recompute=True)
            assert summaries[0]["summary"] == "Fresh summary."

    def test_generate_summary_real_call(self, community_data):
        """Test _generate_summary with cache mock (no real LLM)."""
        summarizer = CommunitySummarizer()
        with patch("src.agents.community_summarizer.get_cache") as mock_cache:
            cache_instance = MagicMock()
            cache_instance.get.return_value = None
            mock_cache.return_value = cache_instance
            with patch("src.agents.community_summarizer.get_chat_model") as mock_llm_func:
                mock_llm = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "A test summary."
                mock_llm.invoke.return_value = mock_response
                mock_llm_func.return_value = mock_llm

                result = summarizer._generate_summary("Entity: IL-6, Entity: macrophage", 0)
                assert result == "A test summary."

    def test_generate_summary_cache_hit(self, community_data):
        summarizer = CommunitySummarizer()
        with patch("src.agents.community_summarizer.get_cache") as mock_cache:
            cache_instance = MagicMock()
            cache_instance.get.return_value = "Cached paragraph."
            mock_cache.return_value = cache_instance

            result = summarizer._generate_summary("Entity: IL-6", 0)
            assert result == "Cached paragraph."
