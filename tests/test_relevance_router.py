"""Tests for relevance router (Phase 11)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.relevance_router import RelevanceRouter


class TestRelevanceRouterEmbedding:
    """Tests for embedding-based routing."""

    @pytest.fixture
    def router(self):
        return RelevanceRouter(use_llm=False)

    @pytest.fixture
    def sample_summaries(self):
        return {
            0: {"summary": "Titanium surface modifications for bone implant integration, "
                           "studying roughness and hydrophilicity effects on osteoblast adhesion."},
            1: {"summary": "Macrophage polarization in wound healing — M1 vs M2 phenotypes "
                           "and cytokine signaling pathways including IL-4 and IL-13."},
            2: {"summary": "Adipokine signaling in obesity-related inflammation, focusing on "
                           "leptin and adiponectin effects on immune cell function."},
        }

    def test_routing_empty_summaries(self, router):
        result = router.route("test query", {})
        assert result["relevant_communities"] == []
        assert result["scores"] == {}
        assert result["method"] == "embedding"

    def test_routing_matches_biomaterial_query(self, router, sample_summaries):
        result = router.route("titanium implant surface roughness and bone healing",
                               sample_summaries)
        scores = result["scores"]
        assert scores[0] > scores[1]  # biomaterial community should rank highest
        assert scores[0] > scores[2]

    def test_routing_matches_immune_query(self, router, sample_summaries):
        result = router.route("macrophage polarization and inflammatory cytokines",
                               sample_summaries)
        scores = result["scores"]
        assert scores[1] > scores[0]  # immune community should rank highest

    def test_routing_matches_obesity_query(self, router, sample_summaries):
        result = router.route("leptin signaling in obese adipose tissue",
                               sample_summaries)
        scores = result["scores"]
        assert scores[2] > scores[0]  # obesity community should rank highest

    def test_routing_threshold_filters(self, router, sample_summaries):
        result = router.route("completely unrelated topic about quantum physics",
                               sample_summaries)
        assert result["method"] in ("embedding", "llm_fallback")
        # No community should score high for a quantum physics query
        max_score = max(result["scores"].values()) if result["scores"] else 0
        assert max_score < 0.4  # well below typical relevance threshold

    def test_routing_returns_all_scores(self, router, sample_summaries):
        result = router.route("biomaterial immune response cross-talk",
                               sample_summaries)
        assert len(result["scores"]) == 3
        assert all(isinstance(s, float) for s in result["scores"].values())

    def test_routing_custom_threshold(self, router, sample_summaries):
        result = router.route("general biomedical research question",
                               sample_summaries, threshold=0.1)
        assert result["threshold"] == 0.1

    def test_routing_returns_structure(self, router, sample_summaries):
        result = router.route("test query", sample_summaries)
        assert "relevant_communities" in result
        assert "scores" in result
        assert "method" in result
        assert "threshold" in result
        assert isinstance(result["relevant_communities"], list)
        assert isinstance(result["scores"], dict)


class TestRelevanceRouterLLM:
    """Tests for LLM-based routing fallback."""

    def test_llm_routing_parses_json(self):
        router = RelevanceRouter(use_llm=True)

        summaries = {
            0: {"summary": "Biomaterial surface chemistry effects on protein adsorption."},
            1: {"summary": "T cell receptor signaling pathways in autoimmune disease."},
        }

        with patch.object(router, "_route_by_llm", wraps=router._route_by_llm):
            with patch("src.agents.relevance_router.get_cache") as mock_cache:
                mock_cache_instance = MagicMock()
                mock_cache_instance.get.return_value = '{"0": 0.9, "1": 0.1}'
                mock_cache.return_value = mock_cache_instance

                result = router.route("biomaterial surface protein adsorption study", summaries)
                assert result["method"] == "llm"
                assert len(result["scores"]) == 2

    def test_llm_parse_fallback_to_embedding(self):
        router = RelevanceRouter(use_llm=True)

        summaries = {
            0: {"summary": "Titanium implant osseointegration."},
            1: {"summary": "Neutrophil extracellular traps in sepsis."},
        }

        # Patch the LLM call inside _route_by_llm to fail
        with patch("src.agents.relevance_router.get_chat_model", side_effect=Exception("LLM failed")):
            result = router.route("test", summaries)
            assert result is not None
            assert "relevant_communities" in result

    def test_parse_routing_invalid_json(self):
        router = RelevanceRouter(use_llm=False)
        result = router._parse_routing("not valid json", [0, 1], 0.35)
        assert "relevant_communities" in result
        assert "scores" in result


class TestRelevanceRouterEdgeCases:
    """Edge case tests."""

    def test_routing_single_community(self):
        router = RelevanceRouter(use_llm=False)
        summaries = {0: {"summary": "All about titanium implants and bone healing."}}
        result = router.route("titanium implant study", summaries)
        assert len(result["scores"]) == 1

    def test_routing_empty_query(self):
        router = RelevanceRouter(use_llm=False)
        summaries = {0: {"summary": "Some research area."}}
        result = router.route("", summaries)
        assert len(result["scores"]) == 1
