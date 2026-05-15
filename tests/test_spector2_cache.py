"""Unit tests for SPECTER2 embedding cache."""
import json
import pytest
from pathlib import Path
from src.utils.spector2_cache import Spector2Cache


@pytest.fixture
def temp_cache_path(tmp_path):
    return tmp_path / "spector2_cache.json"


@pytest.fixture
def cache(temp_cache_path):
    return Spector2Cache(cache_path=temp_cache_path)


class TestSpector2Cache:
    def test_init_empty(self, cache):
        """New cache starts empty."""
        assert cache.stats() == {"total_entries": 0, "with_embedding": 0}

    def test_put_and_get(self, cache):
        """Store and retrieve a valid embedding."""
        emb = [0.1] * 768
        cache.put("10.1016/test.2021.01", "s2_id_123", emb)
        assert cache.has("10.1016/test.2021.01")
        retrieved = cache.get("10.1016/test.2021.01")
        assert retrieved == emb

    def test_get_missing(self, cache):
        """Missing DOI returns None."""
        assert cache.get("nonexistent.doi") is None
        assert not cache.has("nonexistent.doi")

    def test_put_null_embedding(self, cache):
        """Null embedding is not stored."""
        cache.put("10.1016/null.001", "s2_id", None)
        assert not cache.has("10.1016/null.001")

    def test_put_wrong_dimension(self, cache):
        """Embedding with wrong dimension is not stored."""
        cache.put("10.1016/wrong.001", "s2_id", [0.1] * 384)
        assert not cache.has("10.1016/wrong.001")

    def test_put_empty_dimension(self, cache):
        """Empty embedding is not stored."""
        cache.put("10.1016/empty.001", "s2_id", [])
        assert not cache.has("10.1016/empty.001")

    def test_case_insensitive_doi(self, cache):
        """DOI lookup is case-insensitive."""
        emb = [0.2] * 768
        cache.put("10.1016/UPPER.001", "s2_upper", emb)
        assert cache.get("10.1016/upper.001") == emb
        assert cache.get("10.1016/UPPER.001") == emb

    def test_doi_stripping(self, cache):
        """DOI values are stripped before storage."""
        emb = [0.3] * 768
        cache.put("  10.1016/spaces.001  ", "s2_spaces", emb)
        assert cache.has("10.1016/spaces.001")

    def test_flush_persists(self, temp_cache_path):
        """Cache persists to disk after flush."""
        c1 = Spector2Cache(cache_path=temp_cache_path)
        emb = [0.4] * 768
        c1.put("10.1016/persist.001", "s2_persist", emb)
        c1.flush()

        c2 = Spector2Cache(cache_path=temp_cache_path)
        assert c2.has("10.1016/persist.001")
        assert c2.get("10.1016/persist.001") == emb

    def test_corrupted_json_recovery(self, temp_cache_path):
        """Corrupted cache file is handled gracefully."""
        temp_cache_path.write_text("{invalid json")
        cache = Spector2Cache(cache_path=temp_cache_path)
        assert cache.stats() == {"total_entries": 0, "with_embedding": 0}

    def test_non_dict_json_recovery(self, temp_cache_path):
        """Non-dict JSON file is handled gracefully."""
        temp_cache_path.write_text("[1, 2, 3]")
        cache = Spector2Cache(cache_path=temp_cache_path)
        assert cache.stats() == {"total_entries": 0, "with_embedding": 0}

    def test_multiple_entries(self, cache):
        """Multiple DOIs can be cached."""
        for i in range(5):
            cache.put(f"10.1016/multi.{i:03d}", f"s2_{i}", [float(i)] * 768)
        assert cache.stats()["total_entries"] == 5
        assert cache.stats()["with_embedding"] == 5

    def test_stats_accuracy(self, cache):
        """Stats count only entries with embeddings."""
        emb = [0.5] * 768
        cache.put("10.1016/with.emb", "s2_with", emb)
        # The put method only stores if embedding is valid,
        # so total_entries == with_embedding when all have embeddings
        assert cache.stats()["total_entries"] == 1
        assert cache.stats()["with_embedding"] == 1
