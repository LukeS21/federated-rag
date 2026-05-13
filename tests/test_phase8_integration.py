"""
Integration tests for Phase 8 components.

Tests the new infrastructure pieces built in Phase 8:
  - BM25 persistence
  - Claim Ledger O(1) lookup
  - Neo4jStorage (requires Neo4j running, skipped otherwise)
  - CacheStore (SQLite-backed)
  - ClaimIndex (L0 corpus claims)
  - HierarchicalClusterer
  - PDFDownloader
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── BM25 Persistence ──────────────────────────────────────────────────────────

def test_bm25_save_load_roundtrip():
    """BM25 index persists and reloads corpus correctly."""
    from src.retrieval.bm25_index import BM25Index

    with tempfile.TemporaryDirectory() as td:
        idx1 = BM25Index(persist_dir=td)
        idx1.add_documents(["mice were treated with titanium implants",
                           "macrophage polarization was measured by flow cytometry",
                           "IL-6 levels were elevated in obese mice"])
        assert len(idx1) == 3
        idx1.save()

        idx2 = BM25Index(persist_dir=td)
        assert idx2.load() is True
        assert len(idx2) == 3
        results = idx2.query("macrophage polarization", n_results=2)
        assert len(results) == 2
        assert "macrophage" in results[0]


def test_bm25_no_persist_dir():
    """BM25 without persist_dir works as before (no-op save/load)."""
    from src.retrieval.bm25_index import BM25Index
    idx = BM25Index()
    idx.add_documents(["test document"])
    assert idx.load() is False  # no persist dir
    assert len(idx) == 1


# ── Claim Ledger O(1) ─────────────────────────────────────────────────────────

def test_claim_ledger_duplicates_use_index():
    """ClaimLedger uses O(1) dict lookup after Phase 8 upgrade."""
    from src.synthesis.claim_ledger import ClaimLedger
    ledger = ClaimLedger()
    claim_text = "IL-6 elevated in obese mice (@avery2022)"
    ledger.add_claim(claim_text, section="Results",
                     citations=["avery2022"], grounded=True)

    # Exact same text should be detected as duplicate (O(1) via _by_id)
    assert ledger.is_duplicate(claim_text)

    # Case/whitespace normalization should still match
    assert ledger.is_duplicate("  IL-6 elevated in obese mice (@avery2022)  ")

    # Non-duplicate should not match
    assert not ledger.is_duplicate("TNF-alpha was also elevated")

    # Verify _by_id is maintained after add
    assert len(ledger._by_id) == 1


def test_claim_ledger_index_after_load():
    """ClaimLedger._by_id is rebuilt after load()."""
    from src.synthesis.claim_ledger import ClaimLedger
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        f.write('{"claims": []}')
        f.flush()
        fname = f.name

    ledger1 = ClaimLedger(ledger_path=fname)
    ledger1.add_claim("claim one text", section="Intro")
    ledger1.add_claim("claim two text", section="Results")
    ledger1.save()

    ledger2 = ClaimLedger(ledger_path=fname)
    assert len(ledger2._by_id) == 2
    assert ledger2.is_duplicate("claim one text")
    assert ledger2.is_duplicate("claim two text")

    os.unlink(fname)


# ── HybridRetriever include_figures ───────────────────────────────────────────

def test_hybrid_retriever_include_figures_native():
    """HybridRetriever supports include_figures as a native parameter."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.chroma_client import ChromaClient
    from src.retrieval.bm25_index import BM25Index

    chroma = ChromaClient("test_native_figs")
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma, bm25)

    # Ingest text + figure chunks
    chunks = [
        {"text": "IL-6 elevated in obese mice with titanium implants",
         "metadata": {"source": "a.pdf", "chunk_type": "text"}},
        {"text": "Figure showing macrophage polarization via flow cytometry",
         "metadata": {"source": "a.pdf", "chunk_type": "figure"}},
    ]
    retriever.ingest(chunks)

    # Without include_figures
    results = retriever.query("IL-6 macrophage", include_figures=False)
    for r in results:
        assert r.get("metadata", {}).get("chunk_type") != "figure"

    # With include_figures
    results_fig = retriever.query("IL-6 macrophage", include_figures=True)
    assert len(results_fig) > 0


# ── CacheStore (SQLite) ───────────────────────────────────────────────────────

def test_cache_store_basic():
    """CacheStore stores and retrieves values."""
    from src.cache.cache_store import CacheStore
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = CacheStore(f.name)
        store.set({"result": "hello"}, "test_key", ttl_seconds=3600)
        val = store.get("test_key")
        assert val is not None
        assert val["result"] == "hello"
        os.unlink(f.name)


def test_cache_store_ttl_expiry():
    """CacheStore respects TTL."""
    from src.cache.cache_store import CacheStore
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = CacheStore(f.name)
        store.set({"x": 1}, "expired_key", ttl_seconds=0)  # immediate expiry
        val = store.get("expired_key")
        assert val is None
        os.unlink(f.name)


def test_cache_store_invalidate():
    """CacheStore can delete specific entries."""
    from src.cache.cache_store import CacheStore
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = CacheStore(f.name)
        store.set({"a": 1}, "key1")
        store.set({"b": 2}, "key2")
        store.invalidate("key1")
        assert store.get("key1") is None
        assert store.get("key2") is not None
        os.unlink(f.name)


# ── ClaimIndex (L0) ───────────────────────────────────────────────────────────

def test_claim_index_ingest_and_query():
    """ClaimIndex indexes entity claims and retrieves them."""
    from src.retrieval.claim_index import ClaimIndex
    from src.retrieval.chroma_client import ChromaClient

    entities = {
        "cytokines": [
            {"entity": "IL-6", "evidence": "IL-6 was elevated in obese mice with titanium implants",
             "context": "serum measurement"},
            {"entity": "TNF-alpha", "evidence": "TNF-alpha increased in response to rough titanium surfaces",
             "context": "ELISA assay"},
        ],
        "cell_types": [
            {"entity": "macrophage", "evidence": "M2 macrophages were predominant on hydrophilic surfaces",
             "context": "flow cytometry"},
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        ci = ClaimIndex(persist_dir=td)
        count = ci.index_paper_claims("test.pdf", entities)
        assert count == 3

        # Query
        results = ci.query("IL-6 titanium implant obese", n_results=5)
        assert len(results) > 0


# ── CACHE_VERSION ─────────────────────────────────────────────────────────────

def test_cache_version_is_v4():
    """CACHE_VERSION should be v4 for Phase 8."""
    from src.cache import CACHE_VERSION
    assert CACHE_VERSION == "v4", f"Expected v4, got {CACHE_VERSION}"


# ── Graph Factory ─────────────────────────────────────────────────────────────

def test_create_graph_storage_default():
    """Factory creates NetworkXJSONStorage by default."""
    from src.graph import create_graph_storage
    from src.graph.networkx_json_storage import NetworkXJSONStorage
    import tempfile, json
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        json.dump({"directed": True, "multigraph": False, "nodes": [], "edges": []}, f)
        f.flush()
        fname = f.name

    storage = create_graph_storage(file_path=fname, backend="networkx_json")
    assert isinstance(storage, NetworkXJSONStorage)
    os.unlink(fname)
