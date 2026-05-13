"""
Tests for FigureEmbedder — figure-to-text embedding and cross-modal retrieval.
"""
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vision.figure_embedder import FigureEmbedder


@pytest.fixture
def retriever():
    """In-memory HybridRetriever for testing."""
    chroma = ChromaClient("test_figure_embed")
    bm25 = BM25Index()
    return HybridRetriever(chroma, bm25)


@pytest.fixture
def sample_figures():
    """Synthetic figure dicts with descriptions."""
    return [
        {
            "description": "Bar chart showing IL-6 levels elevated in obese mice.",
            "caption": "Figure 1: IL-6 levels.",
            "page_no": 3,
            "file_path": "/tmp/fig1.png",
            "bbox": {"l": 100.0, "t": 700.0, "r": 500.0, "b": 300.0},
            "width": 400,
            "height": 300,
            "figure_index": 0,
            "image": Image.new("RGB", (100, 100)),
        },
        {
            "description": "Flow cytometry dot plot showing macrophage populations.",
            "caption": "Figure 2: Flow cytometry.",
            "page_no": 4,
            "file_path": "/tmp/fig2.png",
            "bbox": {"l": 100.0, "t": 700.0, "r": 500.0, "b": 300.0},
            "width": 600,
            "height": 400,
            "figure_index": 1,
            "image": Image.new("RGB", (100, 100)),
        },
        {
            "description": "",
            "caption": "",
            "page_no": 1,
            "file_path": "/tmp/fig3.png",
            "bbox": {},
            "width": 48,
            "height": 49,
            "figure_index": 2,
            "image": None,
        },
    ]


def test_embed_figures(retriever, sample_figures):
    """Embedding adds figure descriptions to ChromaDB."""
    embedder = FigureEmbedder(retriever)
    count = embedder.embed(sample_figures, pdf_source="test_paper.pdf")

    # Figure 2 has empty description and caption — should be skipped
    assert count == 2

    # Verify ChomaDB can find them
    results = retriever.chroma.query("IL-6 levels", n_results=5)
    texts = results.get("documents", [[]])[0]
    assert any("IL-6" in t for t in texts), f"No IL-6 match in {texts}"

    # Check metadata
    metadatas = results.get("metadatas", [[]])[0]
    figure_metas = [m for m in metadatas if m.get("chunk_type") == "figure"]
    assert len(figure_metas) > 0, "No figure metadata found"
    for m in figure_metas:
        assert m["source"] == "test_paper.pdf"
        assert m["chunk_type"] == "figure"


def test_embed_empty_list(retriever):
    """Empty figure list returns 0."""
    embedder = FigureEmbedder(retriever)
    count = embedder.embed([], pdf_source="test.pdf")
    assert count == 0


def test_embed_all_empty_descriptions(retriever):
    """Figures with no text are skipped."""
    embedder = FigureEmbedder(retriever)
    figures = [
        {"description": "", "caption": "", "file_path": "/tmp/f1.png",
         "page_no": 1, "bbox": {}, "width": 10, "height": 10, "figure_index": 0}
    ]
    count = embedder.embed(figures, pdf_source="test.pdf")
    assert count == 0


def test_include_figures_flag(retriever):
    """The include_figures parameter is accepted by the extended query."""
    # Ingest some text chunks first
    chunks = [
        {"text": "IL-6 increases in obese mice", "metadata": {"source": "a.pdf", "chunk_type": "text"}},
        {"text": "Titanium implants show osseointegration", "metadata": {"source": "a.pdf", "chunk_type": "text"}},
    ]
    retriever.ingest(chunks)

    # Query without figures — should work
    results = retriever.query("IL-6 obesity")
    assert len(results) > 0

    # Query with include_figures — should also work
    results_with = retriever.query("IL-6 obesity", include_figures=True)
    assert len(results_with) > 0


def test_cross_modal_retrieval(retriever, sample_figures):
    """A figure description is retrievable alongside text chunks."""
    embedder = FigureEmbedder(retriever)
    embedder.embed(sample_figures, pdf_source="a.pdf")

    # Also add text chunks
    chunks = [
        {"text": "Titanium implants show osseointegration in mouse models", "metadata": {"source": "a.pdf", "chunk_type": "text"}},
    ]
    retriever.ingest(chunks)

    # Query for something the figure describes
    results = retriever.query("macrophage populations flow cytometry", include_figures=True)
    assert len(results) > 0

    # At least one result should be from a figure
    figure_results = [r for r in results if r.get("metadata", {}).get("chunk_type") == "figure"]
    assert len(figure_results) > 0, f"No figure results found in {[r['metadata'] for r in results]}"


def test_query_without_figures_excludes_figures(retriever, sample_figures):
    """When include_figures=False, only text is returned."""
    embedder = FigureEmbedder(retriever)
    embedder.embed(sample_figures, pdf_source="a.pdf")

    chunks = [
        {"text": "Titanium implants show osseointegration", "metadata": {"source": "a.pdf", "chunk_type": "text"}},
    ]
    retriever.ingest(chunks)

    results = retriever.query("IL-6 macrophage flow cytometry", include_figures=False)
    for r in results:
        assert r.get("metadata", {}).get("chunk_type") != "figure"


def test_figure_metadata_is_valid(retriever, sample_figures):
    """Embedded figure metadata has all expected fields."""
    import json

    embedder = FigureEmbedder(retriever)
    embedder.embed(sample_figures, pdf_source="test.pdf")

    results = retriever.chroma.query("macrophage", n_results=5)
    metadatas = results.get("metadatas", [[]])[0]
    figure_metas = [m for m in metadatas if m.get("chunk_type") == "figure"]

    for meta in figure_metas:
        for key in ("source", "chunk_type", "page_no", "figure_index", "caption", "width", "height"):
            assert key in meta, f"Missing key: {key}"
        assert isinstance(meta["page_no"], int)
        assert isinstance(meta["width"], int)
        assert isinstance(meta["height"], int)
