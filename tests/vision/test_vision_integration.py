"""
Integration test: Vision pipeline end-to-end (extract → filter → embed → retrieve).

Tests the full Phase 7a pipeline with real PDFs and validates that figure
descriptions surface in cross-modal queries.
"""
from pathlib import Path
import pytest
from PIL import Image

from src.vision.figure_extractor import FigureExtractor
from src.vision.figure_filter import FigureFilter
from src.vision.figure_embedder import FigureEmbedder
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever

REAL_PDF = Path(
    "data/Avery et al. - 2024 - CD4+ and CD8+ T cells reduce inflammation and promote bone healing in response to titanium implants.pdf"
)


@pytest.mark.skipif(not REAL_PDF.exists(), reason="No Avery 2024 PDF in data/")
def test_captions_are_proper_figure_labels():
    """Captions should read like 'Fig. 1. Characterization of...' not classification data."""
    extractor = FigureExtractor()
    figures = extractor.extract(REAL_PDF)

    data_figs = [f for f in figures
                 if f["classification"]
                 and f["classification"][0]["class_name"] in ("bar_chart", "table", "box_plot", "scatter_plot")]

    assert len(data_figs) > 0, "Expected data figures in a biomedical PDF"

    for fig in data_figs:
        cap = fig.get("caption", "")
        if cap.strip():
            assert "kind=" not in cap, f"Caption contaminated with classification metadata"
            assert "DocumentPictureClassifier" not in cap


@pytest.mark.skipif(not REAL_PDF.exists(), reason="No Avery 2024 PDF in data/")
def test_filter_removes_logos_keeps_data():
    """Filter discards logos/icons but keeps bar charts with real captions."""
    extractor = FigureExtractor()
    ff = FigureFilter(threshold=0.35)

    figures = extractor.extract(REAL_PDF)
    scored = ff.score_all(figures)

    # Logos should be discarded
    for fig in scored:
        cls = fig["relevance_components"]["classification_top"]
        if cls in ("logo", "icon", "page_thumbnail"):
            assert fig["relevance_score"] < 0.35, f"{cls} should be discarded, got {fig['relevance_score']}"

    # Data figures should be kept
    kept = ff.filter(figures)
    for fig in kept:
        cls = fig["relevance_components"]["classification_top"]
        assert cls not in ("logo", "icon", "page_thumbnail"), f"{cls} should not be in kept"


@pytest.mark.skipif(not REAL_PDF.exists(), reason="No Avery 2024 PDF in data/")
def test_cross_modal_retrieval_with_real_figures():
    """Extract real figures, embed them, and verify they appear in cross-modal search."""
    extractor = FigureExtractor()
    ff = FigureFilter()
    figures = extractor.extract(REAL_PDF)
    kept = ff.filter(figures)

    # Use in-memory ChromaDB
    chroma = ChromaClient("test_vision_integration")
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma, bm25)

    # Embed figures with their captions as descriptions (no vision model needed for test)
    for fig in kept:
        fig["description"] = fig.get("caption", "")

    embedder = FigureEmbedder(retriever)
    embedder.embed(kept, pdf_source=REAL_PDF.name)

    # Also add a text chunk
    chunks = [
        {"text": "Titanium surface modifications affect macrophage polarization in vivo.",
         "metadata": {"source": REAL_PDF.name, "chunk_type": "text"}},
    ]
    retriever.ingest(chunks)

    # Cross-modal: query for figure content
    results = retriever.query("CD4 T cells bone implant contact", include_figures=True)
    assert len(results) > 0

    # At least one figure result should appear
    figure_results = [r for r in results
                      if r.get("metadata", {}).get("chunk_type") == "figure"]
    assert len(figure_results) > 0, (
        f"No figure results in cross-modal query. "
        f"Results metadata: {[r.get('metadata', {}) for r in results]}"
    )

    # Figure results should reference the source PDF
    for fr in figure_results:
        meta = fr.get("metadata", {})
        assert meta.get("source") == REAL_PDF.name
        assert isinstance(meta.get("page_no"), int)
        assert len(meta.get("caption", "")) > 0 or len(fr.get("text", "")) > 0


@pytest.mark.skipif(not REAL_PDF.exists(), reason="No Avery 2024 PDF in data/")
def test_figure_embedding_preserves_metadata():
    """Embedded figure metadata has correct source, type, page_no, and caption."""
    extractor = FigureExtractor()
    ff = FigureFilter()
    figures = extractor.extract(REAL_PDF)
    kept = ff.filter(figures)

    for fig in kept:
        fig["description"] = fig.get("caption", "")

    chroma = ChromaClient("test_vision_meta")
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma, bm25)

    embedder = FigureEmbedder(retriever)
    embedder.embed(kept, pdf_source=REAL_PDF.name)

    results = retriever.chroma.query("characterization of smooth rough", n_results=10)
    metadatas = results.get("metadatas", [[]])[0]
    figure_metas = [m for m in metadatas if m.get("chunk_type") == "figure"]

    assert len(figure_metas) > 0
    for meta in figure_metas:
        assert meta["source"] == REAL_PDF.name
        assert meta["chunk_type"] == "figure"
        assert isinstance(meta["page_no"], int) and meta["page_no"] > 0
        assert meta["figure_index"] >= 0
        # At least one figure should have a caption
        if meta["caption"]:
            assert "Fig." in meta["caption"] or "Figure" in meta["caption"]
