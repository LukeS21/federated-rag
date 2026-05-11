"""
Tests for Phase 7a figure extraction from PDFs via Docling.
"""
from pathlib import Path
import pytest
from PIL import Image

from src.vision.figure_extractor import FigureExtractor

TEST_PDF = Path("data/test.pdf")


@pytest.mark.skipif(not TEST_PDF.exists(), reason="No test PDF in data/")
def test_extract_all_figures():
    """Extraction returns a list of figure dicts with expected keys."""
    extractor = FigureExtractor()
    figures = extractor.extract(TEST_PDF)

    assert isinstance(figures, list)
    assert len(figures) > 0, "Expected at least one extracted figure"

    for fig in figures:
        for key in (
            "file_path", "page_no", "bbox", "caption", "width", "height",
            "image", "pdf_source", "figure_index", "classification",
        ):
            assert key in fig, f"Figure missing key: {key}"

        # Image is a PIL Image
        assert isinstance(fig["image"], Image.Image)

        # File was saved to disk
        assert Path(fig["file_path"]).exists(), f"File not saved: {fig['file_path']}"

        # Dimensions are positive
        assert fig["width"] > 0
        assert fig["height"] > 0
        assert fig["page_no"] > 0
        assert fig["pdf_source"] == TEST_PDF.name

        # Classification is present
        assert isinstance(fig["classification"], list)


@pytest.mark.skipif(not TEST_PDF.exists(), reason="No test PDF in data/")
def test_classification_data():
    """Classification data is populated for each figure."""
    extractor = FigureExtractor()
    figures = extractor.extract(TEST_PDF)

    for fig in figures:
        cls_list = fig["classification"]
        assert isinstance(cls_list, list), f"classification should be a list, got {type(cls_list)}"
        if cls_list:
            top = cls_list[0]
            assert "class_name" in top, f"Missing class_name in {top}"
            assert "confidence" in top, f"Missing confidence in {top}"
            assert 0.0 <= top["confidence"] <= 1.0, f"Confidence out of range: {top['confidence']}"


@pytest.mark.skipif(not TEST_PDF.exists(), reason="No test PDF in data/")
def test_figure_files_saved():
    """Extracted figures are saved to the output directory."""
    import tempfile
    import shutil

    tmpdir = Path(tempfile.mkdtemp())
    try:
        extractor = FigureExtractor(output_dir=tmpdir)
        figures = extractor.extract(TEST_PDF)

        for fig in figures:
            path = Path(fig["file_path"])
            assert path.exists()
            assert path.suffix == ".png"
            # Verify it's a valid PNG
            img = Image.open(path)
            assert img.size == (fig["width"], fig["height"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not TEST_PDF.exists(), reason="No test PDF in data/")
def test_encode_base64():
    """Base64 encoding produces valid data URIs."""
    extractor = FigureExtractor()
    figures = extractor.extract(TEST_PDF)

    for fig in figures[:2]:
        b64 = extractor.encode_base64(fig["image"])
        assert b64.startswith("data:image/jpeg;base64,")
        assert len(b64) > 100, "Base64 string too short"


@pytest.mark.skipif(not TEST_PDF.exists(), reason="No test PDF in data/")
def test_compute_figure_hash():
    """Figure hashing produces consistent 16-char hex digests."""
    extractor = FigureExtractor()
    figures = extractor.extract(TEST_PDF)

    for fig in figures[:3]:
        hash1 = extractor.compute_figure_hash(fig["image"])
        hash2 = extractor.compute_figure_hash(fig["image"])
        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 16
        assert all(c in "0123456789abcdef" for c in hash1)


@pytest.mark.skipif(
    not Path("data/Avery et al. - 2024 - CD4+ and CD8+ T cells reduce inflammation and promote bone healing in response to titanium implants.pdf").exists(),
    reason="No Avery 2024 PDF in data/"
)
def test_real_captions_extracted():
    """Captions should be real Figure labels, not classification data."""
    pdf = Path("data/Avery et al. - 2024 - CD4+ and CD8+ T cells reduce inflammation and promote bone healing in response to titanium implants.pdf")
    extractor = FigureExtractor()
    figures = extractor.extract(pdf)

    figures_with_caption = [f for f in figures if f["caption"].strip()]
    assert len(figures_with_caption) > 0, "Expected some figures to have captions"

    for fig in figures_with_caption:
        cap = fig["caption"]
        # Real captions should NOT contain classification metadata
        assert "kind=" not in cap, f"Caption contains classification metadata: {cap[:80]}"
        assert "provenance=" not in cap, f"Caption contains provenance: {cap[:80]}"
        assert "DocumentPictureClassifier" not in cap, f"Caption contains classifier name: {cap[:80]}"

        # Captions should start with "Fig" for data figures
        cls = fig["classification"][0]["class_name"] if fig["classification"] else ""
        if cls in ("bar_chart", "line_chart", "scatter_plot", "box_plot", "photograph", "table"):
            assert cap.startswith("Fig") or cap.startswith("Table"), \
                f"Data figure caption should start with Fig/Table: {cap[:60]}..."


@pytest.mark.skipif(
    not Path("data/Avery et al. - 2022 - Canonical Wnt signaling enhances pro-inflammatory response to titanium by macrophages.pdf").exists(),
    reason="No Avery 2022 PDF in data/"
)
def test_extract_from_real_biomedical_pdf():
    """Extraction works on a real biomedical PDF."""
    pdf = Path("data/Avery et al. - 2022 - Canonical Wnt signaling enhances pro-inflammatory response to titanium by macrophages.pdf")
    extractor = FigureExtractor()
    figures = extractor.extract(pdf)

    assert len(figures) > 0
    for fig in figures:
        assert fig["pdf_source"] == pdf.name
        assert isinstance(fig["image"], Image.Image)
