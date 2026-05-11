"""
Tests for smart figure filtering (classification, size, caption, page heuristics).
"""
from unittest.mock import patch
import pytest

from src.vision.figure_filter import (
    FigureFilter,
    get_classification_score,
    compute_size_score,
    compute_caption_score,
    compute_page_score,
    DATA_FIGURE_CLASSES,
    EXTRANEOUS_CLASSES,
    ALL_CLASS_WEIGHTS,
)

# ── Classification scoring unit tests ──────────────────────────────────────

def test_classification_score_bar_chart():
    """Bar chart with high confidence → score ~1.0."""
    preds = [
        type("p", (), {"class_name": "bar_chart", "confidence": 0.995})(),
    ]
    top, score = get_classification_score(preds)
    assert top == "bar_chart"
    assert score > 0.9, f"Expected > 0.9, got {score}"


def test_classification_score_logo():
    """Logo → negative score contribution → low overall score."""
    preds = [
        type("p", (), {"class_name": "logo", "confidence": 0.95})(),
    ]
    top, score = get_classification_score(preds)
    assert top == "logo"
    assert score < 0.3, f"Logo should score low, got {score}"


def test_classification_score_mixed():
    """Top class data-figure with low-confidence logo second."""
    preds = [
        type("p", (), {"class_name": "bar_chart", "confidence": 0.8})(),
        type("p", (), {"class_name": "logo", "confidence": 0.15})(),
    ]
    top, score = get_classification_score(preds)
    assert top == "bar_chart"
    assert 0.5 < score < 1.0, f"Expected 0.5-1.0, got {score}"


def test_classification_score_empty():
    """Empty predictions → default 0.3."""
    top, score = get_classification_score([])
    assert top == "unknown"
    assert score == 0.3


def test_all_data_classes_have_positive_weight():
    """Every data-figure class has positive weight."""
    for class_name in DATA_FIGURE_CLASSES:
        assert ALL_CLASS_WEIGHTS[class_name] > 0, f"{class_name} should have positive weight"


def test_all_extraneous_classes_have_negative_weight():
    """Every extraneous class has negative weight."""
    for class_name in EXTRANEOUS_CLASSES:
        assert ALL_CLASS_WEIGHTS[class_name] < 0, f"{class_name} should have negative weight"


# ── Size scoring unit tests ───────────────────────────────────────────────

def test_size_score_tiny():
    """10×10 image → score 0."""
    assert compute_size_score(10, 10) == 0.0


def test_size_score_small():
    """100×100 image → score 0.3."""
    assert compute_size_score(100, 100) == 0.3


def test_size_score_medium():
    """180×180 image → score 0.6."""
    assert compute_size_score(180, 180) == 0.6


def test_size_score_large():
    """500×500 image → score 1.0."""
    assert compute_size_score(500, 500) == 1.0


# ── Caption scoring unit tests ────────────────────────────────────────────

def test_caption_score_figure_label():
    """Caption starting with 'Figure' → score 1.0."""
    assert compute_caption_score("Figure 1: IL-6 levels...") == 1.0


def test_caption_score_fig_label():
    """Caption starting with 'Fig.' → score 1.0."""
    assert compute_caption_score("Fig. 2. Macrophage polarization.") == 1.0


def test_caption_score_contains_figure():
    """Caption containing 'figure' → score 0.7."""
    assert compute_caption_score("Results shown in figure 3") == 0.7


def test_caption_score_empty():
    """Empty caption → score 0.0."""
    assert compute_caption_score("") == 0.0
    assert compute_caption_score(None) == 0.0


def test_caption_score_generic():
    """Long text without 'figure' → score 0.3."""
    assert compute_caption_score("Immunohistochemistry staining of tissue") == 0.3


# ── Page scoring unit tests ────────────────────────────────────────────────

def test_page_score_body_page():
    """Page 5+ → score 1.0."""
    assert compute_page_score(5) == 1.0
    assert compute_page_score(10) == 1.0


def test_page_score_first_page():
    """Page 1 → soft signal, not a hard penalty."""
    assert compute_page_score(1) == 0.6  # soft hint, not 0.0


def test_page_score_second_page():
    """Page 2 → moderate signal."""
    assert compute_page_score(2) == 0.8


def test_page_score_zero():
    """Unknown page → neutral."""
    assert compute_page_score(0) == 0.5


# ── FigureFilter integration tests ─────────────────────────────────────────

def make_figure(class_name="bar_chart", confidence=0.99, width=400, height=300,
                page_no=5, caption=""):
    """Create a synthetic figure dict for testing."""
    return {
        "file_path": "/tmp/fig_001.png",
        "page_no": page_no,
        "caption": caption,
        "width": width,
        "height": height,
        "classification": [
            {"class_name": class_name, "confidence": confidence},
            {"class_name": "logo", "confidence": 0.005},
        ],
        "image": None,
        "pdf_source": "test.pdf",
        "figure_index": 0,
        "bbox": None,
    }


def test_filter_keeps_data_figure():
    """A bar chart on page 5 is kept."""
    ff = FigureFilter()
    figures = [make_figure("bar_chart", 0.99, 400, 300, 5, "Figure 1: Results.")]
    kept = ff.filter(figures)
    assert len(kept) == 1
    assert kept[0]["relevance_score"] > 0.5


def test_filter_discards_logo():
    """A 48×48 logo should be discarded."""
    ff = FigureFilter()
    figures = [make_figure("logo", 0.95, 48, 48, 1, "")]
    kept = ff.filter(figures)
    assert len(kept) == 0, f"Logo should be filtered out, got score {figures[0].get('relevance_score', '?')}"


def test_filter_data_figure_page1_not_penalized():
    """A bar chart on page 1 should still pass (classification dominates)."""
    ff = FigureFilter()
    figures = [make_figure("bar_chart", 0.99, 400, 300, 1, "Figure 1: Key finding.")]
    kept = ff.filter(figures)
    assert len(kept) == 1, "Data figure on page 1 should not be discarded"
    assert kept[0]["relevance_score"] > 0.5


def test_filter_sorts_by_relevance():
    """Results are sorted by relevance_score descending."""
    ff = FigureFilter()
    figures = [
        make_figure("bar_chart", 0.80, 400, 300, 5, "Figure 1: Moderate."),
        make_figure("bar_chart", 0.99, 600, 400, 5, "Figure 2: High conf."),
        make_figure("bar_chart", 0.60, 200, 150, 3, "Figure 3: Low conf."),
    ]
    kept = ff.filter(figures)
    assert len(kept) >= 2
    scores = [f["relevance_score"] for f in kept]
    assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"


def test_score_all_returns_all():
    """score_all returns all figures (no filtering), sorted."""
    ff = FigureFilter()
    figures = [
        make_figure("bar_chart", 0.99, 400, 300, 5, "Fig 1."),
        make_figure("logo", 0.95, 48, 48, 2, ""),
    ]
    scored = ff.score_all(figures)
    assert len(scored) == 2
    for f in scored:
        assert "relevance_score" in f
        assert "relevance_components" in f
        comps = f["relevance_components"]
        for key in ("classification", "classification_top", "size", "caption", "page"):
            assert key in comps, f"Missing component: {key}"


def test_custom_weights():
    """Custom weights are used in scoring."""
    ff = FigureFilter(
        w_classification=1.0,  # classification only
        w_caption=0.0,
        w_size=0.0,
        w_page=0.0,
    )
    fig = make_figure("bar_chart", 0.99, 100, 100, 1, "Figure 1.")
    scored = ff.score_figure(fig)
    cls_score = scored["relevance_components"]["classification"]
    assert abs(scored["relevance_score"] - cls_score) < 0.01


def test_threshold_zero_keeps_all():
    """Threshold=0 keeps everything."""
    ff = FigureFilter(threshold=0.0)
    figures = [
        make_figure("bar_chart", 0.99, 400, 300, 5, "Fig"),
        make_figure("logo", 0.95, 48, 48, 1, ""),
    ]
    kept = ff.filter(figures)
    assert len(kept) == 2


def test_threshold_one_keeps_nothing():
    """Threshold=1.0 discards everything (scores never reach 1.0)."""
    ff = FigureFilter(threshold=1.0)
    figures = [make_figure("bar_chart", 0.999, 600, 400, 5, "Figure 1.")]
    kept = ff.filter(figures)
    assert len(kept) == 0


def test_photograph_class_is_kept():
    """A photograph (e.g., microscopy) is a data figure."""
    ff = FigureFilter()
    figures = [make_figure("photograph", 0.85, 800, 600, 4, "Figure 1: H&E staining.")]
    kept = ff.filter(figures)
    assert len(kept) == 1


def test_invalid_threshold_raises():
    """Threshold outside [0,1] raises ValueError."""
    with pytest.raises(ValueError):
        FigureFilter(threshold=1.5)
    with pytest.raises(ValueError):
        FigureFilter(threshold=-0.1)
