"""
Smart figure filtering for biomedical PDFs.

Uses a multi-signal approach to determine whether an extracted image is a
scientific data figure (graphs, charts, microscopy, diagrams) versus
extraneous content (journal logos, icons, stamps, QR codes, decorative
elements).

Signals (combined into a 0–1 relevance score):
  1. **Docling classification** — the built-in DocumentFigureClassifier assigns
     a top-class and confidence.  Data-figure classes (bar_chart, line_chart,
     photograph, scatter_plot, box_plot, flow_chart, pie_chart, table,
     chemistry_structure) add positive weight.  Extraneous classes (logo, icon,
     stamp, signature, qr_code, calendar, page_thumbnail, geographical_map,
     engineering_drawing, topographical_map, crossword_puzzle, bar_code, music)
     subtract weight.
  2. **Size** — images < 80×80 px are almost certainly logos/icons.
     Images > 200×200 px are likely data figures.
  3. **Page position** — figures on page 1 are likely journal masthead logos
     unless large and classified as data.
  4. **Caption presence** — figures with "Figure"/"Fig" captions are data
     figures.

Configurable threshold (default 0.35) controls which figures are kept.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple


# ── Classification weights ──────────────────────────────────────────────────

# High-confidence data-figure classes → positive score contribution
DATA_FIGURE_CLASSES: Dict[str, float] = {
    "bar_chart": 1.0,
    "line_chart": 1.0,
    "scatter_plot": 1.0,
    "box_plot": 1.0,
    "pie_chart": 0.9,
    "flow_chart": 0.8,
    "photograph": 0.85,
    "chemistry_structure": 0.75,
    "table": 0.7,
}

# Context-dependent → moderate score
CONTEXT_CLASSES: Dict[str, float] = {
    "engineering_drawing": 0.4,
    "screenshot_from_computer": 0.5,
    "screenshot_from_manual": 0.4,
    "geographical_map": 0.3,
    "full_page_image": 0.5,
    "other": 0.3,
}

# Extraneous/decorative → negative score contribution
EXTRANEOUS_CLASSES: Set[str] = {
    "logo",
    "icon",
    "stamp",
    "signature",
    "qr_code",
    "calendar",
    "page_thumbnail",
    "topographical_map",
    "crossword_puzzle",
    "bar_code",
    "music",
}

# Maps annotation-level class names (from PictureClassificationPrediction)
# to our canonical weights dict
ALL_CLASS_WEIGHTS: Dict[str, float] = {
    **DATA_FIGURE_CLASSES,
    **CONTEXT_CLASSES,
    **{k: -0.5 for k in EXTRANEOUS_CLASSES},
}


def get_classification_score(classification_predictions: List) -> Tuple[str, float]:
    """Extract top class and compute a classification-based relevance score.

    Args:
        classification_predictions: List of ``PictureClassificationPrediction``
            objects, each with ``class_name`` and ``confidence``.

    Returns:
        Tuple of (top_class_name, classification_score) where score is in [0, 1].
    """
    if not classification_predictions:
        return "unknown", 0.3

    # Helper: classification items can be dicts or objects
    def _get(pred, key, default=None):
        if isinstance(pred, dict):
            return pred.get(key, default)
        return getattr(pred, key, default)

    # Top prediction
    top = classification_predictions[0]
    top_class = _get(top, "class_name", "unknown")

    # Compute weighted score from all predictions
    score = 0.0
    for pred in classification_predictions:
        class_name = _get(pred, "class_name", "")
        conf = _get(pred, "confidence", 0.0)
        weight = ALL_CLASS_WEIGHTS.get(class_name, 0.0)
        score += conf * weight

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return top_class, score


def compute_size_score(width: int, height: int) -> float:
    """Compute a size-based relevance score.

    Small images (< 80×80) are almost certainly logos/icons.  Large images
    (> 200×200) are likely data figures.
    """
    area = width * height
    if area < 6400:     # < 80×80
        return 0.0
    if area < 22500:    # < 150×150
        return 0.3
    if area < 40000:    # < 200×200
        return 0.6
    return 1.0


def compute_caption_score(caption_text: str) -> float:
    """Score based on caption presence and content.

    "Figure X" / "Fig. X" captions strongly indicate a data figure.
    """
    if not caption_text or not caption_text.strip():
        return 0.0
    text = caption_text.lower().strip()
    if text.startswith("figure") or text.startswith("fig.") or text.startswith("fig "):
        return 1.0
    if "figure" in text or "fig." in text:
        return 0.7
    if len(text) > 20:
        return 0.3
    return 0.1


def compute_page_score(page_no: int) -> float:
    """Page-position signal.

    Body pages (3+) get a small positive signal since data figures
    typically appear there.  But we do NOT hard-penalize early pages —
    a relevant figure on page 1 is still relevant.  The classification
    signal dominates, making this a soft hint only.
    """
    if page_no <= 0:
        return 0.5
    if page_no <= 1:
        return 0.6
    if page_no <= 2:
        return 0.8
    return 1.0


class FigureFilter:
    """Filter extracted figures to keep only scientifically relevant ones.

    Each figure is assigned a ``relevance_score`` (0–1) from combined signals:
    classification (weight 0.65, dominant), caption (0.20), size (0.10),
    page position (0.05).  Figures below *threshold* are discarded.

    The Docling ``DocumentFigureClassifier`` is the primary gate — it's a
    trained model, not a heuristic.  Size, caption, and page are soft hints
    that nudge borderline cases.  A bar_chart on page 1 passes; a tiny logo
    anywhere does not.

    Weights are configurable via constructor parameters so they can be tuned
    per domain without code changes.

    Usage::

        ff = FigureFilter(threshold=0.35)
        relevant = ff.filter(figures)
        for fig in relevant:
            print(fig["caption"], fig["relevance_score"])
    """

    def __init__(
        self,
        threshold: float = 0.35,
        w_classification: float = 0.65,
        w_caption: float = 0.20,
        w_size: float = 0.10,
        w_page: float = 0.05,
    ):
        total = w_classification + w_caption + w_size + w_page
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = float(threshold)
        self.w_classification = w_classification / total
        self.w_caption = w_caption / total
        self.w_size = w_size / total
        self.w_page = w_page / total

    def score_figure(self, figure: Dict) -> Dict:
        """Score a single figure dict in-place, adding ``relevance_score`` and
        ``relevance_components`` keys.

        Returns the figure dict (mutated).
        """
        w = figure.get("width", 0)
        h = figure.get("height", 0)
        page = figure.get("page_no", 0)
        caption = figure.get("caption", "")
        classification = figure.get("classification", [])

        cls_top, cls_score = get_classification_score(classification)
        size_score = compute_size_score(w, h)
        caption_score = compute_caption_score(caption)
        page_score = compute_page_score(page)

        # Weighted combination
        relevance = (
            self.w_classification * cls_score
            + self.w_size * size_score
            + self.w_caption * caption_score
            + self.w_page * page_score
        )

        figure["relevance_score"] = round(relevance, 4)
        figure["relevance_components"] = {
            "classification": round(cls_score, 4),
            "classification_top": cls_top,
            "size": round(size_score, 4),
            "caption": round(caption_score, 4),
            "page": round(page_score, 4),
        }
        return figure

    def filter(self, figures: List[Dict]) -> List[Dict]:
        """Score and filter a list of figure dicts.

        Returns only figures with relevance_score >= threshold, sorted by
        score descending.
        """
        scored = [self.score_figure(f) for f in figures]
        kept = [f for f in scored if f["relevance_score"] >= self.threshold]
        kept.sort(key=lambda f: f["relevance_score"], reverse=True)
        return kept

    def score_all(self, figures: List[Dict]) -> List[Dict]:
        """Score all figures and return all of them (no filtering).

        Sorted by relevance_score descending.  Useful for diagnostics.
        """
        scored = [self.score_figure(f) for f in figures]
        scored.sort(key=lambda f: f["relevance_score"], reverse=True)
        return scored
