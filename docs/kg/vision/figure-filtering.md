---
phase: 7
status: complete
tags: [vision, filtering, classification, docling]
created: 2026-05-10
links: [classification-first-filter, figure-extraction, phase-7-vision]
---

# Figure Filtering

Smart relevance-based filtering of extracted figures to discard non-data imagery.

## Core Class: `FigureFilter`

Location: `src/vision/figure_filter.py`

Constructor accepts a configurable threshold:

```python
filter = FigureFilter(threshold=0.35)  # default
filter = FigureFilter(threshold=0.5)   # stricter
```

## Scoring Formula

Weighted composite score — all weights configurable via constructor:

| Component | Weight | Source |
|-----------|--------|--------|
| **Classification** | 65% | Docling classifier top prediction |
| **Caption text** | 20% | Keyword presence in resolved caption |
| **Figure size** | 10% | Area (px²) relative to typical figure |
| **Page position** | 5% | Distance from page center |

### Classification Weighting

Uses Docling's `DocumentFigureClassifier-v2.5` **top prediction only** (not full distribution):

- **Positive classes** (weight +1.0): `bar_chart`, `line_chart`, `scatter_plot`, `photograph`, `heatmap`, `micrograph`, `box_plot`, `pie_chart`, `histogram`, `flowchart`
- **Negative classes** (weight -0.5): `logo`, `icon`, `stamp`, `qr_code`, `signature`, `page_thumbnail`

Confidence from the classifier is multiplied by the class weight.

### Caption Scoring

Caption text is scored for the presence of figure-relevant keywords (e.g., "Fig.", "Figure", "Table", "shows", "comparison").

### Size & Position

Small figures near page margins (typical of logos, icons) are penalized. Large, centered figures are rewarded.

## Filtering Results

| Metric | Value |
|--------|-------|
| Figures extracted | 47 |
| Figures kept | 38 (80.9%) |
| Figures discarded | 9 |
| - Logos | 3 |
| - Page thumbnails | 3 |
| - Icons | 3 |
| Data loss | 0 (zero false negatives) |
| Avg relevance score (post caption fix) | 0.951 |
| Avg relevance score (pre caption fix) | 0.823 |

## Integration

Called by [[vision-ingest]] immediately after [[figure-extraction]]. Filtering is programmatic (<0.01s) and does not invoke any LLM.
