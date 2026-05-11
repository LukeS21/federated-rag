---
phase: 7
status: complete
tags: [vision, docling, extraction, figures]
created: 2026-05-10
links: [figure-filtering, vision-descriptor-api, caption-resolution-bug, phase-7-vision]
---

# Figure Extraction

Extracts figures from biomedical PDFs using Docling.

## Core Class: `FigureExtractor`

Location: `src/vision/figure_extractor.py`

Initialized with a `DoclingDocument` and configured with:
- `generate_picture_images=True` — renders figure regions as PIL images
- `do_picture_classification=True` — classifies each figure via Docling's classifier

## Extraction Output

Each extracted figure is a dict with fields:

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | str | Path to saved PNG |
| `page_no` | int | Page number (1-indexed) |
| `bbox` | list[float] | [x0, y0, x1, y1] bounding box |
| `caption` | str | Resolved figure caption text |
| `width` | int | Pixel width |
| `height` | int | Pixel height |
| `image` | PIL.Image | PIL Image object |
| `pdf_source` | str | Source PDF filename |
| `figure_index` | int | Sequential index across all figures |
| `classification` | dict | Classifier predictions (see below) |

Figures saved as PNGs to `projects/default/figures/{pdf_name}/`.

## Caption Resolution

Captions are resolved via Docling's reference system:

```python
for picture in doc.pictures:
    if picture.captions:
        ref = picture.captions[0]           # RefItem
        idx = ref.cref.lstrip("#/texts/")   # e.g. "42"
        caption = doc.texts[int(idx)].text   # resolved label
```

> **CRITICAL**: Picture captions come from `picture.captions` (RefItem objects pointing into `doc.texts`), **NOT** from `picture.annotations` — annotations contain classification metadata, not figure labels. See [[caption-resolution-bug]].

## Classification Data

Docling's `DocumentFigureClassifier-v2.5` provides ~25 class predictions per image:

| Top Classes | Confidence |
|-------------|------------|
| bar_chart | 0.82 |
| scatter_plot | 0.11 |
| photograph | 0.04 |
| logo | 0.01 |
| ... | ... |

Used by [[figure-filtering]] for relevance scoring.

## Base64 Encoding

Helper for vision model API calls:

```python
def encode_image_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()
```

## Benchmarks

- **47 figures** extracted from **4 PDFs** in **~60 seconds**
- Includes classification overhead (which is the bulk of extraction time)
