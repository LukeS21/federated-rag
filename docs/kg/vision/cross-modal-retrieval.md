---
phase: 7
status: complete
tags: [vision, retrieval, chromadb, embedding]
created: 2026-05-10
links: [hybrid-retriever, figure-filtering, vision-descriptor-api, survey-mode-graph]
---

# Cross-Modal Retrieval

Architecture for interleaving figure descriptions with text chunks in a single retrieval pipeline.

## Core Class: `FigureEmbedder`

Location: `src/vision/figure_embedder.py`

Embeds figure descriptions into the **same ChromaDB collection** used for text chunks.

### Metadata Schema

Each embedded figure chunk carries:

```json
{
  "chunk_type": "figure",
  "source": "avery_2024.pdf",
  "page_no": 3,
  "bbox": [120, 240, 580, 720],
  "caption": "Fig. 1. Characterization of...",
  "width": 460,
  "height": 480
}
```

## BM25 Exclusion

BM25 index is **NOT** updated with figure descriptions. Rationale: figure descriptions are AI-generated text (not author-authored), and adding AI-generated text to the BM25 sparse index would pollute keyword search scores. Only the dense (ChromaDB) index includes figures.

## HybridRetriever Monkey-Patch

`HybridRetriever.query()` is extended with an `include_figures` parameter:

### When `include_figures=True`

1. Fetches broader results (`k * 2` or similar expansion)
2. Separates results into `text_chunks` and `figure_chunks`
3. Interleaves them: text chunks first, figures interspersed at up to 1/3 ratio
4. Returns a unified list preserving relevance order

### When `include_figures=False`

Actively filters out any chunks where `chunk_type == "figure"` from results.

## Integration Points

| Call site | `include_figures` |
|-----------|-------------------|
| `survey_retrieve_node` | `True` |
| `sectioned_retrieve_node` | `True` |
| `_run_debate_for_theme` | `True` |
| Other retrieval paths | `False` (default) |

## Import Order Caveat

The monkey-patch requires `import src.vision.figure_embedder` to execute **before** `HybridRetriever` is imported anywhere in the application. This ensures the `query()` override is in place before any call site tries to call it. Verify this order in `app.py` entry point.
