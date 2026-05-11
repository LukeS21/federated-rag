---
phase: 7
status: complete
tags: [vision, ingest, integration, app]
created: 2026-05-10
links: [vision-descriptor-api, figure-extraction, figure-filtering, cross-modal-retrieval, phase-7-vision]
---

# Vision Ingest Integration

End-to-end coordination of the vision processing pipeline during PDF ingestion.

## Core Function: `vision_ingest_pdf()`

Location: `src/vision/vision_ingest.py`

Orchestrates the full vision pipeline for a single PDF:

```
vision_ingest_pdf(pdf_path)
  ├── FigureExtractor → extract figures
  ├── FigureFilter   → filter non-data figures
  ├── VisionDescriptor.describe_figures() → generate descriptions
  └── FigureEmbedder → embed into ChromaDB
```

## Calling Context

Invoked from `app.py` during PDF ingestion, **after** text chunk extraction and embedding have completed for the same PDF. This order ensures the ChromaDB collection already exists when figure embeddings need to be inserted.

## Skip Conditions

Vision ingest is bypassed when:

1. `VISION_MODEL` is set to `0`, `none`, `false`, or `no` — disables the entire vision subsystem
2. The PDF has already been ingested and its ChromaDB entries still exist — prevents re-extraction (idempotency check)

## No Model Rotation

gemma4:e4b is already loaded for text-generation tasks (survey synthesis, debate, etc.), so the vision pipeline does not need to load/unload any model. This eliminates the primary cost of model rotation (~30s per switch) that was experienced with earlier experiments.

## Return Value

```python
{
    "extracted": 47,        # Total figures extracted
    "kept": 38,             # After filtering
    "described": 38,        # Successfully described
    "embedded": 38,         # Inserted into ChromaDB
    "skipped_figures": 0    # Any that failed description
}
```

## Description Persistence

After describing, results are saved to `projects/default/figure_descriptions.json` for offline review, debugging, and reprocessing without re-running extraction. This is handled by `VisionDescriptor.save_descriptions()`.
