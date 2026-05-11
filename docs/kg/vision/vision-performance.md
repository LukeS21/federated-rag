---
phase: 7
status: complete
tags: [vision, benchmarks, performance]
created: 2026-05-10
links: [vision-benchmarks, gemma4-e4b, model-comparison, phase-7-vision]
---

# Vision Pipeline Performance

Performance benchmarks for the full Phase 7 vision pipeline.

## Per-Stage Timing

| Stage | Time | Type | Details |
|-------|------|------|---------|
| **Extraction** | ~58 s | Docling CPU | 47 figures from 4 PDFs; classification is the dominant cost |
| **Filtering** | <0.01 s | Programmatic | Weighted scoring on pre-existing classification data |
| **Description** | ~17 s / figure | LLM (gemma4:e4b) | Includes image encoding + API round-trip |
| **Embedding** | <1 s | ChromaDB batch | 38 figure descriptions to a single batch insert |

## Per-PDF Cost

For a typical biomedical PDF with ~10 figures:

| Pipeline stage | Duration |
|----------------|----------|
| Extraction + filtering (no LLM) | ~75 s |
| Full pipeline with descriptions | ~75 s + (n × 17 s) |
| ~10 figures, full pipeline | ~245 s (~4 min) |

## Phase 7 Quality Scorecard

Programmatic quality metrics applied to all figure descriptions:

| Metric | Target | Result |
|--------|--------|--------|
| Non-empty response | 100% | 100% (38/38) |
| ASCII-only output | 100% | 100% (post-scrub) |
| Minimum length | >100 chars | 100% |
| Biomedical keyword presence | >0 | 100% (mean: 3.2) |
| Caption alignment check | pass | 100% |

### Notes

- **LLM-as-Judge** metrics are planned but not yet executed for vision. The text pipeline has judge-evaluated metrics; vision will follow the same methodology.
- gemma4:e4b is pre-loaded (used for text tasks), so model rotation cost is **zero** for the vision pipeline.
- Batch description mode in `describe_figures()` avoids repeated model load/unload cycles.
