---
phase: 7
status: replaced
tags: [models, llava, vision, deprecated]
created: 2026-05-10
links: [model-comparison, gemma4-e4b, qwen3-vl-4b]
---

# llava:7b — Why It Was Replaced

## History

- Originally the **default vision model** — first model tested for Phase 7 figure description
- 2023 release, widely used for general-purpose vision-language tasks
- Installed on the system and still available for reference comparison

## Specifications

| Attribute | Value |
|-----------|-------|
| Size | 4.7 GB |
| Parameters | 7B |
| Family | llama |
| Architecture | Dense transformer |
| Multimodal | Yes |

## Why Replaced

llava:7b produced **generic, non-biomedical descriptions**. For a figure showing IL-6 cytokine levels:

| Model | Description |
|-------|-------------|
| llava:7b | _"A scientific poster with graphs and text"_ |
| gemma4:e4b | _"Bar chart showing IL-6 levels (pg/mL) in CD4-/- vs WT mice at 24h and 48h post-treatment. CD4-/- group shows significantly higher IL-6 (p<0.01) at both timepoints"_ |

The difference is critical for biomedical RAG — llava descriptions provide no searchable biomedical terms.

## Additional Disadvantage

llava:7b required **model rotation**: unload the active text model → load llava → describe figures → unload llava → reload text model. This added ~15–30s of overhead per query. gemma4:e4b eliminated this entirely.

## Comparison

| Metric | llava:7b | gemma4:e4b | qwen3-vl:4b |
|--------|----------|------------|-------------|
| Biomedical terms identified | 0–1/figure | 5–12/figure | 2–5/figure |
| Specificity | Generic labels | Cell types, cytokines, assays | "Gene/protein expression" |
| Latency | ~12s | ~17s | ~14s |
| Rotation overhead | +15–30s | 0s | +15–30s |
| Status | **Replaced** | **Active** | Tested, not selected |

See [[model-comparison]] for the full evaluation with example figure outputs and methodology.
