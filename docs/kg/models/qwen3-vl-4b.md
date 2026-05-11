---
phase: 7
status: tested-not-selected
tags: [models, qwen, vision]
created: 2026-05-10
links: [model-comparison, gemma4-e4b, llava-7b]
---

# qwen3-vl:4b — Tested But Not Selected

## Overview

Dedicated vision-language model evaluated as a potential figure description model during Phase 7.

## Specifications

| Attribute | Value |
|-----------|-------|
| Size | 2.6 GB |
| Parameters | 4.4B |
| Family | qwen3vl |
| Architecture | Vision-language (ViT + LLM) |
| Multimodal | Yes — native vision support |

## Performance

Produced **mid-quality biomedical descriptions**:

| Model | Example Output |
|-------|---------------|
| qwen3-vl:4b | _"A figure showing gene and protein expression data with bar charts"_ |
| gemma4:e4b | _"Bar chart showing IL-6 levels (pg/mL) in CD4-/- vs WT mice at 24h and 48h post-treatment"_ |
| llava:7b | _"A scientific poster with graphs and text"_ |

qwen3-vl:4b identifies that the content is biomedical (genes, proteins, expression) — better than llava's generic output, but lacks the specificity of gemma4 (no cell types, no cytokines, no experimental conditions).

## Why Not Selected

1. **Model rotation required**: like llava:7b, qwen3-vl:4b requires a separate load/unload cycle (~15–30s overhead per query)
2. **Lower accuracy than gemma4:e4b**: identifies general biological categories but misses specific biomedical entities (IL-6, CD4, CD8, WT/knockout, p-values)
3. **Redundant when gemma4 already loaded**: the primary reason to consider a dedicated vision model was that the text model didn't support vision — but gemma4:e4b does

## When It Would Be a Good Choice

qwen3-vl:4b would be the **best dedicated vision model** if:
- gemma4:e4b were not already loaded as the text model
- A small, fast vision model were needed for a separate vision pipeline (e.g., real-time figure analysis in a different application)
- Resource constraints required the smallest viable vision model (2.6 GB vs gemma4's 9.6 GB)

## Comparison

| Metric | qwen3-vl:4b | gemma4:e4b | llava:7b |
|--------|-------------|------------|----------|
| Biomedical specificity | Medium | High | Low |
| Size | 2.6 GB | 9.6 GB | 4.7 GB |
| Latency | ~14s | ~17s | ~12s |
| Rotation needed | Yes | No | Yes |
| Biomedical terms | 2–5/figure | 5–12/figure | 0–1/figure |

See [[model-comparison]] and [[gemma4-e4b]] for full evaluation.
