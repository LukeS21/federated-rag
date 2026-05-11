---
phase: 7
status: decided
tags: [decisions, vision, gemma4]
created: 2026-05-10
links: [model-comparison, gemma4-e4b, no-model-rotation]
---

# Vision Model: gemma4:e4b

## Decision

Use `gemma4:e4b` as the vision model for figure description extraction.

## Rationale

- **Accuracy:** Beat 2 alternative vision models on biomedical figure accuracy benchmarks
- **No rotation:** Already loaded in the Ollama instance, no model rotation required
- **Specificity:** Identifies specific cytokines, cell groups, and pathways from figure labels and legends

## Tradeoff

- Memory: 9.6 GB vs dedicated vision models at 2-5 GB
- Rotation overhead makes dedicated vision models slower *overall* despite smaller memory footprint
- Loading/unloading models adds ~15-30s per query (see [[no-model-rotation]])

## Alternatives Considered

Two alternative vision models were benchmarked and rejected due to lower biomedical accuracy.
