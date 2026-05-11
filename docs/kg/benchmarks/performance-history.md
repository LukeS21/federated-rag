---
phase: [5, 6, 7]
status: reference
tags: [benchmarks, performance, history]
created: 2026-05-10
links: [api-vs-local, baseline-comparison, vision-benchmarks]
---

# Performance History

Performance evolution from Phase 5 to Phase 7.

| Metric | Phase 5 | Phase 6 | Phase 7 |
|---|---|---|---|
| Survey query latency | 12-39 min | 5-8 min | 8.7 min |
| Per-theme wall clock | — | 210 s | 161 s (parallel) |
| Cross-theme synthesis | — | ~200 s | 150-350 s (variable) |
| Gap analysis | — | 368 s | ~40 s (gemma4:e4b) |
| LLM calls per query | ~8-10 (across phases) | — | — |
| Tests | 97 → 102 → 117 → 201 | — | — |

## Figure Benchmarks

- Extraction: ~58 s
- Description: ~17 s/fig
- Filtering: <0.01 s

## Sectioned Survey

~50 s for 4 IMRaD sections.
