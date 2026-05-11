---
phase: 6.5
status: decided
tags: [decisions, parallelization, performance]
created: 2026-05-10
links: [survey-mode-graph, per-theme-debate]
---

# Per-Theme Parallelization

## Decision

`PER_THEME_MAX_WORKERS = 2` with same-model `ThreadPoolExecutor`.

## Rationale

A single model has a single KV cache. Two parallel requests to the same Ollama instance benefit from HTTP request pipelining without doubling KV cache memory usage.

## Performance

~23% faster per-theme wall clock (161s vs 210s sequential).

## Constraint

**Do NOT exceed 2 workers** without memory testing on M3 Max 36 GB. The KV cache is the bottleneck, not CPU cores.

## History

Dual-model parallelism was attempted in Phase 5.5 and abandoned — it exhausted KV cache memory (see [[same-model-parallel]]). Same-model parallelism avoids this issue entirely.
