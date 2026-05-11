---
phase: 5.5
status: decided
tags: [decisions, parallelization, models]
created: 2026-05-10
links: [per-theme-parallel, gemma4-e4b]
---

# Same-Model Parallelism

## Decision

Use same-model parallelism instead of dual-model parallelism.

## Why Dual-Model Failed

Dual-model parallelism (`gemma4:e4b` + `medgemma:4b`) was attempted in Phase 5.5:
- Two models loaded simultaneously exhausted KV cache memory
- M3 Max 36 GB could not hold both model KV caches
- Performance degraded due to memory pressure and swapping

## Why Same-Model Works

Same-model parallelism pipelines HTTP requests to a **single** Ollama instance:
- Single KV cache = predictable memory use
- Overlaps CPU preparation time with HTTP wait time
- No memory penalty for additional workers

## Architecture

`ThreadPoolExecutor` with `max_workers=2` submits requests to the same Ollama model endpoint. HTTP pipelining handles concurrent requests efficiently without requiring multiple model instances.
