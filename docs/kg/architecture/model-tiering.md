---
phase: 4
status: reference
tags: [architecture, models, performance]
created: 2026-05-10
links: [gemma4-e4b, qwen3-6-35b, llm-provider]
---

# Model Tiering

Two-tier model architecture for latency/capability tradeoffs.

## Tier Configuration

| Tier | Model | VRAM | Use Cases |
|------|-------|------|-----------|
| Fast | gemma4:e4b | 9.6 GB | Per-theme drafting, query decomposition, extraction, summarization, gap analysis, figure description |
| Reasoning | qwen3.6:35b | 23 GB | Cross-theme synthesis, critique, arbitration |

## Model Rotation

Fast tier unloads (`OLLAMA_KEEP_ALIVE=60s`) before reasoning tier loads. Peak memory ~28 GB. Fits within 36 GB M3 Max unified memory.

## Configurable Gap Analysis

`GAP_ANALYSIS_MODEL` separately configurable, defaults to gemma4:e4b. Cut gap analysis latency from 368s → ~40s by moving from reasoning tier to fast tier.

## resolve_model() Function

| Input | Tier |
|-------|------|
| `"small"`, `"chat"` | Fast tier |
| `"large"`, `"pro"` | Reasoning tier |
| `"alt"` | Alternate tier (configurable) |

## Agent Model Parameter

All agents accept configurable `model` parameter — no hardcoded model assignments. Enables per-use-case tuning without code changes.
