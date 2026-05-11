---
phase: [5.5]
status: active
tags: [models, qwen, reasoning-tier]
created: 2026-05-10
links: [model-tiering, gemma4-e4b, debate-chain]
---

# qwen3.6:35b — Model Card

## Role

**Reasoning tier model.** Handles high-level synthesis tasks requiring deep biomedical reasoning:

- Cross-theme synthesis (integrating claims across IMRaD sections)
- Critique (evaluating draft quality, coverage, and consistency)
- Arbitration (resolving conflicts between competing claims)

This model does **not** handle per-chunk operations — those are delegated to the fast tier ([[gemma4-e4b]]).

## Specifications

| Attribute | Value |
|-----------|-------|
| Size | 23 GB |
| Parameters | 36B (~3B active) |
| Family | qwen35moe |
| Architecture | MoE (Mixture of Experts) |
| Context window | 262K tokens |
| Agentic score | 81.2% TAU2 |
| Multimodal | **No** — text-only variant |

## Limitations

- **No vision support**: figure descriptions must be handled by [[gemma4-e4b]]. qwen3.6:35b can reference figure descriptions in its synthesis but cannot process images directly.
- **Slower cold start**: model loading takes ~10–15s due to 23 GB size
- **Higher VRAM requirement**: requires ~23 GB VRAM solo, incompatible with simultaneous gemma4:e4b load on current hardware

## Performance

| Task | Latency | Notes |
|------|---------|-------|
| Cross-theme synthesis | ~150–350s | Full IMRaD synthesis across 6+ papers |
| Critique | ~30–50s | Per-section quality evaluation |
| Arbitration | ~30–50s | Claim conflict resolution |
| Cold start load | ~10–15s | One-time penalty per session |

## Memory Management

- **Loads only when needed** — i.e., for synthesis, critique, or arbitration tasks
- **Preceded by gemma4:e4b unload**: `OLLAMA_KEEP_ALIVE=60s` triggers gemma4 unload before qwen3.6 loads
- **Peak combined VRAM**: ~28 GB with gemma4 (gemma4 unloads before qwen3.6 loads in practice, but the peak occurs during the handoff window)
- After synthesis completes, qwen3.6 unloads and gemma4 reloads for subsequent queries

## Two-Tier Architecture

```
Query Per-Chunk (Fast)         Query Synthesis (Reasoning)
┌─────────────────────┐       ┌─────────────────────────┐
│    gemma4:e4b       │  →    │    qwen3.6:35b          │
│    • Draft           │       │    • Synthesize          │
│    • Extract         │       │    • Critique            │
│    • Summarize       │       │    • Arbitrate           │
│    • Figure Descr.   │       │                         │
│    9.6 GB            │       │    23 GB                │
└─────────────────────┘       └─────────────────────────┘
```

See [[model-tiering]] for the full strategy and performance rationale.
