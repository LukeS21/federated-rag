---
phase: 5.5
status: active
tags: [synthesis, drafter, format, decisions]
created: 2026-05-10
links: [drafter-architecture, debate-chain]
---

# Dense Claim Format

Why dense claims (one per line) replaced prose paragraphs.

## Phase 5.5 Optimization

Dense claim format was introduced as a structural optimization to reduce output size and eliminate redundant processing.

## Size Reduction

| Metric | Prose Format | Dense Claims |
|--------|-------------|--------------|
| Per-theme output | 1000-2200 chars | 250-600 chars |

~60-75% reduction in per-theme output size.

## Downstream Impact

No compression step needed between per-theme and cross-theme synthesis — dense claims feed directly into the cross-theme Drafter (see [[cross-theme-synthesis]]).

## Quality Impact

Anchoring scores maintained at 0.88-0.95 despite the format change. The system prompt constraint ("one claim per line, no preamble") prevents filler text while preserving evidentiary quality.

## System Prompt Enforcement

The Drafter system prompt enforces:
```
Format: one claim per line. No preamble, no transitions, no repetition.
```

## Decision

**DO NOT REVERT** to prose format. Dense claims are strictly superior for this pipeline.
