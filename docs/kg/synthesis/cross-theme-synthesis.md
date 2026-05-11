---
phase: 4
status: reference
tags: [synthesis, cross-theme, survey-mode]
created: 2026-05-10
links: [survey-mode-graph, per-theme-debate, gap-analysis]
---

# Cross-Theme Synthesis

Cross-theme synthesis aggregates per-theme results into a unified narrative.

## Input

Consumes the `per_theme_syntheses` dict — mapping of theme names to their synthesized claims.

## Parallel Execution

Uses `ThreadPoolExecutor` with `max_workers=2` to run in parallel:
- **Cross-theme narrative** — Drafter synthesizes across all themes
- **Gap analysis** — identifies missing evidence (see [[gap-analysis]])

## Dense Claims Integration

Dense claims feed **directly** into the cross-theme Drafter — no intermediate compression step. This is a key optimization from Phase 5.5 (see [[dense-claim-format]]).

## Gap Analysis Model

Gap analysis uses a separately configurable model (`GAP_ANALYSIS_MODEL`, defaults to `gemma4:e4b`). This model is optimized for the gap identification task rather than narrative synthesis.

## Evidence Truncation

Uses dynamic tiktoken cap via `_fit_summaries_to_context()` for evidence truncation. Same mechanism as per-theme debate but applied at cross-theme scope.

## Anchoring Expectations

Lower anchoring on cross-theme text is **expected and acceptable**:
- Cross-theme: ~0.56
- Per-theme: 0.88-0.95+

Cross-theme synthesis is inherently more inferential — it draws connections *between* themes rather than grounding claims in specific chunks.
