---
phase: [4, 7]
status: reference
tags: [synthesis, gap-analysis, survey-mode]
created: 2026-05-10
links: [cross-theme-synthesis, survey-mode-graph]
---

# Gap Analysis

Structured identification of evidence gaps in survey synthesis.

## Parallel Execution

Runs in parallel with cross-theme synthesis via `ThreadPoolExecutor` with `max_workers=2`. Gap analysis consumes per-theme syntheses directly — no dependency on cross-theme output.

## Model Configuration

`GAP_ANALYSIS_MODEL` is separately configurable. Defaults to `gemma4:e4b`, which cut gap analysis latency from ~368s (qwen3.6:35b) to ~40s.

## Gap Novelty Validation

Gap novelty is validated via **Discussion-overlap testing**. On a 6-paper corpus, 80% of identified gaps were novel (not addressed in any paper's discussion section). This confirms the gap model identifies genuine evidence gaps rather than restating acknowledged limitations.

## Output Structure

Gap analysis produces structured output with:
- **Gap description** — what evidence is missing
- **Novelty score** — is this gap already discussed in the literature?
- **Actionability score** — can this gap inform future research?

Both scores come from LLM-as-Judge evaluation.
