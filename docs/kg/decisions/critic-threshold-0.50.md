---
phase: 5.5
status: decided
tags: [decisions, critic, threshold]
created: 2026-05-10
links: [conditional-critic, single-pass-debate]
---

# Critic Threshold: 0.50

## Decision

Set `CONDITIONAL_CRITIC_THRESHOLD = 0.50` for local models (Ollama gemma4:e4b).

## Calibration History

- **DeepSeek (API):** Threshold was 0.35 — DeepSeek produces more consistently grounded claims
- **gemma4:e4b (Ollama):** Threshold raised to 0.50 — local models have higher variance in grounding quality

## Tuning Policy

Tune with **benchmarks, not guesswork.** The threshold should be calibrated against:
1. Anchoring score distribution of the current model
2. Acceptable false-positive rate (skipping Critic on poorly-grounded drafts)
3. Cost/latency budget for debate calls

## Decision

**Do NOT revert** below 0.50 without new benchmark data showing that a lower threshold is safe for the current model.
