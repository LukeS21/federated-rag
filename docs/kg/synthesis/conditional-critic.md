---
phase: [5.5, 7]
status: active
tags: [synthesis, debate, critic, optimization]
created: 2026-05-10
links: [debate-chain, per-theme-debate, critic-threshold]
---

# Conditional Critic

Threshold-based Critic invocation to reduce unnecessary LLM calls.

## Threshold Configuration

`CONDITIONAL_CRITIC_THRESHOLD = 0.50`

Calibrated for local models (Ollama gemma4:e4b). The original threshold of 0.35 was calibrated for DeepSeek via API.

## Behavior by Anchoring Score

| Score Range | Action |
|-------------|--------|
| < 0.50 | Invoke full debate chain (Critic → Arbiter) |
| ≥ 0.50, < 0.85 | Skip Critic, accept as-is |
| ≥ 0.85 | Well-grounded, skip Critic |

## Debate Regression Guard

If the debate chain *worsens* the anchoring score (i.e., the Critic/Arbiter introduces errors), the original draft is kept. The regression guard prevents the debate from degrading quality.

## Performance Impact

At 6 papers: **67% of Critic calls saved** with zero anchoring degradation. Only poorly-grounded drafts incur the full debate cost.

## Decision

**Do NOT revert** below 0.50 without benchmark data. The threshold should be tuned with benchmarks, not guesswork (see [[critic-threshold-0.50]]).
