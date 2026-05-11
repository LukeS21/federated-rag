---
phase: [3, 5.5]
status: reference
tags: [architecture, debate, synthesis]
created: 2026-05-10
links: [deep-mode-graph, evidence-anchoring, conditional-critic, drafter-architecture]
---

# Debate Chain

Three-role heterogeneous multi-agent debate for synthesis quality.

## Agent Roles

| Role | Model | Responsibility |
|------|-------|----------------|
| Drafter | Qwen3.6 35B (now gemma4:e4b for per-theme) | Writes initial synthesis with evidence citations |
| Socratic Critic | Gemma 4 26B | Identifies unsupported claims, asks evidence-grounded questions |
| Arbiter | Qwen3.6 35B | Revises draft addressing critiques |

## Critic Behavior

- Identifies claims lacking evidence grounding
- Asks pointed questions about specific unsupported statements
- NEVER proposes alternative text (Socratic role only)
- Returns `"NO_CRITIQUE"` if all claims are fully grounded

## Flow

```
Drafter → Critic →
  NO_CRITIQUE → skip Arbiter, proceed to AnchoringCheck
  has critique → Arbiter → AnchoringCheck
```

## Phase 5.5 Simplifications

- Single-pass debate (was 2 passes)
- Conditional critic threshold: 0.50 (skip critique if grounding promising)
- Regression guard: keep original draft if debate worsens anchoring score

## Heterogeneous Model Strategy

Different model families (Gemma vs Qwen) resist peer-pressure convergence. Distinct architectures and training distributions produce different reasoning biases — disagreement reflects genuine uncertainty rather than cascading errors.
