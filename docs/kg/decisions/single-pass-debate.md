---
phase: 5.5
status: decided
tags: [decisions, debate, optimization]
created: 2026-05-10
links: [debate-chain, conditional-critic]
---

# Single-Pass Debate

## Decision

Simplify the debate chain from 2 Critic→Arbiter passes to a single pass.

## Before

2 Critic→Arbiter passes per theme, totaling 5 LLM calls per debated theme:
1. Critic pass 1
2. Arbiter pass 1
3. Critic pass 2
4. Arbiter pass 2
5. Final synthesis

## After

Single pass, 3 LLM calls per debated theme:
1. Critic
2. Arbiter
3. Final synthesis

## Supporting Changes

- [[conditional-critic|Conditional critic]] threshold raised to 0.50 — fewer themes enter debate at all
- Regression guard prevents quality loss from reduced review depth

## Tradeoff

Less rigorous review, but **67% fewer Critic calls** with zero quality loss at 6 papers. The regression guard ensures the original draft is preserved if debate would worsen it.
