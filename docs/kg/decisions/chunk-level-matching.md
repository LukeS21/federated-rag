---
phase: 6
status: decided
tags: [decisions, anchoring, correctness]
created: 2026-05-10
links: [evidence-anchoring]
---

# Chunk-Level Matching

## Decision

All correctness metrics use **chunk-level matching**, not sentence-level matching.

## The Problem with Sentence-Level

Splitting evidence into sentences inflates grounded rates by 3-5×. With thousands of granular sentence units, almost any claim can find a surface-level "match" — even if the match is semantically irrelevant.

## Chunk-Level Results

| Method | Grounded Rate |
|--------|--------------|
| Sentence-level | ~99% |
| Chunk-level | 83-88% |

## Rationale

Chunk-level matching is more honest. A claim should be anchored to a coherent evidence unit (a chunk), not a decontextualized sentence fragment. The 83-88% grounded rate reflects genuine evidence coverage rather than statistical artifacts of granular tokenization.
