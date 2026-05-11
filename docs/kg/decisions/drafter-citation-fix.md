---
phase: 7
status: decided
tags: [decisions, drafter, citations, bugs]
created: 2026-05-10
links: [citation-handling, drafter-architecture]
---

# Drafter Citation Fix

## Decision

Remove the hardcoded `@author2025` example from the Drafter prompt. Replace with explicit anti-hallucination constraint.

## Changes

1. **Prompt change:** "Use ONLY exact citation keys provided — never invent new ones."
2. **Cache invalidation:** Bumped cache to v3 to invalidate stale responses (see [[cache-version-v3]])

## Rationale

The original prompt contained `@author2025` as a formatting example. The LLM latched onto this specific key and hallucinated it across all generated claims, ignoring the actual citation keys from Zotero.

## Tradeoff

LLMs can still hallucinate plausible-looking citation keys (e.g., `@avery2025` for a 2024 paper). The prompt constraint reduces frequency but does not eliminate the risk.

## Deferred

Post-generation citation validation deferred to **Phase 8**. No runtime validation currently checks that generated citations exist in the available set.
