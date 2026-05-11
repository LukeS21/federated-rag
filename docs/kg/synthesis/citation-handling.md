---
phase: 7
status: fixed
tags: [synthesis, citations, bugs]
created: 2026-05-10
links: [drafter-architecture, claim-ledger, drafter-citation-fix]
---

# Citation Handling

Evolution of citation handling in the synthesis pipeline.

## The Bug

The original Drafter prompt included a hardcoded example: `@author2025`. The LLM hallucinated this exact key across all generated claims, ignoring the actual citation keys from Zotero.

## Phase 7 Fix

**Prompt change:** "Use ONLY the exact citation keys provided — never invent new ones."

**Cache invalidation:** Cache version bumped to `v3` to invalidate all stale cached responses containing hallucinated citations (see [[cache-version-v3]]).

## Citation Flow

1. Zotero generates real citation keys (e.g., `@avery2024`)
2. Citation keys propagate through chunk metadata during retrieval
3. Drafter receives an **Available Citations** list in the user prompt
4. Claims must reference only keys from this list

## Remaining Risk

LLMs can still hallucinate citation keys that *look plausible* but don't exist in the corpus (e.g., `@avery2025` instead of `@avery2024`). The prompt constraint reduces but does not eliminate this behavior.

## Deferred

Post-generation citation validation is not yet implemented. Deferred to **Phase 8**. The [[claim-ledger|ClaimLedger]] tracks which citations are used but does not currently validate them against the available set.
