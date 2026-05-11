---
phase: 7
status: complete
tags: [synthesis, claim-ledger, dedup, citations]
created: 2026-05-10
links: [sectioned-survey-nodes, content-addressed-claims, citation-handling]
---

# Claim Ledger

The `ClaimLedger` class provides content-addressed claim management with cross-section deduplication.

## Content Addressing

Uses SHA-256 hashing of normalized claim text. Each claim is identified by a 16-character hex digest. Normalization strips whitespace, lowercases, and removes punctuation for comparison.

## Operations

### add_claim

Adds a claim to the ledger. Auto-parses `@citations` from the claim text to track which citations are used.

### find_duplicates

Returns existing claims that match a new claim by content hash.

### is_duplicate

Boolean check whether a claim already exists in the ledger.

### filter_new_claims

Given a list of claims, returns only those not already present. Used by [[sectioned-survey-nodes|sectioned_draft_section]] to skip redundant claims across sections.

### get_used_citations

Returns the set of all citation keys referenced across all ledgered claims.

### coverage_report

Generates a report with per-section breakdown: total claims, grounded claims, duplicate claims, ungrounded claims, and citation coverage.

### validate_section

Validates a section's claims for:
- **Ungrounded claims** — claims with no supporting citations
- **Duplicate claims** — claims already appearing in prior sections
- **Min-citation checks** — each claim must have at least one citation

### get_claims_by_section

Returns claims organized by section for output assembly.

## Persistence

JSON serialization via `save()` and `load()` methods. Ledger state persists across section boundaries in the sectioned survey flow.

## Usage

Used in [[sectioned-survey-nodes|sectioned_draft_section]] for cross-section dedup. Each section's draft passes through the ledger before finalization.

## Testing

14 unit tests cover all operations, dedup logic, and validation.

## Performance

- Dedup: O(n) per claim insertion
- Instant for <1000 claims
- SHA-256 overhead negligible at survey scale
