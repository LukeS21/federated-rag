---
phase: 7
status: reference
tags: [synthesis, sectioned-survey, nodes]
created: 2026-05-10
links: [sectioned-survey-graph, claim-ledger, drafter-architecture]
---

# Sectioned Survey Nodes

Walkthrough of all 7 nodes in the sectioned survey graph.

## sectioned_init

Parses the user's query for a section plan (e.g., "include methods and results"). Falls back to default IMRaD structure (Introduction, Methods, Results, and Discussion) if no explicit plan is detected.

## sectioned_retrieve

Retrieval node with `include_figures=True`. Builds `section_context` — a mapping of sections to their relevant evidence chunks and figure descriptions.

## sectioned_draft_section

Core drafting node. Integrates:
1. [[claim-ledger|ClaimLedger]] — `filter_new_claims()` to skip duplicates
2. `check_claim_grounded()` — validates each claim has supporting citations
3. Saves ledger state after drafting
4. Invokes Drafter per section (see [[drafter-architecture]])

## sectioned_review

Returns status for interrupt. Allows human-in-the-loop review of each section before proceeding to the next.

## sectioned_route

Routes to the next section or to assembly. Tracks which sections remain and advances the section cursor.

## sectioned_assemble

Combines all drafted sections into final output. Runs:
- Ledger validation across all sections
- Coverage report generation
- Warning emission for ungrounded/duplicate claims

## sectioned_scrub

ASCII scrub of final assembled output. Removes non-ASCII characters and normalizes formatting for downstream consumption.
