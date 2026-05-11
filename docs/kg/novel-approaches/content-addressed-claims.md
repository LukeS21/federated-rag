---
phase: 7
status: invented
created: 2026-05-10
tags:
  - novel-approaches
  - claims
  - dedup
  - sha256
links:
  - "[[claim-ledger]]"
  - "[[sectioned-survey-nodes]]"
---
SHA-256 content addressing for claim deduplication. Normalize claim text, hash, use first 16 hex chars as stable ID. Cross-section dedup without full-text search. Short enough for log output, collision-resistant for 100K+ claims. Simpler than embedding-based similarity (would miss near-duplicates). Persisted to JSON for cross-session continuity.
