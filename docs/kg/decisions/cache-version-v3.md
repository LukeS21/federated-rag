---
phase: 7
status: decided
tags: [decisions, cache, versioning]
created: 2026-05-10
links: [drafter-citation-fix]
---

# Cache Version: v3

## Version Progression

| Version | Change |
|---------|--------|
| v1 | Original cache keys |
| v2 | Removed `@author2025` from prompts |
| v3 | Stronger anti-hallucination constraint added |

## Bump Policy

Bump the cache version whenever prompts or logic change in a way that would invalidate cached LLM responses. This prevents stale responses from being served.

## Key Format

All cache keys are prepended with the version string (e.g., `v3:theme:draft:hash`).

## Maintenance

Clear cache directories after bumping the version. Old cache entries under previous versions are no longer valid and can be pruned.
