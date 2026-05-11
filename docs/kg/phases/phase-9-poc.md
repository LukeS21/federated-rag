---
phase: 9
status: poc-built
tags: [phase-9, pubmed, semantic-scholar, literature-discovery]
created: 2026-05-10
links: [phase-8-initiation, literature-discovery, keyword-limitation]
---

# Phase 9 — Literature Discovery POC

## What's Built

| Component | Path | Status |
|-----------|------|--------|
| PubMed wrapper | `src/retrieval/pubmed.py` | Complete |
| Semantic Scholar wrapper | `src/retrieval/semantic_scholar.py` | Complete |
| Demo script | `phase9_pubmed_demo.py` | Complete |

## POC Results

- **87% novelty rate**: 13 of 15 returned papers were novel (not in existing 6-paper corpus)
- **5 queries** tested across immunology, oncology, and neurology topics
- **Semantic Scholar** used for title/abstract matching — PubMed for metadata enrichment
- All novel papers successfully indexed and retrievable

## What's NOT Built (Requires Phase 8)

| Feature | Dependency | Reason |
|---------|------------|--------|
| Coverage Check LLM node | Phase 8 clustering | Needs theme mapping to detect gaps |
| Auto-download pipeline | Phase 8 Neo4j | Needs corpus management layer |
| Institutional proxy access | Phase 8 | Lower priority, config-only |
| Auto-re-synthesize | Phase 8 caching | Needs cache invalidation on corpus change |

## Architecture Plan

```
Query → Decompose → Coverage Check → Local Retrieve + External Fetch
                                          ↓
                          Merge → Cluster → Extract → Synthesize
```

1. **Decompose**: break query into biomedical concepts (existing)
2. **Coverage Check** (not built): LLM assesses whether local corpus covers each concept
3. **External Fetch**: PubMed + Semantic Scholar for uncovered concepts
4. **Merge**: combine local + external results
5. **Cluster → Extract → Synthesize**: existing Phase 7 pipeline

## API Strategy

| API | Role | Rate Limit | Auth |
|-----|------|------------|------|
| PubMed Entrez | Primary search, metadata | 3 req/s (10 with key) | API key optional |
| Semantic Scholar | Title/abstract search, citation graph | 100 req/5min | Free tier |
| Unpaywall | OA PDF location | 100k/day | Email required |
| PMC OA Service | Direct PDF download | 3 req/s | None (OA only) |

## Full-Text Acquisition Chain

```
PubMed Search → PMIDs
  → PMC OA Service (free OA full-text)
  → Unpaywall (finds OA versions)
  → Zotero translation-server (paywalled PDFs via institutional access)
  → EZProxy (institutional proxy fallback)
```

Priority order minimizes reliance on institutional access. PMC OA covers ~40% of PubMed.

## Keyword Limitation

- **Current**: static keyword extraction from query text — simple and fast but misses synonyms and related concepts
- **Planned**: LLM-based Coverage Check node that:
  1. Expands query concepts with biomedical synonyms (MeSH-aware)
  2. Maps concepts to existing thematic clusters
  3. Identifies gaps with confidence scores
  4. Only triggers external fetch for concepts below coverage threshold
- This is the primary blocker between POC and production Phase 9.

See [[phase-8-initiation]] for the Phase 8 work required to unblock full Phase 9 implementation.
