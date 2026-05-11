---
phase: 8
status: planned
tags: [phase-8, scale, neo4j, caching, benchmarks]
created: 2026-05-10
links: [knowledge-graph, thematic-clustering, hybrid-retriever, scale-caveat, phase-9-poc]
---

# Phase 8 — Initiation Plan

## Goal

Scale from 6 papers to **100s–1000s** with **sub-minute query times** for biomedical literature synthesis.

*Note: This is where definitive benchmarking belongs — all current scores at 6 papers are preliminary. See [[scale-caveat]].*

## Sub-Tasks

| # | Task | Estimate | Priority | Depends On |
|---|------|----------|----------|------------|
| 1 | Acquire 100+ paper corpus | 1–2 hrs | First | Phase 9 demo script |
| 2 | Neo4j adapter | 4–6 hrs | High | Task 1 |
| 3 | Hierarchical clustering | 3–5 hrs | High | Task 2 |
| 4 | Per-theme top-K retrieval | 2–3 hrs | Medium | Task 3 |
| 5 | Corpus-level claim index / L0 cache | 3–4 hrs | Medium | Task 3 |
| 6 | Multi-tier caching L0–L4 | 2–3 hrs | Medium | Tasks 2–5 |
| 7 | Full-scale benchmarks | 2–4 hrs | Final | All above |

## Recommended Execution Order

```
Acquire Corpus → Neo4j Adapter → Hierarchical Clustering
                                      ├── Per-Theme Top-K Retrieval
                                      ├── Corpus Claim Index (L0)
                                      └── Multi-Tier Caching (L0–L4)
                                              └── Full-Scale Benchmarks
```

### Task Details

**1. Acquire 100+ paper corpus**
- Use `phase9_pubmed_demo.py` to fetch papers programmatically
- Target: 100–500 papers across 3–5 biomedical subdomains (immunology, oncology, neurology, etc.)
- Store raw PDFs for batch ingest

**2. Neo4j adapter**
- Replace JSON-file-based knowledge graph with Neo4j graph database
- Maintain backward-compatible API (same `KnowledgeGraph` interface)
- Cypher query support for fast subgraph traversal
- Estimated latency improvement: 10–50× for graph operations

**3. Hierarchical clustering**
- Thematic clustering of papers (see [[thematic-clustering]])
- Two-level hierarchy: domain → sub-theme
- Enables targeted retrieval (only search relevant clusters per query)
- Clustering algorithm TBD (HDBSCAN or agglomerative with cosine distance)

**4. Per-theme top-K retrieval**
- Once clusters exist, retrieval scoped to relevant themes instead of the full corpus
- Dramatically reduces vector search space: ~1000 → ~50–100 papers per query
- Integrates with existing [[hybrid-retriever]]

**5. Corpus-level claim index / L0 cache**
- Pre-computed claim ledger across entire corpus
- SHA-256 deduplication at corpus scale
- Instant retrieval for previously synthesized claims
- Avoids re-synthesizing the same claim from the same papers

**6. Multi-tier caching L0–L4**

| Tier | Scope | TTL | Contents |
|------|-------|-----|----------|
| L0 | Claim | Persistent | Pre-computed claims |
| L1 | Query | 24h | Full synthesis results |
| L2 | Section | 1h | Per-section retrieval sets |
| L3 | Chunk | 30m | Individual chunk embeddings |
| L4 | Figure | 24h | Figure descriptions |

**7. Full-scale benchmarks**
- Measure end-to-end query latency at 100, 500, 1000 papers
- Compare Neo4j vs JSON, clustered vs flat retrieval
- Memory profiling under load
- Target: **30–90s per query on 1000 papers**

## Target Metrics

| Scale | Current (6 papers) | Target (1000 papers) |
|-------|-------------------|---------------------|
| Query latency | ~60–120s | **30–90s** |
| Retrieval scope | 6 papers | 50–100 (per-theme) |
| Graph traversal | <1s (JSON) | <100ms (Neo4j) |
| Cache hit rate | ~10% (L2 only) | 60–80% (L0–L4) |

See [[phase-9-poc]] for the downstream literature-discovery features that depend on Phase 8 completion.
