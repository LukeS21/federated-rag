---
phase: 4
status: reference
tags: [architecture, clustering, survey-mode]
created: 2026-05-10
links: [survey-mode-graph, per-theme-debate]
---

# Thematic Clustering

Embedding-based thematic clustering for Survey Mode.

## Primary Method: Embedding-Based Clustering

- Model: sentence-transformer all-MiniLM-L6-v2
- Cosine similarity threshold: 0.35
- Embeddings pre-computed at PDF ingest, stored in `projects/default/embeddings/`
- Every paper assigned to 1+ themes

## Fallback Method: LLM-Based Clustering

Preserved for edge cases where embedding similarity fails (highly domain-specific terminology, rare biomedical entities). LLM reads paper abstracts and groups thematically.

## Query Decomposition

Broad research question decomposed into 3-8 themed sub-queries by the reasoning model. Each sub-query targets a distinct thematic angle of the broader question.

## Quality-Driven Theme Count

No hardcoded "3-8 themes" limit. System identifies ALL semantically distinct themes in the corpus. Small corpora may produce fewer themes; large diverse corpora produce more.

## Theme Assignment

Papers assigned to themes based on embedding cosine similarity against theme centroid vectors. Multi-assignment supported — a single paper may contribute to multiple thematic syntheses.
