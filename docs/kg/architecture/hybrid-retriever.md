---
phase: [2, 3]
status: reference
tags: [architecture, retrieval, chromadb, bm25]
created: 2026-05-10
links: [system-overview, evidence-anchoring, cross-modal-retrieval]
---

# Hybrid Retriever

Fused dense + sparse retrieval architecture.

## Dual Index Architecture

| Index | Type | Backend | Embedding Model |
|-------|------|---------|-----------------|
| Dense | ChromaDB | Vector DB | sentence-transformer all-MiniLM-L6-v2 |
| Sparse | BM25 | Tantivy | Token frequency |

## Fusion Strategy

Reciprocal Rank Fusion (RRF, `k=60`). Combines ranked results from both indexes into a single relevance-ordered list. RRF empirically outperforms linear combination for heterogeneous ranking signals.

## Query Parameters

| Parameter | Description |
|-----------|-------------|
| `n_results` | Number of chunks to return |
| `similarity_threshold` | L2 distance cutoff for dense results |
| `max_chunks` | Hard cap on total returned chunks |
| `filter_references` | Boolean to strip citation markers |

## Figure Retrieval (Phase 7a)

`include_figures=True` flag returns figure descriptions alongside text chunks. Figure descriptions are AI-generated (gemma4:e4b vision model), not author-authored.

## Monkey-Patch Import Order Caveat

Must import `figure_embedder` BEFORE `HybridRetriever`. The figure embedder monkey-patches the retriever's document processing pipeline. Incorrect import order silently drops figure descriptions.

## BM25 Limitation

BM25 does NOT index figure descriptions. Dense retrieval (ChromaDB) handles figure-text matching. BM25 exclusively covers author-authored paper text.
