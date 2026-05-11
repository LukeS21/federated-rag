---
phase: 6
status: decided
tags: [decisions, anchoring, retrieval]
created: 2026-05-10
links: [evidence-anchoring, hybrid-retriever]
---

# Hybrid Anchoring

## Decision

Extend `compute_anchoring_score()` with BM25 + ChromaDB fusion for evidence retrieval.

## The Problem

BM25 alone has a keyword-frequency bias: common words like "mice" drown out discriminative terms like "leptin." Claims about specific mechanisms get anchored to irrelevant chunks simply because both mention common biomedical terms.

## Solution

Dual-retriever fusion:
- **BM25:** Primary retriever, anchors 56% of claims
- **ChromaDB:** Handles 3.4% of claims that BM25 misses entirely (semantic matches without keyword overlap)

## Fusion Strategy

BM25 is attempted first. If BM25 fails to find evidence for a claim, ChromaDB semantic search is used as a fallback.
