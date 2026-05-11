---
phase: 7
status: decided
tags: [decisions, retrieval, bm25]
created: 2026-05-10
links: [hybrid-retriever, cross-modal-retrieval]
---

# BM25: No Figure Descriptions

## Decision

Figure descriptions go to ChromaDB only, **not** to the BM25 keyword index.

## Rationale

- AI-generated text (figure descriptions) does not belong in the author-authored keyword index
- BM25 keyword matching on hallucinated or AI-generated terms could produce misleading retrieval results
- Mixing AI-generated text with author text in BM25 creates a semantic mismatch in retrieval modes

## Correct Retrieval Mode

ChromaDB semantic similarity is the right retrieval mode for AI-generated figure descriptions. The vector embedding captures the *meaning* of the description rather than exact keyword matches.

## Impact

Figure descriptions remain searchable through ChromaDB while preserving the purity of the BM25 keyword index for author-authored text only.
