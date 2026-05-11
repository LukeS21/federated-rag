---
phase: 3
status: reference
tags: [architecture, anchoring, evidence]
created: 2026-05-10
links: [hybrid-retriever, debate-chain, deep-mode-graph]
---

# Evidence Anchoring

Programmatic claim grounding check against source documents.

## Approach

Programmatic (no LLM): claim decomposition → BM25 evidence search → cosine similarity scoring. Fully deterministic and reproducible.

## Claim Decomposition

Regex-based sentence splitting with biomedical-domain heuristics. Handles semicolons, numbered lists, and parenthetical asides appropriately.

## Hybrid Anchoring

`compute_anchoring_score()` uses BM25 + ChromaDB fusion — mirroring the main pipeline's HybridRetriever pattern. Hybrid approach fixes BM25 keyword-frequency bias by incorporating semantic matching.

## Thresholds

| Score | Action |
|-------|--------|
| >= 0.85 | Finalize — output accepted |
| < 0.85 | Conditional second pass (Arbiter revises) |
| < 0.85 (after pass 2) | Human escalation (HumanGate) |

## Similarity Metric

TF-IDF cosine similarity. Threshold originally 0.35, evolved to hybrid scoring with BM25 + ChromaDB fusion.

## Chunk-Level Matching

Matching is at chunk level, not sentence level. Sentence splitting inflates match rates 3-5× by fragmenting context — chunks preserve surrounding evidence.

## Caveat

At 6 papers, anchoring primarily measures traceability (keyword matching) rather than factual accuracy. Scores should be interpreted as "claim is traceable to corpus" not "claim is scientifically correct."
