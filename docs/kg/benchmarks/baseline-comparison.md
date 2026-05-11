---
phase: 7
status: complete
tags: [benchmarks, baseline, comparison]
created: 2026-05-10
links: [baseline-single-run, performance-history]
---

# Baseline Comparison

Naive RAG vs full pipeline comparison (May 2026).

| Metric | Naive RAG | Full Pipeline |
|---|---|---|
| Claims | 5 | 134 |
| Grounded | 5 | 133 |
| Ungrounded | 0 | 1 |
| Anchoring | 1.000 | 0.993 |
| Citations | 3 | 6 |
| Output | 959 chars | 28,467 chars |

Naive RAG's perfect score is misleading — only 5 safe claims. Pipeline 27x more coverage with equal grounding.

Cached at `projects/default/baseline_comparison.json`.

**Caveat:** single run, needs multi-query variance.
