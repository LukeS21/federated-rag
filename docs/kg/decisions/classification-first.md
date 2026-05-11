---
phase: 7
status: decided
tags: [decisions, vision, filtering]
created: 2026-05-10
links: [figure-filtering, classification-first-filter]
---

# Classification-First Figure Filtering

## Decision

Give 65% weight to the trained classifier (`DocumentFigureClassifier-v2.5`) in figure filtering, with size, caption, and page as soft hints only.

## Rationale

- Trained model is more reliable than heuristic rules
- Heuristics (size, page position, caption presence) are noisy signals for biomedical figures
- Classifier has been trained on biomedical document figure classification specifically

## Tradeoff

A relevant but tiny image on page 1 could be penalized by the combined scoring. In practice, this has not occurred on our corpus of biomedical papers.

## Configurability

All weights are configurable, allowing per-domain tuning without code changes. Different document types or domains may benefit from different weight distributions.
