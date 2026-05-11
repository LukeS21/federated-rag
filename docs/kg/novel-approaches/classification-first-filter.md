---
phase: 7
status: invented
created: 2026-05-10
tags:
  - novel-approaches
  - vision
  - filtering
  - classification
links:
  - "[[figure-filtering]]"
  - "[[classification-first]]"
---
Using a trained image classifier (`DocumentFigureClassifier-v2.5`) as the primary gate for figure relevance, not heuristics. Size, position, caption are weighted at 35% combined. Classifier runs locally, produces 0.99+ confidence on bar charts. Novel for multi-document biomedical pipelines where filtering journal branding from data figures is critical.
