---
phase: 6
status: invented
created: 2026-05-10
tags:
  - novel-approaches
  - evaluation
  - llm-as-judge
links:
  - "[[test-suite-overview]]"
  - "[[cross-theme-quality]]"
---
Pre-evaluating judge with TRUE/FALSE/GRAY claims before trusting faithfulness scores. If judge can't discriminate fabrication from truth, discard its evaluations. `gemma4:e4b` scored every claim 5/5 (agreeableness bias). DeepSeek chat and v4-pro correctly discriminated. Standard in recent LLM evaluation research.
