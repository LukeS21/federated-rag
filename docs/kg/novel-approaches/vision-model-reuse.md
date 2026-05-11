---
phase: 7
status: invented
created: 2026-05-10
tags:
  - novel-approaches
  - vision
  - gemma4
links:
  - "[[no-model-rotation]]"
  - "[[gemma4-e4b]]"
  - "[[model-rotation]]"
---
Pattern: check whether existing models support multimodal before pulling dedicated vision models. `gemma4:e4b` was already loaded for text tasks; reusing it for figure description eliminated the entire model rotation step. Generalizable: many modern LLMs support multimodal input — test existing models first.
