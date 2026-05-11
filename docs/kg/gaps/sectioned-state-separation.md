---
phase: 7
status: open-minor
created: 2026-05-10
tags:
  - gaps
  - sectioned-survey
  - state
links:
  - "[[sectioned-survey-graph]]"
  - "[[agent-state]]"
---
Sectioned survey graph uses separate state fields (`section_drafts`, `section_plan`) from main Survey Mode. Two graphs, independent state. Merging into unified multi-mode state would simplify `app.py` routing. Minor — works correctly as-is.
