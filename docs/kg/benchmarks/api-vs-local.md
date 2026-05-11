---
phase: 6.5
status: complete
tags: [benchmarks, comparison, api, local]
created: 2026-05-10
links: [performance-history]
---

# API vs Local Comparison

1:1 comparison results (May 2026).

| Metric | DeepSeek API | Local | Delta |
|---|---|---|---|
| Themes matched | 6/6 | 6/6 | — |
| Avg anchoring | 0.969 | 0.947 | +0.022 |
| Per-theme claims | 119 | 96 | — |
| Cross-theme claims | 26 | 17 | — |
| Elapsed | 212 s | 524 s | 2.5x slower |
| Cost | ~$0.50 | Free | — |

## Methodology

Both run `build_survey_graph()` identically. Model tiering preserved.

Cloud results persist to `projects/default/comparison/cloud.json`.
