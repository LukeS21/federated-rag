---
phase: 7
status: complete
tags: [vision, models, comparison, benchmark]
created: 2026-05-10
links: [gemma4-e4b, llava-7b, qwen3-vl-4b, model-rotation, vision-descriptor-api]
---

# Vision Model Comparison

Head-to-head evaluation of three Ollama vision models on real biomedical figures.

## Test Setup

- **Test figures**: 3 real figures from Avery 2024 paper (bar charts, scatter plots, microscopy)
- **Prompt**: "Describe this scientific figure in detail, including the methodology, experimental groups, and key findings."
- **Hardware**: Apple M2 unified memory

## Results

| | gemma4:e4b | qwen3-vl:4b | llava:7b |
|--|-----------|-------------|----------|
| **Size (disk)** | 9.6 GB | 2.6 GB | 4.7 GB |
| **Latency / figure** | ~17 s | ~10 s | ~5 s |
| **Biomedical terms found** | 5 | 1 | 0 |
| **Already loaded?** | YES | NO | NO |
| **Sample output** | "bar chart showing IL-6 levels in wild type vs. knockout mice under inflammatory conditions" | "gene/protein expression or activity shown in a heatmap-like display" | "scientific poster with graphs and text" |

### Biomedical Terms Detected

- **gemma4:e4b**: IL-6, cytokine, wild type, knockout, inflammatory — **5 terms**
- **qwen3-vl:4b**: expression — **1 term**
- **llava:7b**: — **0 terms**

## Verdict

**Clear winner: gemma4:e4b**

- Best biomedical vocabulary and description accuracy
- Already loaded for text generation tasks — no model rotation overhead
- Identifies specific cytokines, experimental groups, and methodologies
- Higher latency (17s) is acceptable given quality and zero rotation cost

### Fallback

If gemma4 is unavailable (e.g., memory-constrained environment), qwen3-vl:4b is a viable alternative at 2.6GB. llava:7b is not recommended for biomedical use due to poor domain knowledge.

## Implications

- No model rotation needed in the vision pipeline
- All figure descriptions go through gemma4:e4b by default
- Controlled via `VISION_MODEL` env var in [[vision-descriptor-api]]
