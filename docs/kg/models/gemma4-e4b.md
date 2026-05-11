---
phase: [5.5, 7]
status: active
tags: [models, gemma4, vision, fast-tier]
created: 2026-05-10
links: [model-tiering, qwen3-6-35b, llava-7b, vision-descriptor-api, model-comparison]
---

# gemma4:e4b — Model Card

## Role

**Fast tier model.** Handles per-chunk operations across the RAG pipeline:

- Per-theme Drafter (drafting individual survey sections)
- Query decomposition (breaking queries into biomedical concepts)
- Extraction (claim extraction from retrieved chunks)
- Summarization (chunk and section summarization)
- Gap analysis (identifying missing evidence)
- **Figure description** (biomedical figure analysis via multimodal input)

## Specifications

| Attribute | Value |
|-----------|-------|
| Size | 9.6 GB |
| Parameters | 8B (4B active experts) |
| Family | gemma4 |
| Architecture | MoE (Mixture of Experts) |
| Context window | 128K tokens |
| Multimodal | Yes — supports image input via Ollama REST API |

## Vision Capability

gemma4:e4b is the project's **sole vision model** for figure description. It produces biomedically-accurate descriptions identifying:

- Cell types (CD4+, CD8+, Tregs, macrophages)
- Cytokines (IL-6, TNF-α, IFN-γ, IL-10)
- Experimental groups (WT vs knockout, treated vs control)
- Assay types (flow cytometry, Western blot, ELISA, IHC)
- Quantitative trends (dose-response, time course, significance markers)

**Key advantage**: gemma4:e4b is already loaded during PDF ingestion as the text model. No model rotation is needed for figure description — zero switching overhead. See [[model-rotation]] for the abandoned alternative.

## Comparison

| Metric | gemma4:e4b | llava:7b | qwen3-vl:4b |
|--------|-----------|----------|-------------|
| Biomedical accuracy | ★★★ | ★☆☆ | ★★☆ |
| Latency (figure) | ~17s | ~12s | ~14s |
| Latency (text) | ~8–30s | N/A | N/A |
| Model rotation needed | No | Yes (~15–30s) | Yes (~15–30s) |
| Specificity | High | Low | Medium |

See [[model-comparison]] for detailed evaluation methodology and example outputs.

## Performance

| Task | Latency | Notes |
|------|---------|-------|
| Figure description | ~17s/figure | Single figure, ~200-token description |
| Text drafting | ~8–30s | Depends on prompt complexity |
| Extraction | ~5–15s | Per-chunk claim extraction |
| Summarization | ~10–20s | Per-section summarization |

## Memory Management

- **Loads** on first use (cold start ~5s)
- **Unloads** via `OLLAMA_KEEP_ALIVE=60s` — 60 seconds of idle timeout before freeing VRAM
- Unloads **before** qwen3.6:35b loads to avoid VRAM contention
- Peak combined memory with qwen3.6:35b: ~28 GB VRAM

See [[model-tiering]] for the full two-tier strategy.
