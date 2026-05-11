---
phase: 7
status: handoff
created: 2026-05-10
tags:
  - dev
  - handoff
links:
  - "[[dashboard]]"
  - "[[phase-overview]]"
  - "[[phase-8-initiation]]"
---

# HANDOFF

## Quick Start

```bash
# Set up environment
cp .env.example .env
# Edit .env with your settings

# Pull required models
ollama pull gemma4:e4b
ollama pull qwen3.6:35b

# Run test suite
python -m pytest tests/

# Launch UI
streamlit run app.py
```

## Current State

Phase 7 (Sectioned Survey + Vision Pipeline) is complete. All benchmarks pass at 6-paper scale. The system produces sectioned survey outputs with evidence anchoring, vision-augmented figure descriptions, and cross-paper inference tracking. Phase 8 (Scale to 100+ papers) is next.

## What Was Accomplished

- **Phase 6**: Grounded vs. inferential claim taxonomy. Calibrated LLM judge. Evidence anchoring with chunk-level matching. Correctness benchmark suite. Tier A benchmarks.
- **Phase 7**: Sectioned survey graph (separate state from main Survey Mode). Vision pipeline: figure extraction, classification (DocumentFigureClassifier-v2.5), description via multimodal LLM. SHA-256 content-addressed claims. Model reuse pattern (gemma4:e4b for both text and vision). Cross-section claim dedup. Baseline comparison. Gap analysis. Concurrency with per-theme workers.

## Lessons Learned

- **Model reuse works**: Testing existing models for multimodal capability before pulling dedicated vision models saves time and complexity.
- **Agreeableness bias is real**: Small models (gemma4:e4b) scored every claim 5/5 in faithfulness evaluation. Calibrate judges before trusting them.
- **Classifier-first filtering is effective**: `DocumentFigureClassifier-v2.5` at 0.99+ confidence eliminates heuristic tuning.
- **Content addressing scales**: SHA-256 hashes provide stable, collision-resistant claim IDs without vector search overhead.

## Key Decisions (DO NOT UNDO)

1. **Gemma4:e4b as VISION_MODEL**: Reuses text model for vision; eliminates model rotation. Do not replace with separate vision model unless proven necessary at scale.
2. **Content-addressed claims**: SHA-256 claim IDs are fundamental to dedup architecture. Do not replace with UUIDs.
3. **Grounded vs. inferential taxonomy**: Core to correctness evaluation. Do not collapse into single correctness score.
4. **Separate sectioned survey state**: Works correctly. Do not refactor into unified state until Phase 8 proves it necessary.
5. **Classifier-first figure filtering**: DocumentFigureClassifier-v2.5 is the primary gate. Do not replace with heuristics.

## What NOT to Change

- `VISION_MODEL=gemma4:e4b` — reuse pattern is proven
- `BaseGraphStorage` interface — already abstracted, one config swaps backends
- `OLLAMA_KEEP_ALIVE=60s` — tuned for current memory usage
- `PER_THEME_MAX_WORKERS=2` — prevents memory exhaustion
- Sectioned survey state separation — works correctly, deferred refactor
- Claim ID format (first 16 hex of SHA-256) — collision-resistant, human-readable

## Known Issues

- [[num-predict-bug]]: `num_predict` causes empty multimodal responses (Ollama bug, workaround in place)
- [[context-length-hardcoded]]: Cannot override per-model context length via ChatOpenAI
- [[figures-no-regen]]: Improved models won't retroactively update existing figure descriptions
- [[scale-caveat]]: All benchmarks at 6 papers; definitive benchmarking at Phase 8
- [[keyword-limitation]]: Static keyword extraction produces noise (replaced in Phase 9)
- [[cross-theme-quality]]: Cross-theme anchoring 0.56 vs per-theme 0.95+

## File Map

See [[file-map]] for full layer-by-layer file reference.

## How to Run

See [[quick-start]] for all run commands organized by category.

## Model Configuration

| Model | Purpose | Phase |
|---|---|---|
| `gemma4:e4b` | Classification, vision, simple tasks | 5+ |
| `qwen3.6:35b` | Synthesis, complex reasoning | 6+ |
| DeepSeek chat/v4-pro | Gap analysis, LLM judge | 6+ |

Memory: `OLLAMA_KEEP_ALIVE=60s`, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_MAX_QUEUE=8`.

## Performance

At 6 papers from single lab:
- **Grounded claims**: 88%
- **Per-theme anchoring**: 0.95+
- **Cross-theme anchoring**: 0.56
- **Correctness (grounded)**: High agreement
- **Classification confidence**: 0.99+ on bar charts

Expect grounded rate to decrease and inferential rate to increase at Phase 8 scale (100+ papers, diverse sources).

## Prompt for Next Session

Continue with Phase 8 initiation: scale corpus to 100+ papers from multiple labs. Implement Neo4jStorage adapter. Run definitive benchmarks at scale. Include cross-theme quality as primary metric. Address multi-query variance in baseline comparison.
