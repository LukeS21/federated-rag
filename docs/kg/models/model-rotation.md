---
phase: 7
status: abandoned
tags: [models, architecture, decisions]
created: 2026-05-10
links: [vision-model-reuse, gemma4-e4b, no-model-rotation]
---

# Model Rotation — Why It Was Abandoned

## Original Plan

The Phase 7 vision pipeline was designed with model rotation:

```
1. Unload text model (gemma4:e4b)
2. Pull vision model (llava:7b or qwen3-vl:4b)
3. Load vision model
4. Generate figure description
5. Unload vision model
6. Reload text model (gemma4:e4b)
```

**Estimated overhead**: 15–30s per query for model switching, plus 12–17s for the description itself.

## Discovery

During implementation, we discovered that **gemma4:e4b** — already loaded as the fast-tier text model — supports **multimodal input** via the Ollama REST API. This meant we could send figure images to the same model instance handling text tasks.

## Decision

**Reuse gemma4:e4b for figure descriptions** instead of pulling a dedicated vision model. Benefits:

| Factor | Rotation Approach | Reuse Approach |
|--------|------------------|----------------|
| Model switching | 2 unloads + 2 loads | 0 |
| Switching latency | 15–30s | 0s |
| Total query time | 27–47s (19%) overhead | 0% overhead |
| Biomedical accuracy | Variable by model | Consistently high |
| System complexity | High (load orchestration) | Low (single model) |

## num_predict Bug

During implementation, we hit a bug: passing `num_predict` to Ollama's multimodal API causes **empty responses**. This affects both gemma4:e4b and any dedicated vision model loaded via the same API.

**Workaround**: send only `temperature` as a generation parameter. Truncate the response post-generation if length control is needed.

This bug is logged with Ollama and expected to be fixed upstream. When fixed, we can re-enable `num_predict` for figure descriptions.

## Key Lesson

> **Always check whether existing models support multimodal before pulling dedicated vision models.**

This pattern — reusing an active text model for vision tasks — is a novel approach in the local-model RAG space. It avoids:
1. VRAM contention from simultaneous model loads
2. Model switching latency
3. Additional model management complexity
4. Downtime between text and vision phases

## Related Decisions

- [[gemma4-e4b]] — the model that made this possible
- [[llava-7b]] — original vision model, replaced
- [[qwen3-vl-4b]] — tested alternative, not needed
- [[no-model-rotation]] — the positive outcome of this decision
