---
phase: [4, 7]
status: reference
tags: [synthesis, debate, per-theme, survey-mode]
created: 2026-05-10
links: [survey-mode-graph, debate-chain, cross-theme-synthesis, evidence-anchoring, conditional-critic]
---

# Per-Theme Debate

The `_run_debate_for_theme()` function orchestrates the complete per-theme debate flow.

## Signature

```python
def _run_debate_for_theme(
    theme, query, entities, theme_chunks, kg,
    drafter, critic, arbiter, model_mgr, cache_mgr, ledger
)
```

## Step-by-Step Flow

### 1. Build Summaries

Build evidence summaries from `chunk_summary` metadata or entity-level evidence. Each summary is a compressed representation of a chunk's relevant content.

### 2. Figure Descriptions

Add figure descriptions via `_extract_figure_descriptions()`. This enriches evidence with visual content from [[gemma4-e4b]].

### 3. Dynamic Evidence Cap

Apply `_fit_summaries_to_context()` to dynamically cap evidence within context window limits. Uses tiktoken to count tokens and truncate as needed.

### 4. L2 Cache Check

Check L2 cache for existing synthesis results for this theme/query combination. Skip computation if cached result exists and is valid.

### 5. Single-Paper Branch

If only one paper covers this theme, format entities directly — no Drafter invocation. Single-source themes don't need multi-paper synthesis.

### 6. Multi-Paper Branch

For multiple papers, invoke the Drafter with KG insights via `compute_graph_insights()`. The KG provides relationship context between entities across papers.

### 7. Entity Compression

Compress entities via `_compress_entities_for_drafter()` to fit within context limits while preserving the most relevant biomedical concepts.

### 8. Conditional Critic

Invoke Critic only if anchoring score < 0.50 (see [[conditional-critic]]). Well-grounded drafts skip the full debate chain.

### 9. Debate Regression Guard

If debate *worsens* the anchoring score, keep the original draft. The Critic/Arbiter should improve quality, not degrade it.

## include_figures Integration

The `include_figures` parameter controls whether figure descriptions are injected. Set to `True` in sectioned_retrieve.
