---
phase: 3
status: reference
tags: [synthesis, drafter, agents]
created: 2026-05-10
links: [debate-chain, dense-claim-format, citation-handling, per-theme-debate]
---

# Drafter Architecture

The `SynthesisDrafter` class handles drafting of dense claims from entity/chunk/citation context.

## Constructor

- `model_name` — Ollama model identifier
- `num_ctx` — context window size
- `client_kwargs` — additional Ollama client parameters
- `callback` — optional progress callback
- `model` — resolved model object

## draft() Signature

```
draft(query, entities, chunks, citations, kg_context)
```

- `query` — the user's research query
- `entities` — extracted biomedical entities
- `chunks` — evidence chunks with metadata
- `citations` — available citation keys
- `kg_context` — knowledge graph insights

## System Prompt

- "Format: one claim per line. No preamble, no transitions, no repetition."
- "Use ONLY exact citation keys provided — never invent new ones."

## User Prompt Template

The user prompt includes:
1. **Query** — the research question
2. **Extracted Entities** — biomedical entities from query and corpus
3. **Evidence Summaries** — compressed chunk summaries
4. **Available Citations** — valid citation keys from Zotero
5. **Knowledge Graph Context** — structured KG insights

## LLM Cache Integration

Responses are cached via LLM cache layer (see [[cache-version-v3]]). Cache version is prepended to cache keys.

## ASCII Scrubbing

Output is cleaned of non-ASCII characters to ensure compatibility with downstream text processing.

## _strip_thinking()

Qwen models wrap reasoning in `<think>...</think>` blocks. The `_strip_thinking()` method removes these blocks before parsing claims. Applies regex to strip everything between `<think>` and `</think>` inclusive.

## Performance Logging

Logs per-draft metrics:
- Prompt characters
- Output characters  
- Latency (ms)
