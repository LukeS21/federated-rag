---
phase: [1, 2, 3, 4, 5, 5.5, 6, 6.5]
status: complete
tags: [phases, history]
created: 2026-05-10
links: [phase-overview, system-overview]
---

# Phases 1–6.5 — Consolidated History

## Phase Summary

| Phase | Status | Key Deliverable |
|-------|--------|-----------------|
| Phase 1 — Foundation | Complete | State management, Unicode sanitization, citation primitives, retrieval abstractions |
| Phase 2 — PDF Ingestion | Complete | PDF parsing (PyMuPDF), chunking, hybrid retrieval (BM25 + vector) |
| Phase 3 — LLM Agents | Complete | LangGraph core, multi-agent debate, Drafter/Critic/Arbiter chain |
| Phase 4 — Citation | Complete | Live @citation tracking, survey mode (single-turn synthesis) |
| Phase 5 — Security | Complete | Air-gap validation, prompt injection guards, output sanitization |
| Phase 5.5 — Optimization | Complete | Local model tiering (gemma4:e4b fast, qwen3.6:35b reasoning), OLLAMA_KEEP_ALIVE tuning |
| Phase 6 — UI & Polish | Complete | Streamlit dashboard, survey viewer, graph visualization, export pipeline |
| Phase 6.5 — Gap Closure | Complete | Parallelization (async ingestion), context compression, fuzzer for prompt robustness, incremental cache |

## Key Technical Decisions

- **LangGraph** selected over raw LangChain for stateful multi-agent orchestration
- **Hybrid retrieval** (BM25 + vector) chosen over vector-only — BM25 catches keyword matches that embeddings miss for biomedical terms
- **Two-tier model architecture**: fast local model (gemma4) for per-chunk work, reasoning model (qwen3.6) for cross-chunk synthesis. No cloud dependency.
- **Air-gap mode**: all processing local, no external API calls during query. Validated via network monitoring in Phase 5.

## Architecture Diagram (Simplified)

```
PDFs → Ingest → Chunk → Embed → Vector Store
                                    ↓
Query → Decompose → Retrieve → Debate (Draft→Critique→Arbitrate) → Synthesize → Survey
                                    ↑
                              Knowledge Graph
```

See [[system-overview]] and the project README for full architectural details, component breakdown, and setup instructions.
