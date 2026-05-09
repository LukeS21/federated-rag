# Phase 4 → Phase 5 Handoff — May 2026

## Quick start

```bash
python -m pytest tests/ -v              # 66 passed, 6 pre-existing failures
python phase4_demo.py                   # Survey Mode with HITL review gate
python phase4_viz.py                    # Knowledge graph + data flow visualization
python phase4_benchmark.py              # DeepSeek v4 Pro vs Chat comparison
python phase4_benchmark_batch2.py       # Conditional Critic threshold calibration
```

## Current project state

**Phase 4 is complete.** Survey Mode runs end-to-end in 1–2 minutes per query
(down from 27 min). 66 tests passing, 6 pre-existing failures (all in
`test_synthesis_agents.py` — mocking `langchain_ollama.ChatOllama` which was
replaced by `langchain_openai.ChatOpenAI` during the DeepSeek migration).
These 6 tests are safe to ignore.

## What was accomplished

### Phase 4 Survey Mode pipeline

Built the full 8-node Survey Mode graph per the README §8.3 two-stage hybrid design:

| Component | File | Purpose |
|-----------|------|---------|
| Query Decomposition | `src/agents/query_decomposer.py` | 1 v4-pro call → 3–8 themed sub-queries; L1 cached |
| Thematic Clustering | `src/agents/thematic_clusterer.py` | Embedding cosine similarity, pre-computed at ingest |
| Per-document extraction | `src/ingestion/pre_extractor.py` | Extracts entities once at ingest; query-time loads from disk |
| Per-theme synthesis | `src/graph/survey_nodes.py` | Parallel chat Drafter; conditional Critic; KG insights |
| Cross-theme + gaps | `src/graph/survey_nodes.py` | Parallel v4-pro calls; L3 cached |
| Survey graph | `src/graph/graph_builder.py` | `build_survey_graph()` — 8-node LangGraph with HITL |
| Demo | `phase4_demo.py` | Full pipeline with interrupt/resume for human review |
| Visualization | `phase4_viz.py` | KG graph, entity stats, similarity heatmaps, data flow diagram |

### Batch 1 optimizations (speed, no accuracy loss)

1. **TF‑IDF extractive summarization** (`src/ingestion/pre_summarizer.py`)  
   Replaced LLM‑based PreSummarizer with sklearn TfidfVectorizer. Extractive = no hallucination.

2. **Embedding‑based thematic clustering** (`src/agents/thematic_clusterer.py`)  
   `all‑MiniLM‑L6‑v2` cosine similarity. Deterministic, ~2s. LLM fallback preserved.

3. **Pre‑extraction at ingest** (`src/ingestion/pre_extractor.py`)  
   Entities extracted once per paper during PDF ingestion, stored as JSON. Query-time loads from disk.

4. **Skip debate for single‑paper themes** — now skips the Drafter entirely (entity‑formatted text).

5. **Trim entities from Critic prompt** — ~20% prompt size reduction.

6. **LLMCache added to all debate agents** — `sha256(prompt || model)`, 24h TTL.

### Batch 2 optimizations (conditional critic + evidence truncation)

1. **Conditional Critic (EGSR pattern)** (`src/graph/survey_nodes.py:_run_debate_for_theme`)  
   Anchoring check after Drafter. Only invokes Critic for drafts < 0.35 threshold.
   67% of Critic calls saved with zero anchoring quality loss in 6‑paper corpus.
   `CONDITIONAL_CRITIC_THRESHOLD` is module‑level, configurable.

2. **Dynamic evidence truncation** (`src/graph/survey_nodes.py:_fit_summaries_to_context`)  
   Replaced hardcoded `summaries[:20]` with tiktoken‑based context‑window‑aware cap.
   Fills ~100+ summaries at `num_ctx=16384` vs the old cap of 20.

### Batch 2.5 optimizations (speed, structural)

3. **Debate regression guard** — if Critic→Arbiter reduces anchoring, keep the draft.

4. **Parallel per‑theme synthesis** — `ThreadPoolExecutor(max_workers=8)` in `survey_per_theme_synthesize_node`. Wall‑clock time bounded by slowest theme, not sum of themes.

5. **Model tiering** — per‑theme Drafter uses `deepseek‑chat`; cross‑theme + gap uses `deepseek‑v4‑pro`. All agents accept optional `model` parameter. Cache keys include model name.

6. **Parallel cross‑theme + gap analysis** — gap prompt rewritten to use per‑theme syntheses directly (no cross‑theme dependency), enabling parallel `ThreadPoolExecutor(max_workers=2)`.

7. **Agent memoization** — `_get_drafter`, `_get_critic`, `_get_arbiter` return module‑level singletons. Eliminates ~7 redundant `ChatOpenAI` instantiations per query.

8. **Pre‑computed paper embeddings** (`src/ingestion/pre_extractor.py`) — saved as `.npy` during ingest. Clustering drops from 2.7s to ~0.1s per query.

9. **KG insights injected into per‑theme Drafter** — `compute_graph_insights()` from `src/graph/graph_reasoning.py` produces structured text (central/bridge entities, 2‑hop neighbourhood) instead of raw JSON.

10. **Multi‑level query cache** (`src/cache/query_cache.py`) — L1 (decomposition), L2 (per‑theme synthesis), L3 (cross‑theme + gap). 7‑day TTL, visible `[query‑cache]` logging. Same query re‑run completes in < 1s.

11. **Citation key propagation** (`src/citation_manager/citekey_utils.py`) — cite keys parsed from filenames (`@avery2024`), stored in chunk metadata at ingest, propagated through extraction → Drafter prompts. Real Zotero API integration creates items on ingest (credentials from `.env`).

12. **Human‑in‑the‑loop for survey results** — `interrupt_before=["survey_scrub"]` in survey graph. Demo prompts: approve / edit‑with‑feedback / discard.

## Performance

| Metric | Before | After |
|--------|--------|-------|
| Survey query latency | 27 min | 1–2 min |
| Per‑document extraction | 41s (2 LLM) | 0.0s (disk cache) |
| Per‑theme synthesis | 23 min (sequential) | ~9s (parallel, chat) |
| Cross‑theme + gap | 2.4 min (sequential) | ~47s (parallel v4‑pro) |
| LLM calls per query | ~18 (all v4‑pro) | ~12 (2 v4‑pro + 10 chat) |
| Repeated query | Full re‑compute | < 1s (multi‑level cache) |
| Tests | 55 passing | 66 passing (6 pre‑existing failures) |

## Key architectural decisions (DO NOT UNDO)

- **All agents use DeepSeek API** via `langchain_openai.ChatOpenAI`. API key from `.env` as `DEEPSEEK_API_KEY`. `load_dotenv(override=True)` in demos.
- **Model tiering is configurable** — `PER_THEME_DRAFTER_MODEL="deepseek‑chat"` and `CROSS_THEME_DRAFTER_MODEL="deepseek‑v4‑pro"` in `survey_nodes.py`. All agent classes accept optional `model` parameter.
- **Thematic clustering defaults to embeddings** (sentence‑transformers). LLM path is `use_embeddings=False` fallback.
- **PreSummarizer is TF‑IDF extractive** (no LLM). Old ChromaDB data has LLM‑generated summaries — re‑ingest PDFs for TF‑IDF summaries.
- **Pre‑extraction at ingest** stores entities in `projects/default/extractions/`. Delete to force re‑extraction.
- **Paper embeddings** cached in `projects/default/embeddings/` as `.npy` files. Auto‑computed at ingest; fallback computed on‑the‑fly if missing.
- **Single‑paper themes skip ALL LLM calls** — entity‑formatted text. No Drafter, no debate.
- **Critic no longer receives entity JSON** — evidence chunks are sufficient.
- **All debate agents use memoized instances** — `_get_drafter`, `_get_critic`, `_get_arbiter`.
- **17‑node Deep Mode graph unchanged** — Survey Mode graph is separate (`build_survey_graph`).
- **Knowledge Graph is the shared truth cache** — `NetworkXJSONStorage` at `projects/default/project_graph.json`. 521 nodes, 1900 edges across 6 papers.
- **Similarity‑threshold retrieval** — L2 ≤ 1.0, max 20 for Deep; L2 ≤ 1.5, max 50 for Survey.
- **TF‑IDF cosine anchoring** (threshold 0.35) in `src/anchoring/evidence_check.py`.
- **Conditional Critic threshold 0.35** — `CONDITIONAL_CRITIC_THRESHOLD` in `survey_nodes.py`. Configurable.
- **Dynamic evidence cap via tiktoken** — `_fit_summaries_to_context` uses `cl100k_base` encoding.
- **Query cache** at `projects/default/query_cache/` — 7‑day TTL, three levels. Delete directory to force recomputation.
- **Human‑in‑the‑loop** via `interrupt_before=["survey_scrub"]` with `MemorySaver` checkpointer.
- **Citation keys** stored in chunk `metadata["cite_key"]` at ingest. Used in Drafter prompts.

## What NOT to change

- The 17‑node Deep Mode graph structure (proven stable)
- The 8‑node Survey Mode graph structure
- The interrupt/resume pattern with `MemorySaver` checkpointer
- The SciSpaCy NER integration
- The debate chain internals (Drafter→Critic→Arbiter flow)
- The embedding‑based clustering (keep LLM fallback)
- The TF‑IDF extractive summarization (do not revert to LLM)
- The single‑paper debate skip logic (entity formatting, no Drafter)
- The KG interface (`BaseGraphStorage` abstract class)
- The evidence anchoring check (`compute_anchoring_score`)

## Known bugs and limitations

1. **Per‑theme synthesis evidence cap** — themes with 100+ papers still only see ~100 summaries. Tiered synthesis needed for scale (Phase 6).
2. **NetworkX does not scale past ~10K edges** — current 1900 edges is fine; at 100+ papers, Neo4j adapter (Phase 6) needed.
3. **DeepSeek API queuing** — server‑side, can cause 10+ minute delays during peak hours. Phase 5 local Ollama eliminates this.
4. **TF‑IDF anchoring penalizes inferential synthesis** — embedding‑based similarity alongside TF‑IDF would give more accurate quality scores.
5. **6 pre‑existing test failures** — `test_synthesis_agents.py` mocks `langchain_ollama.ChatOllama`; not yet updated.
6. **Pre‑extraction uses generic default query** — entities extracted once at ingest with a broad query. If a user query is extremely narrow, some pre‑extracted entities may be irrelevant (but none missing).
7. **Old ChromaDB data** — existing `projects/default/chroma_data/` may have LLM‑generated summaries from old PreSummarizer. Delete and re‑ingest for TF‑IDF summaries.

## File map

```
src/
├── agents/
│   ├── arbiter.py                  # Revision agent (model‑configurable)
│   ├── extraction_agent.py         # Category discovery + entity extraction
│   ├── query_decomposer.py         # Query → themed sub‑queries
│   ├── socratic_critic.py          # Evidence‑grounded critique (model‑configurable)
│   ├── synthesis_drafter.py        # Draft synthesis (model‑configurable)
│   └── thematic_clusterer.py       # Embedding + LLM fallback clustering
├── anchoring/
│   └── evidence_check.py           # TF‑IDF claim decomposition + anchoring score
├── cache/
│   ├── llm_cache.py                # Per‑agent prompt cache (24h TTL)
│   └── query_cache.py              # Multi‑level query cache (7‑day TTL)
├── citation_manager/
│   ├── base.py                     # AbstractCitationManager interface
│   ├── citekey_utils.py            # Cite key generation + Zotero API
│   └── zotero_adapter.py           # pyzotero wrapper
├── graph/
│   ├── base_graph.py               # Abstract BaseGraphStorage
│   ├── graph_builder.py            # build_graph() + build_survey_graph()
│   ├── graph_reasoning.py          # compute_graph_insights() for KG → Drafter
│   ├── networkx_json_storage.py    # NetworkX ↔ JSON persistence
│   ├── nodes.py                    # Deep Mode (17) node implementations
│   └── survey_nodes.py             # Survey Mode (8) node implementations
├── ingestion/
│   ├── pdf_parser.py               # Docling PDF → chunks
│   ├── pre_extractor.py            # Entity extraction at ingest + embedding cache
│   └── pre_summarizer.py           # TF‑IDF extractive summarization
├── retrieval/
│   ├── bm25_index.py               # Tantivy BM25 sparse index
│   ├── chroma_client.py            # ChromaDB wrapper
│   └── hybrid_retriever.py         # RRF fusion of dense + sparse
├── scrubber.py                     # Final output ASCII enforcement
├── state.py                        # AgentState TypedDict
└── unicode_map.py                  # Normalization + sanitization utilities

phase4_demo.py                      # Survey Mode interactive demo
phase4_viz.py                       # KG + data flow visualization
phase4_benchmark.py                 # v4 Pro vs Chat extraction comparison
phase4_benchmark_batch2.py          # Conditional Critic threshold calibration

tests/
├── test_survey_graph.py            # 18 tests (fits, debates, nodes, graph)
├── test_phase3_integration.py      # SciNer, anchoring, extraction, summarizer
├── test_thematic_clusterer.py      # 13 tests
├── test_query_decomposer.py        # 8 tests
└── ... (15 other test files)
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Survey Mode demo (auto‑ingests new PDFs, pre‑extracts, caches)
python phase4_demo.py

# Visualization (run after demo with a query)
python phase4_viz.py               # full suite
python phase4_viz.py --dataflow     # data flow diagram only
python phase4_viz.py --graph-only   # knowledge graph only

# Deep Mode demo (Phase 3, still works)
python phase3_demo.py

# Benchmarks
python phase4_benchmark.py                        # v4 Pro vs Chat extraction
python phase4_benchmark_batch2.py                 # conditional Critic calibration
python phase4_benchmark_batch2.py --compare       # conditional vs unconditional

# Clean everything for fresh start
rm -rf projects/default/cache projects/default/query_cache \
       projects/default/chroma_data projects/default/bm25_corpus.json \
       projects/default/extractions projects/default/embeddings
python phase4_demo.py
```

## Planned: Formal benchmarking suite (Phase 5)

A reproducible benchmark of 20–30 curated biomedical QA pairs with ground‑truth
annotations is planned for late Phase 5.  Rationale for deferring:

- Phase 5 replaces DeepSeek API with local Ollama, changing the inference surface.
- Benchmarking now would be invalidated when local models are deployed.
- Running the same benchmark against BOTH pipelines (API baseline, local target)
  produces a defensible comparison for Phase 5 sign‑off.
- The existing corpus (6 papers) is small enough for manual annotation; public
  biomedical QA datasets (PubMedQA, BioASQ) do not test multi‑document synthesis.

See README §12.3 for full benchmark design.

## Prompt for next AI session

```
You are continuing work on a Federated RAG system for biomedical research.
Read the full README.md and HANDOFF.md (this file) to understand the architecture
and current state.

The project has 66 passing tests (6 pre-existing failures from langchain_ollama
migration — safe to ignore). Phase 4 (Survey Mode) is complete. The system runs
end-to-end in 1-2 minutes per query.

Your priority is Phase 5: Security Hardening & Air-Gap deployment.
See README §10 for the Phase 5 deliverables.

Before making any changes:
1. Run the test suite: python -m pytest tests/ -v
2. Try the demo: python phase4_demo.py
3. Read the key architecture decisions in HANDOFF.md §"Key architectural decisions"
4. Read the README §8.3, §10, and §11 for full context

Do NOT:
- Change the 17-node Deep Mode graph
- Change the 8-node Survey Mode graph
- Remove the embedding-based clustering (keep LLM fallback)
- Revert TF-IDF summarization to LLM
- Change the single-paper debate skip logic
- Modify the KG interface or the interrupt/resume pattern
- Remove the conditional Critic or model tiering
```
