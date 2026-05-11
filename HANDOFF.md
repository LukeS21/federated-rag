# Phase 7 → Phase 8 Handoff — May 2026

## Quick start

```bash
# Tests (201 passing, 0 failures)
python -m pytest phase5_benchmark.py test_correctness.py phase6_security_fuzzer.py phase6_multi_run.py phase5_api_comparison.py tests/ -v

# Tier A programmatic benchmark (zero LLM calls)
python phase5_benchmark.py

# Correctness tests (false claims, OOC, Discussion overlap, grounded/inferential)
python -m pytest test_correctness.py -v

# Security fuzzer (regex + GLiNER-PII, 100% regex detection)
python phase6_security_fuzzer.py --no-gliner --samples 15 --clean 100

# Vision pipeline benchmark (extract + filter, zero LLM, ~60s)
python phase7_vision_benchmark.py

# Vision pipeline with descriptions (~5 min, uses gemma4:e4b)
python phase7_vision_benchmark.py --describe

# Vision quality check (instant if cached)
python phase7_vision_quality.py

# Sectioned survey (requires Ollama, ~1 min)
python phase7_section_writing.py

# Sectioned survey cached results (instant)
python phase7_section_writing.py --cached

# Baseline comparison: naive RAG vs full pipeline (instant if cached)
python phase7_baseline_comparison.py --cached

# Literature discovery POC (instant if cached)
python phase9_pubmed_demo.py --cached

# Streamlit UI (localhost:8501, all 4 modes available)
streamlit run app.py
```

## Current project state

**Phase 7 complete. All known gaps closed.** 201 tests passing (0 failures).
The system runs entirely on local Ollama models (gemma4:e4b for per‑theme, gap
analysis, and figure description; qwen3.6:35b for cross‑theme synthesis, critic,
and arbiter). Vision pipeline is fully integrated into PDF ingestion. Sectioned
survey mode is available in the Streamlit UI alongside Survey, Deep, and Quick
modes. Literature discovery POC and baseline comparison are built with cached
results for offline review.

> ⚠ **Important scale caveat**: All benchmarks, tests, and comparisons are at
> **6 papers** from a single lab with heavy topical overlap.  Anchor scores at
> this scale primarily measure traceability (keyword matching), not factual
> accuracy or cross‑paper inference quality.  The evaluation framework is proven
> to work, but specific scores (0.993 anchoring, 88% grounded) should NOT be
> interpreted as production‑grade quality metrics.  They will change significantly
> at scale.  Phase 8 (100+ papers) is where definitive benchmarking belongs.

**Vision pipeline (Phase 7a)**: gemma4:e4b describes figures directly — no model
rotation needed since it's already loaded as the fast-tier text model.  Produces
biomedically-accurate descriptions (identifies IL-6, CD4, CD8, cytokines,
WT/knockout groups, significance markers from figure labels).  Figure filtering
via Docling's `DocumentFigureClassifier-v2.5` correctly discards logos/icons/
thumbnails (80.9% keep rate, zero data figures lost).  Figures embedded into
ChromaDB with `chunk_type="figure"` metadata; `include_figures=True` flag
extends `HybridRetriever.query()` for cross-modal retrieval.

**Multi-turn synthesis (Phase 7b)**: Sectioned survey graph (8 nodes) writes
IMRaD sections iteratively with interrupt-at-review checkpointing.  Claim ledger
uses SHA-256 normalization for cross-section deduplication; tracks citation
coverage, grounded/ungrounded status, and per-section validation.  Sectioned mode
available in app.py alongside existing Survey/Deep/Quick modes.

**Baseline comparison**: Full pipeline produces 27× more claims (134 vs 5) than
naive single-pass RAG, with essentially identical anchoring quality (0.993 vs
1.000).  Naive RAG's perfect score is a small-sample artifact — only 5
conservative claims, all trivially matched.  The pipeline's value is in coverage
breadth and cross-paper synthesis, not grounding precision.

**Literature discovery POC (Phase 9)**: Semantic Scholar and PubMed wrappers
built (`src/retrieval/semantic_scholar.py`, `src/retrieval/pubmed.py`).  87%
of papers found via external search are novel vs our 6-paper corpus.  Static
keyword extraction used for POC only — planned LLM-based Coverage Check node
replaces this in full Phase 9 implementation.

## What was accomplished in Phase 7

### Phase 7a: Vision Pipeline

| Component | File | Status |
|-----------|------|--------|
| Figure extraction | `src/vision/figure_extractor.py` | ✅ Extraction via Docling with `do_picture_classification=True` |
| Smart filtering | `src/vision/figure_filter.py` | ✅ Docling classifier (65% weight) + caption/size/page soft hints |
| Vision model integration | `src/vision/vision_descriptor.py` | ✅ gemma4:e4b via Ollama REST API. Fixed `num_predict` bug. |
| Figure-to-text embedding | `src/vision/figure_embedder.py` | ✅ ChromaDB with `chunk_type="figure"`, monkey-patched `include_figures=True` |
| Vision ingest hook | `src/vision/vision_ingest.py` | ✅ `vision_ingest_pdf()` called during app.py PDF ingestion |
| Figure filter | 23 tests | ✅ Unit + integration (real PDFs) |
| Figure extraction | 7 tests | ✅ Caption resolution, classification data, real PDFs |
| Vision descriptor | 13 tests | ✅ Mocked API, encoding, ASCII scrubbing, model lifecycle |
| Figure embedder | 7 tests | ✅ Cross-modal retrieval, metadata validity |
| Vision integration | 4 tests | ✅ End-to-end with real PDFs |

### Phase 7b: Multi-Turn Synthesis

| Component | File | Status |
|-----------|------|--------|
| Claim/citation ledger | `src/synthesis/claim_ledger.py` | ✅ SHA-256 dedup, @citation parsing, coverage, persistence |
| Sectioned survey nodes | `src/graph/sectioned_survey_nodes.py` | ✅ 7 nodes: init, retrieve, draft, review, route, assemble, scrub |
| Sectioned survey graph | `src/graph/sectioned_survey_graph.py` | ✅ 8-node LangGraph with interrupt-at-review |
| State extensions | `src/state.py` | ✅ 6 new fields for sectioned survey |
| Figure→synthesis wiring | `src/graph/survey_nodes.py` | ✅ `include_figures=True` in survey_retrieve; `_extract_figure_descriptions()` in debate |
| Drafter citation fix | `src/agents/synthesis_drafter.py` | ✅ Removed hardcoded `@author2025`; "use ONLY provided keys" constraint |
| Claim ledger tests | 14 tests | ✅ Unit tests for all operations |
| Sectioned survey tests | 9 tests | ✅ Graph compilation, init node, integration |

### POC and Benchmarking

| Component | File | Status |
|-----------|------|--------|
| PubMed wrapper | `src/retrieval/pubmed.py` | ✅ NCBI E-utilities (esearch, efetch, XML parsing) |
| Semantic Scholar wrapper | `src/retrieval/semantic_scholar.py` | ✅ Search, paper lookup, DOI lookup |
| Literature discovery demo | `phase9_pubmed_demo.py` | ✅ Searches S2/PubMed, compares vs local corpus, caches results |
| Baseline comparison | `phase7_baseline_comparison.py` | ✅ Naive RAG vs full pipeline with 6 metrics |
| Sectioned survey demo | `phase7_section_writing.py` | ✅ Multi-section writing with ledger, caches results |

## Lessons learned in Phase 7

### 1. Vision model selection matters — and reuse eliminates rotation

We compared three multimodal models on actual BME figures. llava:7b (2023):
generic "scientific poster." qwen3-vl:4b: "gene/protein expression." gemma4:e4b:
names IL-6, CD4, CD8, cytokines, and WT/knockout groups directly from figure
labels. The winning model was already our fast-tier text model — zero rotation
overhead. The architecture originally planned model rotation (unload text →
load vision → describe → swap back); this turned out to be unnecessary.

### 2. num_predict breaks multimodal Ollama models (known bug)

Passing `num_predict` in Ollama's `/api/generate` options causes multimodal models
to return empty responses with `done_reason=length`. `temperature` alone works
fine. Workaround: send temperature only; enforce max_tokens via post-generation
truncation. This may be fixed in future Ollama versions — re-test before removing
the workaround.

### 3. Docling picture annotations are classification data, not captions

`PictureItem.annotations` contains classification metadata like
`kind='classification' provenance='DocumentPictureClassifier'`. Real figure
captions are in `PictureItem.captions` which reference `DoclingDocument.texts[idx].text`
via `#/texts/{idx}` refs. This mistake caused every extracted figure to show
raw classification strings instead of "Fig. 1. Characterization of..."

### 4. Monkey-patch import order is critical

The `include_figures=True` monkey-patch on `HybridRetriever.query()` is applied
at module import time in `src/vision/figure_embedder.py`. Any script calling
`retriever.query(include_figures=True)` must `import src.vision.figure_embedder`
BEFORE importing `HybridRetriever`. Failure produces `TypeError: unexpected
keyword argument 'include_figures'`.

### 5. @author2025 was a hardcoded prompt example, not a data problem

All claims in early sectioned survey output cited `@author2025`. This was traced
to the Drafter system prompt: `"claims with inline citation keys (@author2025)"`.
The LLM was literally following the example format. Fix: changed prompt to
"Use ONLY the exact citation keys provided — never invent new ones." Cache
version bumped to v3.

### 6. Baseline comparison validates the architecture

Full pipeline (134 claims, 0.993 anchoring) vs naive RAG (5 claims, 1.000
anchoring). The naive RAG's perfect score is misleading — it's easy to ground
5 safe claims. The pipeline maintains quality across 27× more claims spanning
5 themes. Every SOTA paper needs this comparison; publish alongside every
Phase 8 benchmark.

### 7. Static keyword extraction is insufficient for novelty detection

Phase 9 POC ranked "rat" as the #1 novel keyword because it appears frequently
in biomaterials papers — despite our lab using mouse models. The static keyword
list has no understanding of query relevance or lab-specific context. The planned
Phase 9 Coverage Check LLM node replaces this entirely.

## Key architectural decisions (DO NOT UNDO)

All previous DO NOT UNDO from Phase 4–6.5 still apply. Additional Phase 7
decisions:

- **Vision model = gemma4:e4b** — already loaded as fast-tier text model. No rotation. Better accuracy than dedicated vision-only models. Configurable via `VISION_MODEL` env var.
- **Figure filtering is classification-first** — Docling's `DocumentFigureClassifier-v2.5` at 65% weight. Size/caption/page are soft hints at 35% combined. Threshold 0.35 configurable.
- **Figures embedded in ChromaDB with chunk_type="figure"** — same collection as text. BM25 is NOT updated (figure text is AI-generated, not author-authored). `include_figures=True` on `HybridRetriever.query()` for cross-modal retrieval.
- **Monkey-patch import order**: `import src.vision.figure_embedder` BEFORE `from src.retrieval.hybrid_retriever import HybridRetriever`.
- **Sectioned survey is a separate LangGraph** — 8-node graph (`build_sectioned_survey_graph()`), independent from Survey/Deep Mode graphs. Interrupt at review for human-in-the-loop.
- **Claim ledger uses SHA-256 content addressing** — 16-char hex digests of normalized claim text. Stable across sessions. Persisted to `projects/default/section_ledger.json`.
- **Drafter prompt enforces citation discipline** — "Use ONLY the exact citation keys provided — never invent new ones." Post-generation citation validation not yet implemented (deferred to Phase 8).
- **num_predict workaround**: VisionDescriptor sends `temperature` only. `max_tokens` enforced via post-generation string truncation (~4 chars/token).
- **Cache version = v3** — bumped for Drafter prompt changes. Clear cache dirs after prompt modifications.

## What NOT to change

All previous What NOT to change from Phase 4–6.5 still apply. Additional Phase 7
constraints:

- Do NOT remove `include_figures=True` from `survey_retrieve_node()` — figures are now part of the evidence stream
- Do NOT revert vision model default from gemma4:e4b to llava:7b — gemma4 is superior for biomedical figures and eliminates rotation
- Do NOT remove `_extract_figure_descriptions()` from `_run_debate_for_theme()` — it formats figure descriptions into the Drafter's evidence
- Do NOT re-add `num_predict` to VisionDescriptor options without testing on the current Ollama version
- Do NOT change `picture.captions → doc.texts[idx].text` resolution back to `picture.annotations` — the latter is classification data
- Do NOT remove the monkey-patch import-order constraint without a cleaner integration approach (e.g., moving the patch into HybridRetriever's module)
- Do NOT delete the Drafter's "use ONLY exact citation keys" constraint — it prevents citation hallucination
- Do NOT remove the 4 Sectioned Survey modes (Survey/Deep/Quick/Sectioned) from app.py without replacing them
- Do NOT change `CACHE_VERSION` from v3 without understanding cached-prompt invalidation consequences

## Current known issues

1. **`OLLAMA_CONTEXT_LENGTH=32768` hardcoded by Ollama** — cannot override via LangChain. Requires native API or Modelfile.

2. **6-paper corpus limits synthesis depth** — 88% grounded claims. At 100+ papers, inferential rate should rise naturally. Phase 8 scale will expose pipeline behavior under diverse evidence.

3. **num_predict bug in Ollama multimodal endpoint** — workaround in place (post-generation truncation). Re-test after Ollama updates.

4. **Figure descriptions not regenerated on re-ingest** — vision pipeline skips already-ingested PDFs. To force re-description, delete the ChromaDB entry for that PDF.

5. **Sectioned survey graph uses separate state from main survey** — the sectioned graph uses `section_drafts` and `section_plan` fields that don't exist in the standard Survey Mode state. Merging these into a unified multi-mode state would simplify app.py routing.

6. **Phase 9 POC keyword extraction is POC-only** — static keyword list generates noise (e.g., "rat" as top keyword). Planned LLM-based Coverage Check node replaces this in full Phase 9.

7. **Baseline comparison is single-run** — cached result is from one query. Multi-query variance analysis would strengthen the comparison. Deferred to Phase 8 scale benchmarking.

8. **Cross-theme synthesis quality gap** — anchoring on cross-theme text is lower than per-theme (0.56 vs 0.95+). LLM-as-Judge is the only current evaluation tool. Phase 8 should include this metric.

## File map (new and changed in Phase 7)

```
NEW FILES (Phase 7):
src/vision/__init__.py                        # Vision module exports
src/vision/figure_extractor.py                # Docling figure extraction + classification
src/vision/figure_filter.py                   # Classification-weighted relevance scoring
src/vision/vision_descriptor.py               # Ollama multimodal API + model lifecycle
src/vision/figure_embedder.py                # ChromaDB embedding + HybridRetriever monkey-patch
src/vision/vision_ingest.py                  # PDF ingest integration hook
src/synthesis/__init__.py                    # Synthesis module
src/synthesis/claim_ledger.py                # Claim/citation ledger with SHA-256 dedup
src/graph/sectioned_survey_nodes.py          # 7 nodes for multi-turn section writing
src/graph/sectioned_survey_graph.py          # 8-node sectioned survey LangGraph
src/retrieval/pubmed.py                      # NCBI E-utilities PubMed client
src/retrieval/semantic_scholar.py            # Semantic Scholar API client
tests/vision/__init__.py                     # Vision test package
tests/vision/test_figure_extraction.py       # 7 tests: real PDFs, captions, hashing
tests/vision/test_figure_filter.py           # 23 tests: scoring, weights, edge cases
tests/vision/test_vision_descriptor.py       # 13 tests: API, encoding, lifecycle
tests/vision/test_figure_embedder.py         # 7 tests: cross-modal retrieval, metadata
tests/vision/test_vision_integration.py      # 4 tests: e2e with real PDFs
tests/synthesis/__init__.py                  # Synthesis test package
tests/synthesis/test_claim_ledger.py         # 14 tests: dedup, coverage, persistence
tests/synthesis/test_sectioned_survey.py     # 9 tests: graph, init node, integration
phase7_vision_benchmark.py                   # Figure extraction + filtering benchmark
phase7_vision_quality.py                     # Description quality metrics
phase7_section_writing.py                    # Sectioned survey demo (caches results)
phase7_baseline_comparison.py               # Naive RAG vs full pipeline comparison
phase9_pubmed_demo.py                        # Literature discovery demo (caches results)

MODIFIED FILES (Phase 7):
src/state.py                                 # Added 6 sectioned survey fields
src/graph/survey_nodes.py                    # include_figures=True; _extract_figure_descriptions()
src/agents/synthesis_drafter.py             # Fixed citation prompt; anti-hallucination constraint
src/cache/__init__.py                        # CACHE_VERSION=v3
.env                                         # VISION_MODEL=gemma4:e4b
app.py                                       # Vision ingest + Sectioned Survey mode + UI tabs
README.md                                    # Phase 7 status, lessons, Phase 8+9 planning

PROJECT DATA (auto-generated):
projects/default/figures/                    # Extracted figure PNGs per PDF
projects/default/vision_scorecard.json       # Figure extraction stats
projects/default/vision_quality_scorecard.json  # Description quality metrics
projects/default/figure_descriptions.json    # Cached vision descriptions
projects/default/section_ledger.json         # Claim/citation ledger (persistent)
projects/default/sectioned_survey_result.json  # Sectioned survey output
projects/default/baseline_comparison.json    # Naive RAG vs pipeline comparison
projects/default/literature_discovery.json   # External literature search results
```

## Current model configuration

| Tier | Model | Size | Purpose |
|------|-------|------|---------|
| Fast (small) | `gemma4:e4b` | 9.6 GB | Per‑theme Drafter, extraction, summarization, gap analysis, **figure description** |
| Reasoning (large) | `qwen3.6:35b` | 23 GB | Cross‑theme synthesis, critique, arbitration |
| Vision | `gemma4:e4b` | 9.6 GB | Same as fast tier — figures described inline, zero rotation overhead |
| Alt (disabled) | — | — | `PER_THEME_MODEL_B` unset; single‑model parallel used instead |

Memory: fast tier (9.6 GB) handles text + vision. Reasoning tier (23 GB) loads
separately. Peak ~28 GB, fits in 36 GB M3 Max. No third model needed.

## Performance (local Ollama, M3 Max 36 GB)

| Metric | Phase 6.5 | Phase 7 |
|--------|-----------|---------|
| Survey query latency | ~8.7 min | ~8.7 min (unchanged) |
| Figure extraction | — | ~60s (all PDFs, Docling) |
| Figure description/fig | — | ~17s (gemma4:e4b) |
| Sectioned survey | — | ~50s (4 IMRaD sections) |
| Baseline (naive RAG) | — | ~30s |
| Tests | 117 | 201 |
| LLM calls per survey query | ~8–10 | ~8–10 (unchanged) |

## Phase 8 initiation order (recommended)

Phase 8 targets publication scale: 100+ papers, sub‑minute queries.  This is
where definitive benchmarking belongs — all current scores are at 6 papers and
should not be interpreted as production‑grade.

1. **Acquire 100+ paper corpus** — use `phase9_pubmed_demo.py` to identify and download papers via Semantic Scholar/PubMed. Target: diverse BME papers across labs, years, and sub-topics.

2. **Neo4j adapter** (~4–6 hrs) — implement `Neo4jStorage` satisfying `BaseGraphStorage`. One config value swaps all consumers. Unblocks graph scalability (100K+ edges vs current ~2K).

3. **Hierarchical clustering** (~3–5 hrs) — two-level: broad topic via embeddings (Level 1), fine-grained themes via LLM (Level 2). Avoids O(n) context explosion.

4. **Corpus-level claim index** (~3–4 hrs) — pre-extract claims at ingest, index in dedicated ChromaDB collection. Query-time synthesis draws from claim store.

5. **Multi-tier caching L0–L4** (~2–3 hrs) — L0: corpus claims, L4: sectioned manuscript cache. Extends existing L1–L3.

6. **Full-scale benchmarks** (~2–4 hrs) — run the full pipeline at 100+ papers. Document anchoring drift, inferential rate, latency. Compare against 6-paper baseline.

## Prompt for next AI session

```
You are an expert senior software developer continuing work on a Federated RAG
system for biomedical research. Read the full README.md (especially the Phase
Recap, §Phase 7 Status, §Phase 8 Initiation Plan, and §Phase 9 Preview) and
this HANDOFF.md to understand the architecture and current state.

The project has 201 passing tests (0 failures). Phase 7 is complete — vision
pipeline, multi-turn section writing, claim ledger, baseline comparison, and
literature discovery POC are all built, tested, and wired. The system runs on
local Ollama models (gemma4:e4b for per‑theme, figure description, and gap
analysis; qwen3.6:35b for cross‑theme synthesis, critic, and arbiter).

Key Phase 7 additions you need to know:
  - Vision pipeline: gemma4:e4b describes figures during PDF ingest (no model rotation)
  - Smart filtering: Docling DocumentFigureClassifier-v2.5 keeps 80.9% of figures
  - Sectioned survey: 8-node LangGraph for IMRaD section writing
  - Claim ledger: SHA-256 dedup, citation coverage tracking, JSON persistence
  - Drafter citation fix: removed hardcoded @author2025, anti-hallucination constraint
  - Baseline comparison: full pipeline 27× more claims than naive RAG with equal grounding
  - Literature discovery POC: PubMed + Semantic Scholar wrappers, 87% novelty rate
  - num_predict bug workaround: send temperature only in VisionDescriptor

Phase 8 (Publication-Scale Retrieval) is next — see README.md §"Phase 8
Initiation Order" for the detailed plan and time estimates.

Phase 9 is planned with POC built. The full implementation requires Phase 8
prerequisites (Neo4j, hierarchical clustering, corpus-level claim index).

Before making any changes:
  1. Run the test suite: python -m pytest phase5_benchmark.py test_correctness.py phase6_security_fuzzer.py phase6_multi_run.py phase5_api_comparison.py tests/ -v
  2. Read HANDOFF.md §"Key architectural decisions" and §"What NOT to change"
  3. Read README.md §"Phase 7 Lessons Learned" for pitfalls to avoid

Do NOT:
  - Remove include_figures=True from survey_retrieve_node()
  - Revert vision model from gemma4:e4b to llava:7b
  - Remove _extract_figure_descriptions() from _run_debate_for_theme()
  - Re-add num_predict to VisionDescriptor options
  - Change picture.captions resolution back to picture.annotations
  - Remove the Drafter's "use ONLY exact citation keys" constraint
  - Revert to verbose prose Drafter format (keep dense claims)
  - Revert to DeepSeek API as default (Ollama is default)
  - Change the 17-node Deep Mode, 8-node Survey Mode, or 8-node Sectioned Survey graphs
  - Remove security modules or the unified LLM provider
  - Change condition critique threshold or debate chain structure without benchmarks
  - Remove hybrid retrieval, calibration framework, or GLiNER restrictions
  - Exceed PER_THEME_MAX_WORKERS=2 without memory testing
```

## Git summary (this session)

```
NEW FILES:
  src/vision/**, src/synthesis/**, src/retrieval/pubmed.py,
  src/retrieval/semantic_scholar.py, src/graph/sectioned_survey_*.py,
  tests/vision/**, tests/synthesis/**, phase7_*.py, phase9_pubmed_demo.py

MODIFIED FILES:
  src/state.py, src/graph/survey_nodes.py, src/agents/synthesis_drafter.py,
  src/cache/__init__.py, .env, app.py, README.md, HANDOFF.md

NEW PROJECT DATA:
  projects/default/figures/**, vision_scorecard.json,
  vision_quality_scorecard.json, figure_descriptions.json,
  section_ledger.json, sectioned_survey_result.json,
  baseline_comparison.json, literature_discovery.json

TESTS: 117 → 201 (+84 new tests, 0 failures)
```
