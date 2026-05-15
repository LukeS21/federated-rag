# Federated RAG — Phase Build History

> Extracted from the README and consolidated on 15 May 2026.
> This documents the build history. For current architecture, see README.md.
> For next-phase handoff, see HANDOFF.md.

---

## Phase Summary

| Phase | Name | Status |
|-------|------|--------|
| 1 | Foundation (state, unicode, citation, retrieval primitives) | ✅ Complete |
| 2 | PDF Ingestion & Hybrid Retrieval | ✅ Complete |
| 3 | LLM Agents & LangGraph Core (extraction, debate synthesis, KG, anchoring, Deep Mode) | ✅ Complete |
| 4 | Live Citation & Survey Mode | ✅ Complete |
| 5 | Security Hardening & Air-Gap | ✅ Complete |
| 5.5 | Local Model Optimization & Speed | ✅ Complete |
| 6 | UI, Polish & Deployment (Streamlit, GLiNER-PII, benchmarking) | ✅ Complete |
| 6.5 | Gap Closure (parallelization, compression, cache versioning, security fuzzer) | ✅ Complete |
| 7 | Vision Pipeline & Multi-Turn Synthesis | ✅ Complete |
| 8 | Publication-Scale Retrieval (EZProxy/Playwright) | ✅ Deprecated → Phase 9 |
| 9 | API-Based Literature Ingestion (Europe PMC, SPECTER2) | ✅ Complete |
| 10 | Autonomous Background Agent (orchestrator daemon) | ✅ Complete |
| 11 | Memory Cascade & Community Routing | ⬜ Designed — partial build |
| 12 | Skills & Experiential Memory | ⬜ Designed |
| 13 | Output Tools & Structured Writing | ⬜ Designed |

---

## Phase 1: Foundation

**Delivered**: AgentState TypedDict, Unicode-to-ASCII mapping, CitationManager abstract class + ZoteroAdapter stub, ChromaClient + BM25Index, HybridRetriever class. 7+ unit tests.

---

## Phase 2: PDF Ingestion & Hybrid Retrieval

**Delivered**: PDFParser wrapping Docling (text + table extraction, reference detection), HybridRetriever with RRF fusion, chunk metadata (source, chunk_type, chunk_index). Ingest pipeline with inline unicode scrub.

---

## Phase 3: LLM Agents & LangGraph Core

**Delivered**:
- Ollama setup (Qwen3.6 35B-A3B + Gemma 4 26B A4B)
- ExtractionAgent: category discovery + two-pass extraction (SciSpaCy NER + LLM structuring)
- SynthesisDrafter, SocraticCritic, Arbiter agent classes
- EvidenceAnchoringCheck: TF-IDF cosine similarity
- BaseGraphStorage interface + NetworkXJSONStorage
- GraphBuilder: entity-relationship construction with co-occurrence edges
- LangGraph Deep Mode: 17-node graph with conditional routing
- Interactive phase3_demo.py

**Confirmed behaviors**:
- Chunk summarization via Summarizer agent (cuts token usage ~5x)
- SciSpaCy NER (`en_core_sci_sm`): 155+ biomedical entities per query
- Anchoring: TF-IDF cosine similarity (threshold 0.35)
- Human-in-the-loop: two checkpoints via `interrupt_before`
- Debate chain: Drafter→Critic→Arbiter→ArbiterPass2 with conditional skip (NO_CRITIQUE)

**Key design decisions**:
- DeepSeek v4 Pro API for faster iteration (Ollama path preserved for air-gap)
- TF-IDF cosine (0.35) for anchoring
- 10 retrieved chunks per query via hybrid retriever
- Chunk summarization at query-time
- API key sanitization for unicode safety

**Tests**: 27 passing, 6 pre-existing failures (Ollama→DeepSeek migration not yet updated)

---

## Phase 4: Live Citation & Survey Mode

**Delivered**:
- Zotero API integration (real item creation, PDF attachment, CiteKey generation)
- Query decomposition agent (3-8 themed sub-queries)
- Thematic clustering (sentence-transformer embeddings with LLM fallback)
- Per-document parallel extraction (ThreadPoolExecutor)
- Per-theme deep synthesis (reuses Drafter→Critic→Arbiter debate chain)
- Cross-theme synthesis + gap analysis
- Multi-level query caching (7-day TTL)
- Human-in-the-loop gates

**Survey Mode Graph** (8 nodes):
```
survey_query_decompose → survey_retrieve → survey_thematic_cluster
  → survey_per_document_extract → survey_per_theme_synthesize
  → survey_cross_theme_synthesize → [HITL gate] → survey_scrub → END
```

**Key innovations**:
- Pre-extraction at ingest (entities cached to disk, zero LLM cost on repeat queries)
- Embedding-based clustering as primary (2s wall-clock, zero API cost)
- Single-paper themes skip all LLM calls
- Conditional Critic threshold at 0.35 (67% of Critic calls saved)
- Model tiering: deepseek-chat for per-theme, v4-pro for cross-theme
- tiktoken-based dynamic evidence truncation

**Performance**: Survey query 27→1-2min, LLM calls 18→12 per query, repeated query <1s.

**Tests**: 66 passing

---

## Phase 5: Security Hardening & Air-Gap

**Delivered**:
- Docker Compose with 3 services (orchestrator, public ollama, secure ollama)
- Two Ollama instances; secure instance air-gapped (`internal: true`)
- BoundaryScrubber: regex redaction at secure-public boundary
- LangGraph routing for `query_scope` field
- Security audit log
- Unified LLM provider (Ollama + DeepSeek via `get_chat_model_for_scope()`)

---

## Phase 5.5: Local Model Optimization & Speed

**Model selection**:
- Fast tier: `gemma4:e4b` (~9.6GB) — 2-3× faster than granite4.1:8b
- Reasoning tier: `qwen3.6:35b` (~23GB)
- Dual-model parallelism tested (gemma4:e4b + medgemma:4b)

**Optimizations**:
- Drafter prompt changed to dense claims format (1000-2200 → 250-600 chars)
- Second Critic→Arbiter pass removed (5 LLM calls → 3 per debated theme)
- Conditional critic threshold raised to 0.50 for local models
- `LLM_MAX_TOKENS=4096`, `LLM_TIMEOUT=900s`
- `max_workers=1` for sequential per-theme execution
- Diagnostic per-call timing logging

**Result**: Latency 12-39 min → ~5-8 min on M3 Max (36GB).

---

## Phase 6: UI, Polish & Deployment

**Delivered**:
- Streamlit UI (`app.py`): 5-tab interface (Query, Results, Benchmarks, History, Logs)
- Session history with re-run support
- Export formats: Markdown, Plain Text, JSON
- GLiNER-PII privacy model (570M params, 55+ entity types, Apache 2.0)
- Tier A programmatic benchmark (9 metrics)
- Correctness test suite (false-claim injection, OOC detection, Discussion-overlap)
- LLM-as-Judge evaluation (RAGAS faithfulness + gap quality)
- Hybrid anchoring (BM25 + ChromaDB in evidence check)

**Benchmarks** (6-paper corpus):
- Anchoring: 0.993 mean (99.2% grounded)
- Claim density: 118 claims across 22K chars
- Gap novelty: 80% (don't match Discussion sections)
- Grounded/inferential: 88% / 12%

**API vs Local comparison** (full survey graph):
- Anchor: 0.969 (cloud) vs 0.947 (local)
- Time: 212s (cloud) vs 524s (local)
- Cost: ~$0.50 (cloud) vs free (local)

**Novel approaches**:
- Calibrated LLM-as-Judge (TRUE/FALSE/GRAY pre-evaluation)
- Discussion-overlap gap novelty test
- Grounded vs inferential claim tagging
- Hybrid retrieval in anchoring (matches main pipeline)

**Tests**: 27→66→107

---

## Phase 6.5: Gap Closure

**Additions**:
- Per-theme parallelization (`PER_THEME_MAX_WORKERS=2`, ~23% faster)
- Prompt compression (21% reduction, 57K→45K chars)
- GLiNER-PII label restriction (FPR 58%→12%)
- 1:1 API vs local comparison refactored
- Security scrubber fuzzer (500+ lines, 9 tests)
- Multi-run variance testing
- Cache key versioning (`CACHE_VERSION = "v1"`)
- DOB YYYY-MM-DD pattern fix (33%→100% detection)

---

## Phase 7: Vision Pipeline & Multi-Turn Synthesis

**Delivered**:
- Figure extraction (Docling `generate_picture_images=True`)
- Smart figure filtering (Docling DocumentFigureClassifier, 80.9% keep rate)
- Vision model: gemma4:e4b via Ollama REST (zero model rotation overhead)
- Figure-to-text embedding (ChromaDB with `chunk_type="figure"`)
- Multi-turn section writing (8-node IMRaD LangGraph)
- Claim/citation ledger (SHA-256 dedup, @citation parsing, coverage reporting)

**Key lessons**:
- Vision model selection matters dramatically for biomedical figures
- `num_predict` breaks multimodal Ollama models (known bug)
- Docling's built-in figure classifier is production-grade
- Figure captions must resolve from document text items, not picture annotations
- SHA-256 claim dedup works across sections

**Novel approaches**: Vision model reuse (eliminates rotation), classification-first figure filtering, claims-as-content-addressed ledger, cross-paper claim provenance via compact identifiers.

**Tests**: 14 claim ledger, 7+23+13+7+9 vision tests

---

## Phase 8: Publication-Scale Retrieval (Deprecated)

Goal: Scale from 6 papers to 100s-1000s.

Deliverables built: Corpus acquisition script (PubMed + Semantic Scholar), EZProxy/Playwright PDF pipeline, hierarchical clustering module.

**Deprecated** — Phase 9 API-based ingestion (Europe PMC XML) replaced the EZProxy/Playwright pipeline. BM25 persistence retained. Neo4j adapter built but deferred. Hierarchical clusterer built but never wired into production.

---

## Phase 9: API-Based Literature Ingestion

**6 gaps closed**:
1. Retry logic (3-retry exponential backoff on EPMC)
2. Progress persistence (10-paper checkpoints via IngestProgress)
3. Ingestion wiring into ChromaDB + BM25
4. Coverage diagnostic (EPMC vs Semantic Scholar comparison)
5. Figure pipeline (XML `<graphic>` URLs → vision_ingest)
6. SPECTER2 caching (DOI-keyed JSON, skip re-fetch)

**Phase 10 foundations built**:
- PreExtractor + graph_storage wiring
- Gap resolver (parse gaps → search → ingest)
- Web search (DuckDuckGo discovery-only, never evidence)

**API strategy**: Europe PMC REST (10 req/s) → PMC OAI-PMH fallback (transparent), Semantic Scholar (1 req/s, 429 backoff), DuckDuckGo for web discovery.

**Tests**: 246 passing

---

## Phase 10: Autonomous Background Agent

**Core deliverables**:
- Orchestrator daemon (418 lines): web discovery → parallel EPMC → ingest → KG → handoff
- Subagents (54 lines): `run_parallel()` via ThreadPoolExecutor
- Handoff generator (147 lines): cycle-specific markdown files
- Scheduler (69 lines): interval timer with daemon thread, crash recovery

**Daemon pipeline**:
```
Web discovery → Query extraction → Parallel EPMC fetch (4 workers)
  → Batch ingest (single BM25 rebuild) → PreExtractor per-paper
  → KG save → Community detection → Handoff → State persist
```

**Beyond-spec enhancements**:
- Parallel EPMC wiring (~15s saved per cycle)
- Line-tagged extraction format (eliminates 70% JSON parse failure rate)
- Cycle handoff preservation (never overwrites human HANDOFF.md)
- State file + PID management
- Dry-run mode as first-class feature

**Live validation**: Single cycle ingested 13 new OA papers, grew KG 172→232 nodes, 520→1216 edges, BM25 21,112→22,085 docs.

**Gaps closed during build**: All gaps A-C (state load, handoff cleanup, daemon log management) fixed.
**Gaps remaining**: D (line-tagged untested with real Ollama), E (no long-running validation), F (coverage-gated routing), G (SPECTER2 unused).

**Novel approaches**: Line-tagged extraction, thread-parallel fetch + batch ingest, module-level workers for ThreadPoolExecutor, dry-run as first-class feature, cycle-specific handoff files.

**Tests**: 307 passing (61 new for Phase 10)

---

## Phases 11-13: Planned

| Phase | Description | Status |
|-------|-------------|--------|
| 11 | Memory cascade & community routing — Leiden/Louvain community detection, relevance router, progressive disclosure tiers | Designed, partial build |
| 12 | Skills & experiential memory — skill library (.md), trajectory logging (JSONL), agent learnings, skill evaluations | Designed |
| 13 | Output tools & structured writing — grant/proposal templates, evidence-anchored writer, citation integration | Designed |

---

## Lessons Learned (Cross-Phase)

1. **Thread-based timers need generous test margins** — internal multiplier conversions are hidden.
2. **BM25 rebuild contention is the real parallel bottleneck** — batch mutations after parallel reads.
3. **Mock import paths must target definition site** — not the call site.
4. **Know minimum chunk thresholds** when writing parser tests (PMCXMLParser.MIN_CHUNK_WORDS=20).
5. **State must be written AFTER counters are incremented** — persist after mutation.
6. **Don't trust LLMs to output valid JSON** — give them a format they can't break (line-tagged).
7. **Generated output should NEVER share a path with human documentation** — namespace machine output.
8. **Write-only state files are a code smell** — round-trip state (write + read).
9. **Single-retriever anchoring produces false low scores** — hybrid retrieval fixes BM25 keyword-frequency bias.
10. **LLM-as-Judge requires calibration** — pre-evaluate the judge before trusting its scores.
11. **Sentence-level TF-IDF inflates grounded rates** — chunk-level matching is more honest.
12. **Vision model selection matters dramatically** for biomedical figures.
13. **Baseline comparison validates pipeline architecture** — 134 claims vs 5 for naive RAG.
