# Phase 6 → Phase 7 Handoff — May 2026

## Quick start

```bash
python -m pytest phase5_benchmark.py test_correctness.py tests/ -v   # 102 passed, 0 failures
python phase5_verify.py --quick                                        # Phase 5 security verification
python phase4_demo.py                                                  # Survey Mode demo (~5-8 min on M3 Max 36 GB)
python phase5_benchmark.py                                             # Tier A programmatic benchmark (zero LLM calls)
python -m pytest test_correctness.py -v                                # Correctness tests (false claims, OOC, Discussion overlap)
python ragas_correctness.py --calibrate --sample 20                    # LLM-as-Judge with calibration (~1 min, requires DeepSeek API key)
streamlit run app.py                                                   # Web UI (localhost)
```

## Current project state

**Phase 6 core complete.** 102 tests passing (0 failures). The system runs
entirely on local Ollama models (gemma4:e4b for per‑theme tasks, qwen3.6:35b
for cross‑theme synthesis and gap analysis). Gap analysis model is now
separately configurable via `GAP_ANALYSIS_MODEL` (defaults to gemma4:e4b,
cutting ~5 min from runtime). Per‑query latency is ~5-8 min on M3 Max 36 GB.

The system has a rigorous multi‑layer correctness evaluation suite:
programmatic anchoring/coverage metrics, false‑claim injection, OOC detection,
Discussion‑overlap gap novelty testing, grounded/inferential claim tagging,
and LLM‑as‑Judge with TRUE/FALSE/GRAY calibration validated against DeepSeek
chat and v4‑pro.

## What was accomplished in Phase 6

### Benchmarking & correctness layer (the key deliverable)

| Component | File | Purpose |
|-----------|------|---------|
| Tier A benchmark (9 metrics) | `phase5_benchmark.py` | Anchoring, density, entity rate, debate invocation, cross‑theme coverage, redundancy, gap novelty (Discussion‑overlap), grounded/inferential, citation provenance. Zero LLM calls, pytest‑compatible. |
| Correctness tests (4 tests) | `test_correctness.py` | False‑claim injection (3 planted, all caught), negative controls (3 OOC queries, all below threshold), Discussion‑overlap (80% gap novelty), grounded/inferential tagging (chunk‑level, 88%/12%) |
| LLM‑as‑Judge with calibration | `ragas_correctness.py` | Faithfulness evaluation + gap quality (novelty, actionability). TRUE/FALSE/GRAY calibration validates judge before trusting scores. Supports DeepSeek chat + v4‑pro as judge (local synthesis, cloud evaluation). |
| Hybrid anchoring (BM25 + ChromaDB) | `src/anchoring/evidence_check.py` | `compute_anchoring_score()` now uses same hybrid retrieval as main pipeline. Module‑level singleton via `set_anchoring_chroma()`. Falls back to BM25‑only when ChromaDB unavailable (tests pass unchanged). |
| API comparison framework | `phase5_api_comparison.py` | Framework for DeepSeek v4‑pro vs local Ollama comparison. Script built, not yet executed with `--live`. |
| Dataset generation script | `generate_benchmark_dataset.py` | Automated 80–100 sample QA pair generator via LLMs + RAGAS. Script built, not yet executed. |

### UI & deployment

| Component | File | Purpose |
|-----------|------|---------|
| Streamlit UI | `app.py` | 5‑tab interface: Query, Results, Benchmarks, History, Logs. Export to MD/TXT/JSON. |
| Gap analysis model switch | `src/graph/survey_nodes.py` | `GAP_ANALYSIS_MODEL` env var (defaults to gemma4:e4b). Cuts gap analysis from ~368 s to ~40 s. Quality validated via RAGAS. |

### Privacy & security

| Component | File | Purpose |
|-----------|------|---------|
| GLiNER‑PII model | `src/security/gliner_privacy.py` | 570 M params, 55+ entity types, Apache 2.0. Drop‑in `PrivacyModel` implementation. Lazy‑loaded. `GLINER_PRIVACY_ENABLED=0` to disable. Integrated into `default_boundary_scrubber()`. |

### Bug fixes & improvements

| Fix | File | Purpose |
|-----|------|---------|
| Empty LLM cache guard | `src/cache/llm_cache.py` | `get()` now rejects empty/corrupt cached responses (caused JSONDecodeError in query decomposer) |
| Query decomposer cache guard | `src/agents/query_decomposer.py` | `cached.strip()` check before `json.loads()` |
| Sentence‑level inflation fixes | `test_correctness.py`, `phase5_benchmark.py` | Discussion‑overlap and grounded/inferential now use chunk‑level TF‑IDF (consistent with pipeline). Sentence‑level splitting inflated matches by 3‑5×. |
| RAGAS judge hybrid retrieval | `ragas_correctness.py` | `judge_claim_faithfulness()` now uses ChromaDB alongside BM25. Calibration TRUE claim was scoring 1/5 because BM25 returned irrelevant evidence. |
| Critical judge prompt | `ragas_correctness.py` | Reserves 5/5 for verbatim matches only, defaults to 3/5 when unsure, instructs judge not to be generous. |
| Inferential claim prompt | `ragas_correctness.py` | Separate rubric: "directionally supported" instead of "verbatim match." Scorecard labels distinguish grounded vs inferential. |

### Model tiering & configuration

| Config | Env var | Default | Notes |
|--------|---------|---------|-------|
| Fast tier | `OLLAMA_SMALL_MODEL` | gemma4:e4b | Per‑theme, decomposition, summarization, extraction |
| Reasoning tier | `OLLAMA_LARGE_MODEL` | qwen3.6:35b | Cross‑theme synthesis, critic, arbiter |
| Gap analysis | `GAP_ANALYSIS_MODEL` | gemma4:e4b | Separately configurable from cross‑theme. Set to qwen3.6:35b to revert. |
| Privacy AI | `GLINER_PRIVACY_ENABLED` | 1 | Set to 0 to disable for CI/fast startup |

### Test suite improvements

- +2 pytest tests (`test_benchmark_scores`, `test_false_claim_injection` → `test_grounded_vs_inferential`)
- Total: **102 passing, 0 failures** (was 97 at Phase 5 start)
- 6 pre‑existing `langchain_ollama` mock failures were fixed in Phase 5
- All sentence‑level inflation bugs fixed in Phase 6 correctness tests

## Lessons learned

### 1. BM25 keyword-frequency bias causes false anchoring low scores — hybrid retrieval fixes it

BM25 scores documents by aggregate term frequency. A complex claim like
"Obese mice exhibit elevated serum leptin and C‑reactive protein" matches
"obese," "mice," "serum" heavily in a bone‑formation chunk → high BM25 score
despite missing "leptin" entirely. The correct evidence chunk (containing
"leptin, resistin, and CRP were significantly higher") scores lower because
"leptin" appears fewer times. ChromaDB dense retrieval catches these cases
via semantic similarity.

Investigation across 118 claims: BM25 finds better evidence for 55.9%,
ChromaDB for 3.4%, tie for 40.7%. BM25 is the primary workhorse; ChromaDB
is the safety net. Both are essential.

**Architectural change**: `compute_anchoring_score()` now uses hybrid
retrieval. BM25‑only was producing false‑low scores that would trigger
unnecessary debate passes and incorrect HumanGate escalations.

### 2. LLM‑as‑Judge requires calibration — agreeableness bias is real

The original judge prompt produced 18/20 claims at 5/5 (gemma4:e4b).
Adding a critical prompt ("default to 3 if unsure, reserve 5 for verbatim")
dropped scores to 4.5‑4.7 with real score distribution (1‑2 at score 3 per
20 claims). TRUE/FALSE/GRAY calibration validated the judge actually
discriminates: all TRUE 5/5, all FALSE 1/5, GRAY 1.5‑2.5/5.

### 3. Sentence‑level TF‑IDF inflates match rates artificially

Splitting evidence into sentences (2609 sentences from 658 chunks) creates
granular units where almost any claim finds a "match." Grounded rate showed
99% at sentence level vs 83‑88% at chunk level. All correctness metrics now
use chunk‑level matching.

### 4. Gap novelty is real — 80‑90% of gaps don't match Discussion sections

The user's core concern was whether the gap analysis was producing genuine
gaps or just copying authors' future directions. Discussion‑overlap testing
(discovered categories searching gap questions against 64 Discussion chunks
from the last 10% of each paper) shows 80% novelty with qwen3.6:35b gap
analysis. gemma4:e4b gap analysis is shorter but comparably novel.

### 5. DeepSeek v4‑pro vs chat for judging: chat is sufficient for routine use

v4‑pro (509s, 22.6 min) is 10× slower than chat (88s, 1.5 min) for 20 claims.
Quality delta is small: faithfulness 4.5 vs 4.7, actionability 4.8 vs 4.0. For
routine benchmarking, use chat. Reserve v4‑pro for final validation or when
the benchmark explicitly requires stronger reasoning.

### 6. 6 papers limits synthesis depth — scale will increase inferential rate

88% of claims are grounded (traceable to a single evidence chunk). This is
expected with 6 papers on closely related topics. At 100+ papers with diverse
content, the inferential rate would naturally increase. Phase 8 scale test
will expose whether hybrid retrieval + multi‑tier caching can maintain quality.

## Key architectural decisions (DO NOT UNDO)

- **Unified LLM provider** — all agents call `get_chat_model()` from `src/llm/__init__.py`. Never go back to direct `ChatOpenAI` construction.
- **Default provider is Ollama** — `LLM_PROVIDER=ollama` in `.env`. DeepSeek is opt‑in with privacy warning and secure‑scope block.
- **Model tiering is env‑var configured** — `OLLAMA_SMALL_MODEL`, `OLLAMA_ALT_MODEL`, `OLLAMA_LARGE_MODEL`, `GAP_ANALYSIS_MODEL`. Agents resolve via `resolve_model()`.
- **Hybrid retrieval everywhere** — `compute_anchoring_score()` uses BM25 + ChromaDB fusion (set via `set_anchoring_chroma()`). Same pattern as main `HybridRetriever`.
- **Dense claim format** — Drafter produces one‑claim‑per‑line. Do NOT revert to prose paragraphs.
- **No compression step** — `_compress_syntheses_for_cross_theme` was deleted. Per‑theme dense claims feed directly to cross‑theme.
- **Single worker for per‑theme** — `max_workers=1` due to KV cache memory constraints on M3 Max.
- **Secure scope NEVER routes to cloud** — `get_chat_model_for_scope(scope="secure")` raises `RuntimeError` if `LLM_PROVIDER=deepseek`.
- **Debate chain is single‑pass** — Critic→Arbiter once, regression guard keeps draft if score worsens.
- **`CONDITIONAL_CRITIC_THRESHOLD = 0.50`** — calibrated for local models. Tune with benchmarks, not guesswork.
- **Gap analysis model is separately configurable** — `GAP_ANALYSIS_MODEL` env var, defaults to fast tier (gemma4:e4b).
- **All correctness tools use chunk‑level matching** — not sentence‑level. Sentence splitting inflates match rates 3‑5×.
- **LLM‑as‑Judge uses calibration before evaluation** — `--calibrate` flag validates the judge discriminates before trusting faithfulness scores.
- **All previous DO NOT UNDO from Phase 4 and Phase 5 still apply** (graph structures, clustering, TF‑IDF, single‑paper skip, KG interface, security modules, BoundaryScrubber, AuditLogger).

## What NOT to change

- The 17‑node Deep Mode graph structure
- The 8‑node Survey Mode graph structure
- The interrupt/resume pattern with `MemorySaver` checkpointer
- The SciSpaCy NER integration
- The debate chain (single‑pass Drafter→Critic→Arbiter)
- The embedding‑based thematic clustering (keep LLM fallback)
- The TF‑IDF extractive summarization (do NOT revert to LLM)
- The single‑paper debate skip logic
- The KG interface (`BaseGraphStorage` abstract class)
- The evidence anchoring check (`compute_anchoring_score`)
- The dense‑claim Drafter system prompt
- The unified LLM provider (`src/llm/__init__.py`)
- The security module (`src/security/`)
- The hybrid retrieval pattern in anchoring (BM25 + ChromaDB fusion)
- The calibration framework in `ragas_correctness.py`
- The chunk‑level matching in all correctness metrics

## Current known issues

1. **`OLLAMA_CONTEXT_LENGTH=32768` is hardcoded by Ollama** — cannot be overridden via LangChain's `ChatOpenAI`. Per‑model context control requires native Ollama API or Modelfile.

2. **Per‑theme synthesis text can be verbose** — gemma4:e4b still produces 20‑30 claims per theme despite dense‑claim prompt. Benchmark whether fewer, higher‑quality claims improve downstream synthesis.

3. **No baseline comparison exists** — the system is well‑tested and validated internally, but we haven't compared against a simpler alternative (naive RAG, single‑agent, or different architecture). Every SOTA paper includes at least one comparison point.

4. **6‑paper corpus limits synthesis depth** — 88% of claims are grounded. At scale (100+ papers), the inferential rate should increase. Phase 8 will expose whether the pipeline handles this.

5. **GRAY calibration claim "IL‑10 alone is sufficient to reverse obesity‑induced peri‑implant inflammation"** — classified as GRAY but v4‑pro scores it 1/5. The claim genuinely overstates the evidence. Consider reclassifying as FALSE in calibration.

6. **`ragas_correctness.py` calibration TRUE claim "Obese mice exhibit elevated serum leptin…"** — now fixed with hybrid retrieval (ChromaDB finds the correct chunk). Was scoring 1‑2/5 due to BM25 returning a bone‑formation chunk.

7. **API vs local comparison not yet executed** — `phase5_api_comparison.py` script exists but `--live` flag has not been run. Requires DeepSeek API credits.

8. **No multi‑run variance analysis** — we observed anchoring scores can vary between runs (0.818→0.922) due to decomposition variability. Should run the same query 3× and compute mean ± std.

## File map (new and changed in Phase 6)

```
NEW FILES:
phase5_benchmark.py                        # Tier A programmatic benchmark (9 metrics, pytest)
test_correctness.py                        # Correctness tests (4 tests: false‑claim, OOC, Discussion‑overlap, grounded/inferential)
ragas_correctness.py                       # LLM‑as‑Judge with calibration (faithfulness + gap quality)
phase5_api_comparison.py                   # API vs local comparison framework
generate_benchmark_dataset.py             # Automated 80–100 sample QA dataset generator
app.py                                     # Streamlit UI (localhost:8501)
src/security/gliner_privacy.py             # NVIDIA GLiNER‑PII privacy model implementation
investigate_bm25.py                        # BM25 vs ChromaDB contribution investigation

MODIFIED FILES (Phase 6 changes):
src/anchoring/evidence_check.py            # Hybrid retrieval (BM25 + ChromaDB) + module‑level singleton
src/cache/llm_cache.py                     # Empty cached response guard
src/agents/query_decomposer.py             # Empty cached response guard
src/graph/survey_nodes.py                  # GAP_ANALYSIS_MODEL env var
src/security/boundary_scrubber.py          # GLiNER‑PII integration, lazy env‑var reading
requirements.txt                           # Added streamlit, gliner, ragas
.env                                       # GAP_ANALYSIS_MODEL, GLINER_PRIVACY_ENABLED
phase3_demo.py                             # set_anchoring_chroma()
phase4_demo.py                             # set_anchoring_chroma()
phase5_benchmark.py                        # ChromaClient import, set_anchoring_chroma()
test_correctness.py                        # ChromaClient import, set_anchoring_chroma(), chunk‑level fixes
README.md                                  # Phase 6 completion, benchmarking results, lessons learned
HANDOFF.md                                 # This file — complete rewrite for Phase 6→7 handoff

PROJECT DATA (auto‑generated, not hand‑edited):
projects/default/project_graph.json        # Knowledge graph (NetworkX JSON)
projects/default/bm25_corpus.json          # BM25 sparse index
projects/default/chroma_data/              # ChromaDB persistent storage
projects/default/extractions/              # Pre‑extracted entities per paper
projects/default/query_cache/              # Multi‑level query cache (L1/L2/L3)
projects/default/cache/                    # LLM prompt cache (24h TTL)
projects/default/survey_result.json        # Latest cached survey synthesis
projects/default/correctness_scorecard.json# Latest RAGAS correctness scorecard
projects/default/benchmark_scorecard.json  # Latest Tier A benchmark scorecard
projects/default/content_hashes.json       # PDF deduplication hashes
logs/security_audit.log                    # Security event log (created at runtime)
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models (if not already pulled)
ollama pull gemma4:e4b
ollama pull qwen3.6:35b

# Run tests
python -m pytest phase5_benchmark.py test_correctness.py tests/ -v    # 102 passed, 0 failures

# Phase 5 security verification
python phase5_verify.py --quick

# Survey Mode demo (auto‑ingests new PDFs, pre‑extracts, caches)
python phase4_demo.py

# Tier A benchmark (zero LLM calls, reads cached survey_result.json)
python phase5_benchmark.py

# Correctness tests
python -m pytest test_correctness.py -v
python test_correctness.py                   # Verbose CLI report

# LLM‑as‑Judge (requires DEEPSEEK_API_KEY in .env for judge)
python ragas_correctness.py --calibrate --sample 20                  # ~1 min, deepseek‑chat judge
python ragas_correctness.py --calibrate --judge-model deepseek-v4-pro --sample 20  # ~20 min, v4‑pro judge
python ragas_correctness.py --judge-provider ollama --sample 10      # Local Ollama judge (slower)

# Streamlit UI
streamlit run app.py

# Clean everything for fresh start
rm -rf projects/default/cache projects/default/query_cache \
       projects/default/chroma_data projects/default/bm25_corpus.json \
       projects/default/extractions projects/default/embeddings
python phase4_demo.py
```

## Current model configuration

| Tier | Model | Size | Purpose |
|------|-------|------|---------|
| Fast (small) | `gemma4:e4b` | 9.6 GB | Per‑theme Drafter, query decomposition, extraction, summarization, gap analysis |
| Reasoning (large) | `qwen3.6:35b` | 23 GB | Cross‑theme synthesis, critique, arbitration |
| Alt (disabled) | — | — | medgemma:4b tested but too slow; future alternative needed |

Memory: fast tier (9.6 GB) unloads via `OLLAMA_KEEP_ALIVE=60s` before reasoning tier
(23 GB) loads. Peak ~28 GB, fits in 36 GB M3 Max.

## Performance (local Ollama, M3 Max 36 GB)

| Metric | Phase 5 (initial) | Phase 6 (current) |
|--------|-------------------|-------------------|
| Survey query latency | 5-8 min | ~5-8 min |
| Per‑theme synthesis | ~60-100 s (sequential, gemma4:e4b) | ~60-100 s (unchanged) |
| Gap analysis | ~368 s (qwen3.6:35b) | ~40 s (gemma4:e4b, configurable) |
| Cross‑theme synthesis | ~200 s (qwen3.6:35b) | ~200 s (unchanged) |
| LLM calls per query | ~8-10 | ~8-10 (unchanged) |
| Repeated query | < 1 s (cache) | < 1 s (cache) |
| Tests | 97 pass, 0 fail | 102 pass, 0 fail |

## Benchmarking status

### Benchmarks built and validated

| Benchmark | File | Key results (latest run) |
|-----------|------|--------------------------|
| Tier A (9 metrics) | `phase5_benchmark.py` | Anchor 0.993, 80% gap novelty, 88% grounded |
| Correctness (4 tests) | `test_correctness.py` | 3/3 false claims caught, 3/3 OOC low, 80% novelty |
| LLM‑Judge + calibration | `ragas_correctness.py` | CAL VALID, faith 4.5‑4.7, gaps 4.5‑4.8 |

### Benchmarks scripted but not executed

| Benchmark | File | Why deferred |
|-----------|------|-------------|
| API vs local comparison | `phase5_api_comparison.py --live` | Requires API credits, ~$1‑2 |
| 80‑100 sample dataset | `generate_benchmark_dataset.py` | Better value at Phase 8 scale |

### Benchmarks planned but not built

| Benchmark | Priority | Effort |
|-----------|----------|--------|
| Cache key versioning | High | ~20 lines |
| Security scrubber fuzzer | Medium | ~500 lines |
| Multi‑run variance analysis | Medium | Re‑run query 3× |
| Config comparison matrix | Low | Useful when evaluating alternatives |

## Future directions

### Immediate (remaining Phase 6, ~3-4 hours)

1. **Cache key versioning** — bump version constant when output format changes. Prevents stale cached results from misleading benchmarks. Trivial (~20 lines).
2. **API vs local comparison** — run `phase5_api_comparison.py --live` with 3 cached queries. Validate local Ollama produces comparable quality to DeepSeek API. ~$1‑2, ~10 min.
3. **Multi‑run variance** — run the same query 3×, compute mean ± std for anchoring scores. Confirms stability.
4. **Security scrubber fuzzer** — generate 1000 random PHI‑like strings, test GLiNER detection rate. Important for production readiness.

### Phase 7: Vision Pipeline & Multi‑Turn Synthesis

- Extract figures from PDFs via Docling's `generate_page_images=True`
- Load a lightweight multimodal model (LLaVA 7 B or Qwen‑VL, ~3‑5 GB)
- Model rotation: unload text model, load vision model per figure, swap back
- Figure‑to‑text embedding: embed descriptions alongside chunk text for cross‑modal retrieval
- Multi‑turn section writing: extend LangGraph with stateful section tracking across Introduction → Methods → Results → Discussion
- Claim/citation ledger: programmatic tracking prevents duplicate claims, flags ungrounded assertions

**Readiness**: Phase 7 is architecturally independent of remaining Phase 6 items.
You can start it in parallel. The core text pipeline is stable and well‑tested.

### Phase 8: Publication‑Scale Retrieval

- Hierarchical thematic clustering for 100‑1000s of papers
- Per‑theme top‑K retrieval (prevents O(n_papers) context scaling)
- Neo4j graph storage for >10K edges
- Corpus‑level claim extraction at ingest time
- Multi‑tier caching (L0: corpus claim index, L4: publication‑section output)
- Target: 30‑90 s per query on 1000 papers

**Deferred from Phase 6**: Neo4j adapter — NetworkX JSON handles current 6‑paper,
~500‑node graph. Only needed at scale.

## Prompt for next AI session

```
You are continuing work on a Federated RAG system for biomedical research.
Read the full README.md (especially §Phase 6 Status and §12.3 Benchmarking)
and this HANDOFF.md to understand the architecture and current state.

The project has 102 passing tests (0 failures). Phase 6 core is complete:
Streamlit UI, GLiNER‑PII privacy model, multi‑layer correctness benchmarking
(Tier A programmatic, false‑claim injection, OOC detection, Discussion‑overlap
gap novelty, LLM‑as‑Judge with calibration). The system runs on local Ollama
models (gemma4:e4b + qwen3.6:35b) at ~5‑8 min per query on M3 Max 36 GB.
Gap analysis now uses gemma4:e4b (configurable via GAP_ANALYSIS_MODEL).

The highest‑impact remaining items are:
  1. Cache key versioning (~20 lines, prevents stale benchmark results)
  2. API vs local comparison (run phase5_api_comparison.py --live, ~$1‑2, ~10 min)
  3. Security scrubber fuzzer (~500 lines)
  4. Multi‑run variance analysis (re‑run same query 3×)

Phase 7 (Vision Pipeline & Multi‑Turn Synthesis) can start in parallel — it's
architecturally independent of the remaining Phase 6 items.

Before making any changes:
  1. Run the test suite: python -m pytest phase5_benchmark.py test_correctness.py tests/ -v
  2. Run the benchmark: python phase5_benchmark.py
  3. Run correctness tests: python -m pytest test_correctness.py -v
  4. Read the key architectural decisions in HANDOFF.md §"Key architectural decisions"
  5. Read HANDOFF.md §"What NOT to change"

Do NOT:
  - Revert to verbose prose Drafter format (keep dense claims)
  - Re‑add the compression step we removed
  - Revert to DeepSeek API as default (Ollama is now default)
  - Change the 17‑node Deep Mode graph or 8‑node Survey Mode graph
  - Remove the unified LLM provider (src/llm/__init__.py)
  - Remove security modules (BoundaryScrubber, AuditLogger, PrivacyModel)
  - Change the debate chain structure (now single‑pass Critic→Arbiter)
  - Revert CONDITIONAL_CRITIC_THRESHOLD below 0.50 without benchmark data
  - Remove hybrid retrieval from anchoring (BM25 + ChromaDB fusion)
  - Remove the calibration framework from LLM‑as‑Judge
  - Build human‑scored factual accuracy benchmarks
  - Skip programmatic benchmarks before making quality‑impacting changes
  - Revert to sentence‑level TF‑IDF matching in correctness metrics (keep chunk‑level)
```