# Phase 5 → Phase 6 Handoff — May 2026

## Quick start

```bash
python -m pytest tests/ -v              # 97 passed, 0 failures
python phase5_verify.py                 # Phase 5 security feature verification
python phase5_verify.py --quick         # Skip Ollama health check
python phase4_demo.py                   # Survey Mode demo (local models, ~5-8 min/query)
```

## Current project state

**Phase 5 is complete.** 97 tests passing (zero failures — the 6 pre-existing
`langchain_ollama` mock failures from Phase 4 were fixed).  The system runs
entirely on local Ollama models (gemma4:e4b + qwen3.6:35b) with no cloud API
dependency.  Per-query latency is ~5-8 minutes on M3 Max 36GB

## What was accomplished in Phase 5

### Security hardening (per README §10)

| Component | File | Purpose |
|-----------|------|---------|
| BoundaryScrubber | `src/security/boundary_scrubber.py` | Regex redaction of 12 PHI/PII types at secure→public boundary; configurable via file |
| Audit Logger | `src/security/audit_log.py` | Thread-safe JSON-lines security event log (LLM calls, boundary crossings, scope routing, access patterns) |
| PrivacyModel interface | `src/security/privacy_model.py` | Abstract interface for AI-based PII detection (Phase 6: NVIDIA GLiNER-PII drop-in) |
| Scope routing | `src/graph/graph_builder.py`, `nodes.py`, `survey_nodes.py` | `query_scope` branching: `"both"` routes through `boundary_scrub` node |
| Penetration tests | `tests/security/test_security.py` | 25 tests: PHI redaction, prompt injection, audit event integrity, thread safety |
| Docker Compose | `docker-compose.yml`, `Dockerfile` | 3-service air-gap (orchestrator, ollama-public, ollama-secure with `internal:true`) |
| macOS Seatbelt | `sandbox/federated_rag.sb`, `sandbox/run_sandboxed.sh` | Native kernel sandbox as Docker alternative (lighter, no VM overhead) |

### LLM provider unification

All agents now route through `src/llm/__init__.py` via `get_chat_model()` instead
of directly constructing `ChatOpenAI`.  Supports two backends:

- `ollama` (DEFAULT) — local models, no cloud egress
- `deepseek` — cloud API (opt-in, logs privacy warning, blocks secure scope)

Key capabilities:
- `resolve_model()` maps tier keywords (small/chat, large/pro, alt) to configured models
- `get_chat_model_for_scope()` enforces secure scope never routes to cloud
- `LLM_TIMEOUT`, `LLM_MAX_TOKENS`, `LLM_NUM_CTX` env vars for configuration

### Local model optimization (Phase 5.5)

| Optimization | File | Impact |
|-------------|------|--------|
| Dense claim output format | `synthesis_drafter.py` | Drafter produces 1-claim-per-line instead of prose paragraphs — 3-4× shorter, 2-3× faster generation. Full narrative preserved in cross-theme synthesis. |
| Compression step removed | `survey_nodes.py` | Deleted `_compress_syntheses_for_cross_theme` (LLM compression + 600-char truncation). Per-theme dense claims feed directly to cross-theme. |
| Debate chain simplified | `survey_nodes.py` | Removed second Critic→Arbiter pass (was 5 calls per debated theme, now 3). Regression guard still keeps draft if debate worsens anchoring. |
| `CONDITIONAL_CRITIC_THRESHOLD` raised | `survey_nodes.py` | 0.35 → 0.50 for local models (fewer themes trigger expensive debate chain) |
| `LLM_MAX_TOKENS` reduced | `.env`, `llm/__init__.py` | Default 8192 → 4096 via env var; agents use env default (cuts generation time ~30-50%) |
| `LLM_TIMEOUT` configured | `.env`, demos | 900s default; env var `LLM_TIMEOUT` respected by all agents |
| Hardcoded theme count removed | `query_decomposer.py` | "3-8 themes" → "ALL semantically distinct themes" — scale-independent |
| Timing diagnostics | `synthesis_drafter.py`, `survey_nodes.py` | Per-call start/end logging with prompt size, output size, and latency |

### Model selection journey

| Iteration | Fast tier | Reasoning tier | Result |
|-----------|-----------|----------------|--------|
| 1 (Phase 4) | deepseek-chat (API) | deepseek-v4-pro (API) | 1-2 min (cloud, not local) |
| 2 | granite4.1:8b | qwen3.6:35b | 12-39 min (local, KV cache explosion + parallelism queue) |
| 3 | gemma4:e4b | qwen3.6:35b | 8-12 min (2× faster 4B active params vs 8B) |
| 4 | gemma4:e4b + medgemma:4b (dual) | qwen3.6:35b | 5-8 min (dual-model, then medgemma proved too slow — 10+ min/theme) |
| 5 | gemma4:e4b solo | qwen3.6:35b | ~5-8 min (current, reliable, but would like to improve latency (while retaining information accuracy/reliability) if possible) |

### Test suite improvements

- 6 pre-existing `langchain_ollama.ChatOllama` mock failures → fixed by updating mocks to `langchain_openai.ChatOpenAI`
- +25 security tests (boundary scrubber, audit logger, prompt injection)
- Total: 97 passing, 0 failures

## Lessons learned

### 1. Local models are 4-10× slower than cloud API — this is hardware physics, not a code problem
The Phase 4 "1-2 minute" benchmark was against DeepSeek's A100/H100 clusters.  Local
models on M3 Max generate tokens at 10-25 tok/s vs 50-100+ tok/s on cloud GPUs.
You cannot close this gap on consumer silicon.  The goal shifted from "match cloud
speed" to "reliable quality at acceptable local latency." Still, effort should be made to have maximum quality and minimum latency, as possible on the local system.

### 2. Ollama parallelism (`OLLAMA_NUM_PARALLEL`) + `max_workers` must respect KV cache memory
Running 4 concurrent requests with 32K context each caused KV cache exhaustion
(~30GB for the model + ~32GB for 4× KV caches).  The fix: `max_workers=1`
(sequential) or `max_workers=2` with `OLLAMA_NUM_PARALLEL` tuned.  Different models
in parallel (gemma4:e4b + medgemma:4b) worked because they have different weights,
enabling true GPU parallelism without memory multiplication.

### 3. `OLLAMA_CONTEXT_LENGTH` is set internally by Ollama, not from `.zshrc`
Spent significant time debugging why the env var showed 32768 despite `launchctl
setenv`.  Turns out Ollama's Electron app sets this as an internal default based
on hardware detection.  The `.zshrc` export was irrelevant.  Context length cannot
be overridden via LangChain's `model_kwargs` (ChatOpenAI doesn't forward Ollama-
specific params).  Use the native Ollama API or Modelfile for per-model context control.

### 4. Dense claim format preserves quality while dramatically reducing token count
The Drafter was producing verbose prose paragraphs with repeated preamble across
themes.  Switching to "one claim per line, no preamble" reduced output from
1000-2200 chars to 250-600 chars per theme while preserving all citations and
anchoring scores (0.88-0.95).  The LLM processes dense text efficiently — humans
need narrative flow; LLMs need information density.

### 5. medgemma:4b is not viable for per-theme synthesis on M3 Max
Despite being only 3.3GB on disk, medgemma:4b took 10+ minutes per theme due to
32K context overhead and older Gemma 3 architecture.  gemma4:e4b (9.6GB, ~4B
active experts) completed the same task in 60-70s.  Model architecture and Metal
optimization matter more than raw parameter count.

### 6. Benchmarking needs to be programmatic for a single developer
The README's planned 20-30 human-annotated QA pair benchmark is impractical for
one person.  The redesigned approach uses: (a) anchoring scores as a free quality
proxy, (b) per-phase latency breakdown from diagnostic logs, (c) claim density
and entity appearance rate as information-efficiency metrics, (d) golden query
tripwires (3 queries you know well, 6 min/week), (e) optionally RAGAS for
LLM-as-judge evaluation.  All automated except golden queries.

### 7. macOS `sandbox-exec` does not reliably block TCP sockets
Seatbelt profiles provide filesystem and process restrictions, but TCP socket
connections from user-space processes bypass the sandbox's network rules on macOS.
Docker (with its Linux VM and separate network namespaces) is required for
guaranteed network isolation.

## Key architectural decisions (DO NOT UNDO)

- **LLM provider is unified** — all agents call `get_chat_model()` from `src/llm/__init__.py`.  Never go back to direct `ChatOpenAI` construction.
- **Default provider is Ollama** — `LLM_PROVIDER=ollama` in `.env`.  DeepSeek is opt-in with privacy warning and secure-scope block.
- **Model tiering is env-var configured** — `OLLAMA_SMALL_MODEL`, `OLLAMA_ALT_MODEL`, `OLLAMA_LARGE_MODEL`.  Agents resolve via `resolve_model()` using tier keywords (small, alt, large).
- **Dense claim format** — the Drafter system prompt produces one-claim-per-line.  Do not revert to prose paragraphs.
- **No compression step** — the `_compress_syntheses_for_cross_theme` function was deleted.  Per-theme dense claims feed directly to cross-theme.  Do not re-add an intermediate compression step.
- **Single worker for per-theme** — `max_workers=1` due to KV cache memory constraints on M3 Max.  Can be increased if hardware changes.
- **Secure scope NEVER routes to cloud** — `get_chat_model_for_scope(scope="secure")` raises `RuntimeError` if `LLM_PROVIDER=deepseek`.
- **BoundaryScrubber runs on `query_scope="both"`** — both Deep and Survey graphs have scope-based conditional routing.
- **Debate chain is single-pass** — Critic→Arbiter once, regression guard keeps draft if score worsens.  No second pass.
- **`CONDITIONAL_CRITIC_THRESHOLD = 0.50`** — calibrated for local models.  Tune with benchmarks, not guesswork.
- **All previous "DO NOT UNDO" items from Phase 4 still apply** (graph structures, clustering, TF-IDF, single-paper skip, KG interface, anchoring check).

## What NOT to change

- The 17-node Deep Mode graph structure
- The 8-node Survey Mode graph structure
- The interrupt/resume pattern with `MemorySaver` checkpointer
- The SciSpaCy NER integration
- The debate chain internals (Drafter→Critic→Arbiter flow, now single-pass)
- The embedding-based clustering (keep LLM fallback)
- The TF-IDF extractive summarization (do not revert to LLM)
- The single-paper debate skip logic
- The KG interface (`BaseGraphStorage` abstract class)
- The evidence anchoring check (`compute_anchoring_score`)
- The dense-claim Drafter system prompt
- The unified LLM provider (`src/llm/__init__.py`)
- The security module (`src/security/`)

## Current known issues

1. **Gap analysis takes 368s on qwen3.6:35b** — the single biggest remaining time sink (49% of total runtime).  Switching to gemma4:e4b for gap analysis would cut this to ~40s but needs quality benchmarking first.

2. **`OLLAMA_CONTEXT_LENGTH=32768` is hardcoded by Ollama** — cannot be overridden via LangChain's `ChatOpenAI` (doesn't forward `num_ctx`).  Per-model context control requires native Ollama API or Modelfile approach.  Current workaround: live with 32K context, which is fine for quality at the cost of speed.

3. **gemma4:e4b solo (no dual-model)** — medgemma:4b was too slow for dual-model parallelism.  Finding a viable second fast-tier model for true GPU parallelism would cut per-theme time roughly in half. Candidates to evaluate: `gemma:latest` (5GB), `granite4.1:3b` (2.1GB), or a second gemma4:e4b variant.

4. **No formal benchmark suite exists** — programmatic tier is designed but not built.  Without it, optimization decisions are based on anchoring scores and latency alone — no objective quality measurement for gap analysis, citation accuracy, or cross-theme coverage.

5. **Per-theme synthesis text is still verbose** — the dense claim format produces 3400-5864 chars for some themes (due to the model generating 20-30 claims).  The prompt says "one claim per line, be concise" but gemma4:e4b still produces many claims.  Need to benchmark whether fewer, higher-quality claims improve downstream synthesis.

6. **32 hardcoded numeric limits identified** — a comprehensive inventory was compiled across 9 files (thresholds, truncation chars, top-N caps, worker limits).  Most have quality-driven alternatives.  Only 4 were fixed (theme count, critic threshold, max_tokens, debate pass count).  The remaining 28 are deferred to Phase 6+.

## File map (new and changed in Phase 5)

```
NEW FILES:
src/llm/__init__.py                  # Unified LLM provider (Ollama + DeepSeek routing)
src/security/                        # Security hardening module
├── __init__.py
├── audit_log.py                     # JSON-lines security event logging
├── boundary_scrubber.py             # Regex PHI/PII redaction + PrivacyModel interface
└── privacy_model.py                 # Abstract interface for AI PII detection (Phase 6)
config/scrub_patterns.txt            # Custom regex patterns for boundary scrubber
sandbox/
├── federated_rag.sb                 # macOS Seatbelt sandbox profile
└── run_sandboxed.sh                 # Launcher script with --no-sandbox fallback
.env.example                         # Environment template with documentation
phase5_verify.py                     # 8-section security verification demo
Dockerfile                           # Application container
docker-compose.yml                   # 3-service air-gap deployment
tests/security/test_security.py      # 25 penetration/security tests (all passing)

MODIFIED FILES (Phase 5 changes):
src/agents/arbiter.py                # Uses get_chat_model() instead of direct ChatOpenAI
src/agents/extraction_agent.py       # Same
src/agents/query_decomposer.py       # Same + hardcoded "3-8 themes" removed
src/agents/socratic_critic.py        # Uses get_chat_model()
src/agents/summarizer.py             # Uses get_chat_model()
src/agents/synthesis_drafter.py      # Dense claim system prompt + timing diagnostics
src/agents/thematic_clusterer.py     # Uses get_chat_model()
src/graph/graph_builder.py           # Scope routing (boundary_scrub node, conditional edges)
src/graph/nodes.py                   # Audit logging, boundary scrubbing, scope-aware scrub
src/graph/survey_nodes.py            # Dual-model parallelism, debate simplification,
                                     #   dense claims for cross-theme, timing diagnostics,
                                     #   removed compression step
src/state.py                         # (unchanged but query_scope was already present)
tests/test_synthesis_agents.py       # Fixed langchain_ollama mocks → ChatOpenAI
tests/test_survey_graph.py           # Updated threshold test (0.35 → 0.50)
.env                                # Two-tier model config, timeout settings
phase3_demo.py                       # LLM_TIMEOUT export
phase4_demo.py                       # LLM_TIMEOUT export
README.md                            # Phase 5 completion, Phase 5.5, Phase 7-8 plans

PROJECT DATA (auto-generated, not hand-edited):
projects/default/project_graph.json  # Knowledge graph (NetworkX JSON)
projects/default/bm25_corpus.json    # BM25 sparse index
projects/default/chroma_data/        # ChromaDB persistent storage
projects/default/extractions/        # Pre-extracted entities per paper
projects/default/query_cache/        # Multi-level query cache (L1/L2/L3)
projects/default/cache/              # LLM prompt cache (24h TTL)
projects/default/content_hashes.json # PDF deduplication hashes
logs/security_audit.log              # Security event log (created at runtime)
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models (if not already pulled)
ollama pull gemma4:e4b
ollama pull qwen3.6:35b

# Run tests
python -m pytest tests/ -v              # 97 passed, 0 failures

# Phase 5 security verification
python phase5_verify.py                 # Full (includes Ollama health check)
python phase5_verify.py --quick         # Skip Ollama check

# Survey Mode demo (auto-ingests new PDFs, pre-extracts, caches)
python phase4_demo.py

# Deep Mode demo (Phase 3, still works)
python phase3_demo.py

# macOS sandbox launcher
./sandbox/run_sandboxed.sh              # Survey mode under Seatbelt
./sandbox/run_sandboxed.sh verify       # Phase 5 verification under sandbox
./sandbox/run_sandboxed.sh --no-sandbox # Unsandboxed dev mode

# Clean everything for fresh start
rm -rf projects/default/cache projects/default/query_cache \
       projects/default/chroma_data projects/default/bm25_corpus.json \
       projects/default/extractions projects/default/embeddings
python phase4_demo.py
```

## Current model configuration

| Tier | Model | Size | Purpose |
|------|-------|------|---------|
| Fast (small) | `gemma4:e4b` | 9.6 GB | Per-theme Drafter, query decomposition, extraction, summarization |
| Reasoning (large) | `qwen3.6:35b` | 23 GB | Cross-theme synthesis, gap analysis, critique, arbitration |
| Alt (disabled) | — | — | medgemma:4b tested but too slow; future alternative needed |

Memory: fast tier (9.6GB) unloads via `OLLAMA_KEEP_ALIVE=60s` before reasoning tier
(23GB) loads.  Peak ~28GB, fits in 36GB M3 Max.

## Performance (local Ollama, M3 Max 36GB)

| Metric | Phase 4 (API) | Phase 5 (local) |
|--------|---------------|-----------------|
| Survey query latency | 1-2 min | 5-8 min |
| Per-theme synthesis | ~9s (parallel, chat API) | ~60-100s (sequential, gemma4:e4b) |
| Cross-theme + gap | ~47s (parallel v4-pro) | ~200-370s (qwen3.6:35b) |
| LLM calls per query | ~12 | ~8-10 |
| Repeated query | < 1s (cache) | < 1s (cache) |
| Tests | 66 pass, 6 fail | 97 pass, 0 fail |

Note: The 5-8 min latency is NOT an architecture problem — it's the physics of
running 35B + 9GB models on consumer silicon at 32K context with token-by-token
generation.  Cloud API (A100/H100 clusters) cannot be matched on a local machine.

## Benchmarking strategy (Phase 6 priority)

### Philosophy

The goal is not to prove the system is correct.  The goal is to know if your
next commit made things worse.  All programmatic metrics measure direction, not
absolute quality.  If you run the same queries before and after a change and
anchoring drops 0.15, you caught a regression — no human needed.

### What to measure (all automated, zero human effort)

| Metric | Why it matters | How to compute |
|--------|---------------|----------------|
| Anchoring score distribution (mean, min, std, % below threshold) | Quality proxy — catches hallucination regressions | Aggregate `anchoring_score` across cached queries |
| Per-phase latency breakdown | Shows exactly where time goes (gap analysis = 49% of current runtime) | Parse diagnostic timing logs; aggregate by phase |
| Claim density (claims per char) | Measures information efficiency of dense format | `decompose_claims()` count / output chars |
| Entity appearance rate | Verifies pre-extracted knowledge surfaces in output | grep pre-extracted entity names from `projects/default/extractions/` in synthesis |
| Debate invocation rate (% themes scoring < threshold) | Rising rate = model quality degrading | Count themes with score < threshold / total themes |
| Cross-theme coverage ratio | Does cross-theme drop per-theme findings? | Unique claims in cross-theme / total unique claims across all per-themes |
| Redundancy score | Are decomposer themes overlapping? | Claims appearing in ≥2 themes / total unique claims |
| Gap analysis specificity | Are gap questions vague or specific? (proxy: avg words per question) | Split gap output by question marks, count avg tokens per question |
| Citation provenance (spot-check) | Do inline citations point to the correct paper? | Pick 5 random claims, verify cited paper actually contains the claim (5 min/check, semi-automated) |

### Configuration comparison matrix

Run the same 3–5 cached queries through config A vs config B, diff all metrics side-by-side:

| Parameter to vary | Values to test | Primary metrics to compare |
|-------------------|---------------|---------------------------|
| Gap analysis model | gemma4:e4b vs qwen3.6:35b | Gap specificity, latency, cross-theme coverage |
| `LLM_MAX_TOKENS` | 2048 vs 4096 vs 8192 | Per-theme claim count, anchoring, latency |
| `CONDITIONAL_CRITIC_THRESHOLD` | 0.35 vs 0.50 vs 0.65 | Debate invocation rate, final anchoring scores |
| Fast tier model | gemma4:e4b vs granite4.1:8b | Per-theme anchoring, claim density, latency |
| Dense claim vs verbose prose | Compare cached verbose run vs dense-claim run | Anchoring, latency, claim count per theme |
| **API vs local comparison** | DeepSeek v4-pro API vs local Ollama | Factual correctness, claim accuracy, anchoring scores, latency — validates whether local models produce comparable synthesis quality to cloud API |

### Security benchmark

| Metric | How to test |
|--------|------------|
| Boundary scrubber false positive rate | Inject synthetic biomedical text with 0 PHI, verify no redactions |
| Boundary scrubber false negative rate | Inject synthetic clinical note with known PHI, verify all caught |
 | Audit log completeness | Verify every LLM call, boundary crossing, and scope routing decision produces a log entry |

### Existing benchmarks — what's available (May 2026)

**No multi-document biomedical engineering synthesis benchmarks exist.**  HuggingFace
dataset registry returns 0 results for biomaterial/immune-response/implant synthesis.
The closest datasets test very different capabilities:

| Dataset | Tests | Why not usable |
|---------|-------|----------------|
| PubMedQA | Single-abstract yes/no/maybe | No multi-paper synthesis, no evidence anchoring |
| BioASQ | Factoid retrieval from PubMed snippets | No cross-paper reasoning, no gap analysis |
| MedQA | USMLE multiple-choice | Clinical diagnosis, not research synthesis |
| MedBench | Clinical data from EHRs | Structured patient data, not literature |
| Biomedical Q&A (Shushant) | General biomedical QA | No multi-document reasoning, no citation tracing |

None test the core value of this system: reading N papers, synthesizing claims across them,
anchoring each claim to specific evidence chunks, identifying cross-paper contradictions,
and proposing research gaps.

### Automated benchmark creation — feasible for 1 person + AI

LLMs can generate evaluation data that is *better than human-written references* for
summarization tasks (arXiv:2309.09558 — LLM-generated summaries preferred over human
ones for factual consistency).  Combined with RAGAS (arXiv:2309.15217) for reference-free
evaluation, a 80–100 sample benchmark can be built with ~2 hours of human effort:

| Step | Automated by | How |
|------|-------------|-----|
| Generate questions | gemma4:e4b | Feed paper chunks → "generate 5 research questions this paper can answer" |
| Generate "gold" answers | qwen3.6:35b | Feed full paper → "produce a complete synthesis answering this question" |
| Extract key entities | Pre-extraction | Existing entity data per paper (already built) |
| Validate against source | RAGAS faithfulness | LLM-as-judge checks claim-by-claim against evidence (arXiv:2309.15217) |
| Filter hallucinated claims | Anchoring + self-consistency | Run 3×, keep only claims appearing in ≥2 runs |
| Human spot-check | You | 5 random claims per 20 questions ≈ 30–45 min |

Result: 80–100 high-confidence benchmark samples with quasi-ground-truth answers.
This is the same methodology behind published evaluation datasets — but adapted
to multi-document synthesis, which no existing benchmark covers.

**Scale note:** Benchmarking across 10s–100s of papers should come AFTER search
capabilities (Phase 8) so papers can be auto-discovered rather than manually
curated.  Long-running benchmarks (overnight/multi-day) need a progress indicator
(log lines showing questions completed / total, ETA) so stalls are visible.

### Holes in current benchmarking approach

These failure modes are NOT caught by the current metrics.  Each needs a specific test:

| Hole | Risk | Fix |
|------|------|-----|
| No negative control queries | System may confidently hallucinate on topics the corpus can't answer | Add 2 "out-of-corpus" queries. System should return "insufficient evidence" or skip themes entirely |
| No known-false claim injection | Critic might miss subtle fabrications that pass anchoring | Plant 1-2 false assertions per cached synthesis, verify anchoring/critic flag them |
| No multilingual/messy input test | Real PDFs have OCR errors, Unicode, special chars | Add 1 noisy PDF test case with known OCR errors, measure synthesis degradation |
| No scale test | Can't predict behavior at 50 or 500 papers | Simulate by duplicating chunk entries 10× in a test corpus, measure retrieval + synthesis latency scaling |
| Scrubber false negatives untested | Only positive cases tested (PHI detected); silent passes on new PHI formats untested | Build a fuzzer: generate 1000 random PHI-like strings, measure detection rate |
| Cache staleness | Old cached answers from prior code version may mislead benchmarks | Add cache key versioning — bump version constant when output format changes |
| Single-run variance | Stochastic outputs may cause false positives/negatives in benchmark | Run each query 3×, report mean ± std for all metrics |

### What 1 person + AI can accomplish (May 2026 perspective)

The entire Phase 5 — LLM provider unification, security module, sandbox, 25 tests,
model tiering, dense claim optimization, diagnostic logging — was built by 1 person
+ AI in ~12 hours of interaction.  This is roughly 3,000+ lines of working, tested
code across 17 new files and 13 modified files.  A year ago, this would have been
a 2-person, 2-week sprint.

For benchmarking, this means:

**Feasible (Phase 6, ~1 week elapsed with AI):**
- RAGAS integration with programmatic benchmark runner (`phase5_benchmark.py`)
- 80–100 sample automated dataset (LLM writes questions, answers, validates)
- Configuration comparison matrix (diff tool runs all configs, outputs heatmap)
- Negative control queries + false claim injection tests
- Scrubber fuzzer for false negative detection (~500 lines)
- Cache key versioning (~20 lines)

**Stretch (Phase 7, ~2 weeks):**
- Scale simulation with chunk duplication for 100+ paper behavior prediction
- Progress indicators for overnight multi-query benchmarks
- Publishable benchmark report with generated dataset (if validation shows strong
  agreement with your spot-checks)

### What NOT to benchmark (waste of time for single developer)

- Retrieval precision — threshold-based retrieval is self-calibrating (already validated in Phase 4)
- Human-scored factual accuracy at scale — not needed when RAGAS + self-consistency + anchoring provide automated alternatives
- Human-scored completeness rubrics — same; automated entity appearance rate + cross-theme coverage ratio replace this
- Building a benchmark from scratch manually — LLMs can generate 80–100 sample QAs; invest the 2 hours in spot-checking, not in writing every question
- NOTE: this does NOT mean eliminate human checks entirely.  The golden query tripwire (3 queries, 6 min/week) still catches catastrophic regressions.  And the initial 30–45 min spot-check of the automated dataset is essential for validation.

### Action items for Phase 6

1. **Build `phase5_benchmark.py`** — Tier A programmatic benchmark.  Runs on 3–5 cached queries (no live LLM calls).  Outputs a scorecard with pass/warn/fail thresholds.  Runs with `pytest` to catch regressions automatically.  Include progress indicator (X/Y queries complete, elapsed/ETA).

2. **Gap analysis model comparison** — run the benchmark with gemma4:e4b vs qwen3.6:35b for gap analysis.  Measure anchoring, gap specificity, and latency diff.  Switch to gemma4:e4b if quality is within acceptable range.

3. **API vs local comparison** — run the same 3–5 queries against both DeepSeek v4-pro API and local Ollama.  Compare: anchoring scores, claim count, entity appearance rate, latency.  Determine the quality gap (if any) between cloud and local models.

4. **Automated 80–100 sample dataset** — use LLMs to generate benchmark questions, gold answers, key entities, and RAGAS validation.  2 hours human spot-check.  This is the foundation for all future benchmarking.

5. **Negative control queries** — add 2 "out-of-corpus" queries.  System should produce "insufficient evidence" or empty themes.  Verifies hallucination detection.

6. **False claim injection test** — plant 1-2 false assertions in cached syntheses.  Verify critic/anchoring flags them.  Tests the debate chain's error detection.

7. **Security benchmark** — build the false positive/negative rate tests for BoundaryScrubber.  Add scrubber fuzzer (1000 random PHI-like strings).

8. **Cache key versioning** — bump version constant when output format changes.  Prevents stale cached syntheses from misleading benchmarks.

## Future directions (next session priorities)

### Immediate (Phase 6)

1. **Gap analysis model switch** — run gap analysis on gemma4:e4b instead of
   qwen3.6:35b.  Cut 328s from runtime (368s → 40s).  Benchmark quality impact
   using the configuration comparison matrix above.

2. **Build Tier A programmatic benchmark** — produce `phase5_benchmark.py` with
   automated metrics as specified in the benchmarking strategy above.  Must run
   with `pytest` and catch regressions.  Include progress indicators for
   long-running multi-query benchmarks (X/Y complete, elapsed/ETA).

3. **Automated dataset generation** — use LLMs to generate 80–100 benchmark QA
   pairs from your corpus.  2 hours human spot-check.  Foundation for all
   future evaluation.  Includes negative controls, false claim injection, and
   RAGAS validation.

4. **API vs local comparison** — run same queries through DeepSeek v4-pro API and
   local Ollama.  Diff anchoring, claims, latency.  Publish results as Phase 5
   sign‑off validation.

4. **NVIDIA GLiNER-PII integration** — drop-in implementation of the
   `PrivacyModel` interface from `src/security/privacy_model.py`.  570M params,
   55+ entity types, ~1GB at FP16.  `pip install gliner`.

5. **UI & polish** (per README Phase 6): Streamlit/Gradio, session history,
   export formats, Neo4j adapter.

### Near-term (Phase 7)

6. **Vision pipeline** — extract and analyze figures from PDFs.  Load a small
   multimodal model (LLaVA 7B, Qwen-VL, or Granite vision variant).  Memory:
   unload text model, load vision model per figure.

7. **Multi-turn section writing** — extend LangGraph with stateful section
   tracking across Introduction → Methods → Results → Discussion sections.

### Scale (Phase 8)

8. **Publication-scale retrieval** — hierarchical clustering for 100-1000s of
   papers.  Per-theme top-K retrieval instead of loading all evidence.  Neo4j
   graph storage for >10K edges.

## Prompt for next AI session

```
You are continuing work on a Federated RAG system for biomedical research.
Read the full README.md and HANDOFF.md (this file) to understand the architecture
and current state.

The project has 97 passing tests (0 failures). Phase 5 (Security Hardening &
Air-Gap) is complete. The system runs on local Ollama models (gemma4:e4b +
qwen3.6:35b) at ~5-8 minutes per query on M3 Max 36GB.

Your priority is Phase 6: UI, Polish & Deployment.
See README §Phase 6 for the deliverables.  The highest-impact immediate items are:
  1. Build Tier A programmatic benchmark (phase5_benchmark.py) per §Benchmarking strategy
  2. Generate automated 80–100 sample benchmark dataset using LLMs + RAGAS
  3. Switch gap analysis to gemma4:e4b + benchmark quality impact
  4. API vs local comparison (DeepSeek v4-pro vs local Ollama on 3-5 queries)
  5. Integrate NVIDIA GLiNER-PII privacy model
  6. Build Streamlit/Gradio UI

Before making any changes:
  1. Run the test suite: python -m pytest tests/ -v
  2. Run the Phase 5 verify: python phase5_verify.py
  3. Try the demo: python phase4_demo.py
  4. Read the key architecture decisions in HANDOFF.md §"Key architectural decisions"
  5. Read the README §Phase 6, §12.3 (benchmarks), and §Phase 7-8 (future)

Do NOT:
  - Revert to verbose prose Drafter format (keep dense claims)
  - Re-add the compression step we removed
  - Revert to DeepSeek API as default (Ollama is now default)
  - Change the 17-node Deep Mode graph or 8-node Survey Mode graph
  - Remove the unified LLM provider (src/llm/__init__.py)
  - Remove security modules (BoundaryScrubber, AuditLogger, PrivacyModel interface)
  - Change the debate chain structure (now single-pass Critic→Arbiter)
  - Revert CONDITIONAL_CRITIC_THRESHOLD below 0.50 without benchmark data
  - Build human-scored factual accuracy benchmarks — impractical for 1 person
  - Skip programmatic benchmarks before making quality-impacting changes — measure first, change second
```
