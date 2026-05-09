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
dependency.  Per-query latency is ~5-8 minutes on M3 Max 36GB — limited by
local model generation speed, not architecture.

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
| 5 | gemma4:e4b solo | qwen3.6:35b | ~5-8 min (current, reliable) |

### Test suite improvements

- 6 pre-existing `langchain_ollama.ChatOllama` mock failures → fixed by updating mocks to `langchain_openai.ChatOpenAI`
- +25 security tests (boundary scrubber, audit logger, prompt injection)
- Total: 97 passing, 0 failures

## Lessons learned

### 1. Local models are 4-10× slower than cloud API — this is hardware physics, not a code problem
The Phase 4 "1-2 minute" benchmark was against DeepSeek's A100/H100 clusters.  Local
models on M3 Max generate tokens at 10-25 tok/s vs 50-100+ tok/s on cloud GPUs.
You cannot close this gap on consumer silicon.  The goal shifted from "match cloud
speed" to "reliable quality at acceptable local latency."

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

## Future directions (next session priorities)

### Immediate (Phase 6)

1. **Gap analysis model switch** — run gap analysis on gemma4:e4b instead of
   qwen3.6:35b.  Cut 328s from runtime (368s → 40s).  Benchmark quality impact.

2. **Build Tier A programmatic benchmark** — produce `phase5_benchmark.py` with
   automated metrics: anchoring score distribution, per-phase latency, claim
   density, entity appearance rate, debate invocation rate.  Catch regressions
   without human effort.

3. **NVIDIA GLiNER-PII integration** — drop-in implementation of the
   `PrivacyModel` interface from `src/security/privacy_model.py`.  570M params,
   55+ entity types, ~1GB at FP16.  `pip install gliner`.

4. **UI & polish** (per README Phase 6): Streamlit/Gradio, session history,
   export formats, Neo4j adapter.

### Near-term (Phase 7)

5. **Vision pipeline** — extract and analyze figures from PDFs.  Load a small
   multimodal model (LLaVA 7B, Qwen-VL, or Granite vision variant).  Memory:
   unload text model, load vision model per figure.

6. **Multi-turn section writing** — extend LangGraph with stateful section
   tracking across Introduction → Methods → Results → Discussion sections.

### Scale (Phase 8)

7. **Publication-scale retrieval** — hierarchical clustering for 100-1000s of
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
See README §Phase 6 for the deliverables. The highest-impact immediate items are:
  1. Switch gap analysis to gemma4:e4b (currently 368s on qwen3.6:35b)
  2. Build Tier A programmatic benchmark (anchoring + latency + density)
  3. Integrate NVIDIA GLiNER-PII privacy model

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
```
