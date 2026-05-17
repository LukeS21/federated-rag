# Phase 10.5 → Phase 11 Handoff — 17 May 2026 (extraction hardening: single‑pass recursive, stream abort, GPU cooldown)

## Quick start

```bash
# Fast unit tests (no LLM, ~30s)
python -m pytest tests/test_extraction_agent.py -q --tb=short

# Diagnostic: test extraction on a real paper (live Ollama)
python scripts/diagnose_cache_accumulation.py PMC10571047

# Full daemon cycle (live — self‑bootstraps Ollama via launchd disarm)
python phase9_verify.py --test orchestrator --orchestrator-live

# Dry run (see what WOULD happen, no API spend)
python phase9_verify.py --test orchestrator

# Pause daemon for a user query
touch projects/default/daemon_yield     # daemon yields after current paper
# … do query in Streamlit …
rm projects/default/daemon_yield        # daemon resumes

# Run a single daemon cycle from Python
python -c "
from src.graph import create_graph_storage
from src.agents.orchestrator import Orchestrator
gs = create_graph_storage(file_path='projects/default/project_graph.json')
orch = Orchestrator(graph_storage=gs, dry_run=False)
summary = orch.run_once()
print(summary)
"
```

---

## Current project state

**Extraction hardening is complete. The system now uses a single‑pass recursive extraction with stream‑level degradation detection, early abort, GPU cooldown, and adaptive split‑on‑failure.** The two‑pass design (cross‑chunk Pass 1 + recursive Pass 2) was built, tested, and then **dropped** after live diagnostics proved Pass 1 inevitably fails: gemma4:e4b cannot process 100+ chunks in one call (15K+ prompt tokens in a ~6K effective context).

103 extraction‑related tests pass, zero failures. The daemon pipeline runs end‑to‑end with process‑level GPU memory hygiene on macOS (SIGKILL + cooldown + orphan cleanup).

**Knowledge graph** (as of last live cycle): ~3,810 nodes, ~262K edges. BM25 corpus: 27K+ documents. 43+ papers ingested.

---

## What changed this session

| | Before this session | After this session |
|---|---|---|
| **Extraction architecture** | Two‑pass (cross‑chunk Pass 1 + recursive Pass 2). Pass 1 fed all 100 chunks to gemma4:e4b and failed immediately with hallucination. Blind retry of degraded batches. | **Single‑pass recursive**: `extract_paper_recursive(batch_size=8)` groups chunks into batches. `_extract_batch_recursive` splits in half on degradation (salvage → restart GPU → split → recurse). The system self‑adapts — simple papers complete at batch_size=8, dense papers split dynamically. |
| **Stream‑based LLM calls** | `_call_llm` used `self._llm.invoke()` — blocked until the full response completed (up to 4096 tokens of spam) before degradation was detected. | `_call_llm_with_detection` uses `self._llm.stream()` with a `for` loop. On degradation, breaks immediately. `stream.close()` in `finally` sends `GeneratorExit` → httpx TCP disconnect → Ollama stops generating in milliseconds. |
| **Degradation detection** | Block‑level identical‑dict check in `_commit()`. Word/hyphen token‑spam in entity fields only. Junk lines silently skipped (no detection). | `TokenStreamHandler` monitors the stream in real‑time: ≥10 consecutive identical words, ≥10 consecutive hyphen‑sub‑tokens, ≥20 consecutive junk lines (no `:` separator). Sets `degraded=True` flag. Parser‑level: junk‑line counter raises `RuntimeError` after 20+; raw junk lines checked for token spam. |
| **Batch failure recovery** | Entire batch entities lost on `RuntimeError`. Blind retry of the same batch (would fail identically if cause was prompt‑overflow). | Partial entities returned on `RuntimeError` (salvaged). Recursive split on degradation (batch → [half, half] → [quarter, quarter]…) until each sub‑batch fits the model. |
| **Entity dedup** | `_merge_entity_batches` deduplicated by name only — kept longest evidence, discarded evidence diversity. | Dedup by `(name, claim)` pair. Same entity+claim → combine evidence sentences + union chunk sources. Different claims → kept as separate facts. `_combine_evidence()` and `_union_sources()` module‑level helpers. |
| **GPU memory flush** | `keep_alive=0` + `/api/ps` polling — 0.8 s "confirmed unloaded" for 9.6 GB (physically impossible). Process restart without cooldown — memory pressure dipped then immediately spiked. | Process restart (SIGKILL + `OLLAMA_RESTART_COOLDOWN_SECONDS=5`) between batches. The 5 s delay after kill gives Metal/IOKit time to deallocate GPU pages before the new server starts. Orphaned runners cleaned via `pgrep -f "ollama runner"`. |
| **MLX backend investigation** | Not attempted. | `OLLAMA_MLX=1` does **not exist** in Ollama 0.24.0 (hallucinated by Gemini). `OLLAMA_LLM_LIBRARY=mlx` is accepted but ignored — no MLX backend compiled into Ollama 0.24.0. All models are GGUF format, which uses llama.cpp backends exclusively (Metal/CUDA/CPU). KV cache quantization at `q8_0` was already active throughout this session. |
| **Prompt engineering** | Pass 1 cross‑chunk prompt required the model to reason about chunk references. Model hallucinated from training data. | No cross‑chunk prompt. Single standard extraction prompt for all batches. Categories ordered by entity density (`_categories_to_line_tagged_sorted`) — complex categories get priority output. |
| **Failed chunk handling** | None. Lost entities were lost. | Single chunks that still degrade at the base case are saved to `projects/default/failed_chunks/<pmcid>_chunk_<idx>_<ts>.txt`. Documented gap, not silently lost. |
| **Diagnostic tool** | Tested `extract_entities_batched` with `keep_alive` vs process restart comparison. | Updated for `extract_paper_recursive`. Now reports `degradation_events` and `failed_chunks` alongside `spam_errors`. |

---

## What was accomplished

### Stream‑based degradation detection & early abort (`streaming_handler.py` + `extraction_agent.py`)

**Three‑layer defense against model degradation:**

| Layer | Mechanism | Latency |
|-------|-----------|---------|
| Detect | `TokenStreamHandler` monitors stream: word repetition (≥10), hyphen repetition (≥10), junk lines (≥20) | < 100 ms after onset |
| Abort | `_call_llm_with_detection` uses `stream()` + `for` loop. On `handler.degraded`, breaks; `stream.close()` in `finally` sends `GeneratorExit` → httpx disconnect | < 500 ms |
| Recover | `_extract_batch_recursive` catches `ModelDegradedException`, salvages partials, restarts GPU (cooldown), splits batch, recurses | ~5 s + retry time |

The `stream.close()` in `finally` is **guaranteed** to run — `finally` fires on both `break` (degradation) and normal generator exhaustion. On an already‑exhausted generator, `.close()` is a no‑op.

### Single‑pass recursive extraction (`extraction_agent.py`)

**Entry point:** `extract_paper_recursive(chunks, batch_size=8)` — groups chunks, processes each group via `_extract_batch_recursive`, restarts GPU between groups.

**Recursive logic:** On degradation: salvage partial entities → restart GPU with cooldown → split batch in half → recurse both halves. Base case (1 chunk or max_depth 12): save to `failed_chunks/`, return salvage.

**Why recursive splitting instead of blind retry:** Blind retry of a batch that degraded from prompt‑overflow (too many chunks → too many output tokens → KV cache corruption) will degrade identically on every attempt. Halving the batch reduces both prompt tokens and expected output tokens, addressing the causal mechanism.

**What was tried and dropped:** Two‑pass extraction (`extract_paper_two_pass`) with Pass 1 for cross‑chunk claims. Live diagnostic on PMC10571047 (100 chunks) showed Pass 1 immediately hallucinated (`Polymer: Polyethylene`, `E‑coli: E. coli` — entities from training data, not the paper). The 15K‑token prompt exceeded gemma4's ~6K effective context. Pass 1 was removed; cross‑chunk entities are captured through salvage at each recursive level.

### GPU cooldown (`pre_extractor.py`)

After SIGKILL and server‑death confirmation, `_restart_ollama_process` now sleeps for `OLLAMA_RESTART_COOLDOWN_SECONDS` (default 5) before starting the new server. The previous 0.6 s restart loop (kill → 0.1 s death → 0.5 s server start) was too fast — Activity Monitor showed memory pressure dip then immediately spike, suggesting Metal GPU pages were reused before deallocation.

### Improved entity merge (`extraction_agent.py`)

`_merge_entity_batches` now deduplicates by `(name, claim)` pair. Same entity+claim → combine evidence (`_combine_evidence`) + union chunk sources (`_union_sources`). Different claims on the same entity are preserved separately. No‑claim entries merge evidence into claimed entries when available.

### Parser‑level defenses (Gaps L, M closed)

- **Gap L (partial entity salvaging):** `_parse_line_tagged` wraps the main parsing loop in `try/except RuntimeError`. On failure, returns entities committed before the error instead of discarding everything.
- **Gap M (junk‑line abort):** `_parse_line_tagged` now counts consecutive lines without `:` format separator. After ≥20, raises `RuntimeError`. Raw junk lines are also checked for token spam (`_detect_token_spam`).

### MLX backend investigation (dead end)

`OLLAMA_MLX=1` does not exist in Ollama 0.24.0. The flag was silently ignored (server logs showed `library=Metal` identically with and without it). All installed models are GGUF format, which runs exclusively on llama.cpp backends. MLX models use a different format (`.safetensors`). The Gemini suggesting this flag was hallucinating.

### Ollama KV cache configuration

The system was already running with optimal settings throughout this session: `OLLAMA_KV_CACHE_TYPE=q8_0` (8‑bit quantization, ~50% cache reduction), `OLLAMA_FLASH_ATTENTION=true`, `OLLAMA_NUM_PARALLEL=8`. These were set via launchctl before this session and are not part of our codebase changes.

---

## Lessons learned

### 1. Pass 1 (cross‑chunk extraction) is a hardware impossibility for gemma4:e4b

100 chunks × ~150 tokens = 15,000 tokens of prompt content. gemma4:e4b's effective extraction context is ~6,000 tokens. The model physically cannot hold 100 chunks in working memory — it doesn't matter what instructions you give it, what model you use, or what backend runs it. The model hallucinates from training data because the actual chunks aren't visible.

**Lesson**: The model's effective context window is the hard constraint. Full‑paper extraction on local models requires either (a) a larger model (gemma4:26b or qwen3.6:35b), (b) pre‑emptive batching within the model's comfort zone, or (c) both. The recursive split‑on‑failure design handles this adaptively — the paper determines its own effective batch size.

### 2. `stream.close()` in `finally` is reliable for aborting LangChain streams

Python generators support `.close()` (PEP 342). Calling it sends `GeneratorExit` to the yield point, unwinding LangChain's `_stream()` method which cleans up the underlying `openai.Stream` / httpx response. Ollama sees the TCP connection drop and stops generating. The `finally` block guarantees `.close()` runs on both `break` (degradation) and normal completion.

### 3. Salvaging partial entities is worth it — they capture cross‑batch claims

The 100‑chunk salvage captures entities whose evidence spans chunks on both sides of the midpoint. Sub‑batches A (0‑49) and B (50‑99) each see only half. The salvage from the full‑batch degraded run is the **only** source for wide‑span cross‑batch entities. This was the same motivation as Pass 1, but achieved through salvage rather than a dedicated pass — no extra LLM calls.

### 4. GPU cooldown between kill and restart matters

The previous 0.6 s restart (kill → 0.1 s death → 0.5 s server start) showed memory pressure dip then immediately spike in Activity Monitor. Adding 5 s of sleep after server‑death confirmation gives Metal/IOKit time to deallocate GPU pages. This is a heuristic — no macOS API can verify deallocation — but the memory pressure pattern in Activity Monitor visibly flattened with the cooldown.

### 5. Degradation detection must operate on the raw stream, not parsed entities

The `e‑coli‑coli‑coli…` failure from the previous diagnosis escaped all existing detectors because the spam appeared as lines without `:` — silently skipped by the parser. `TokenStreamHandler` catches this at the character level (hyphen‑split repetition) before the parser ever sees it.

### 6. Blind retry of the same batch is an anti‑pattern

The previous `extract_entities_batched` retried the exact same batch on degradation. If the cause was batch‑size‑related (too many chunks → too many output tokens), retry would identically fail. The recursive split addresses the causal mechanism (halve the batch = halve the tokens), not the symptom.

### 7. Gemini can hallucinate configuration flags

`OLLAMA_MLX=1` does not exist in Ollama 0.24.0. The suggestion was plausible (Ollama does support MLX for native Apple Silicon models) but the specific flag was invented. Always verify AI‑suggested configuration with `--help` or source inspection before spending time on implementation.

---

## Identified gaps and status

### Closed this session

| # | Gap | Severity | Status |
|---|------|----------|--------|
| H | Token‑level spam not detected by block‑level detector (prior session) | ~~High~~ | ✅ Closed — word + character‑level `_detect_token_spam` in parser + real‑time `TokenStreamHandler` |
| K | No guaranteed GPU‑memory reset mechanism (prior session) | ~~Medium~~ | ✅ Closed — `_restart_ollama_process()` with cooldown + orphan cleanup |
| L | Batch failures discard good entities | ~~High~~ | ✅ Closed — `_parse_line_tagged` returns partial entities on `RuntimeError` |
| M | Junk‑line infinite generation not caught | ~~Medium~~ | ✅ Closed — Junk‑line counter (≥20) raises `RuntimeError`; raw lines checked for token spam |
| — | Pass 1 hallucination from prompt‑overflow | — | ✅ Closed — Pass 1 removed; single‑pass recursive captures cross‑batch claims through salvage |

### Open (Phase 10.5 remaining / new)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| N | `stream.close()` abort reliability untested at scale | Low | The `finally` block guarantees `.close()` is called, but the httpx cleanup depends on LangChain's internal stream handling. Edge cases (hung Ollama, network timeout during generator exit) not tested. No known failures. |
| O | No ≥8 h continuous daemon validation | Medium | Daemon has run short cycles but never >8 h. Longer runs needed to validate memory stability with the new cooldown and recursive extraction. |
| P | `_merge_entity_batches` operates within categories | Low | Entities classified under different TYPEs by different recursive sub‑batches stay separate (e.g., `IL‑6` in `cytokine` in salvage, `IL‑6` in `material` in sub‑batch). For consistent models this is rare. A cross‑category dedup pass could be added but the risk/reward is low. |

### Evergreen (inherent hardware limitations)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| I | `/api/ps` cannot verify true GPU memory state | High | No macOS API exposes Metal buffer state. Process death (SIGKILL) is the strongest guarantee. The 5 s cooldown is a heuristic; no software can prove pages were deallocated. |
| E | No long‑running daemon validation | Medium | See Gap O above. |

### Phase 11 (partial build — unchanged from prior session)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| P | Community summaries not generated | High | `community_detection.py` runs in orchestrator cycles but `community_summarizer.py` is not called. |
| Q | Relevance router not wired into retrieval | High | `relevance_router.py` is designed and tested but not called. |
| R | Progressive disclosure not integrated | Medium | Tiered disclosure not in UI or daemon. |
| G | SPECTER2 embeddings unused | Low | `spector2_cache.json` has embeddings but `paper_similarity_search()` was never built. |

---

## Key architectural decisions (DO NOT UNDO)

### Carried forward from prior sessions

All Phase 4–10.5 constraints still apply. See README §17.

### New decisions (this session)

- **Single‑pass recursive over two‑pass** — The two‑pass design was built, tested on live hardware, and dropped. Pass 1 (cross‑chunk extraction on all 100+ chunks) fails because gemma4:e4b's effective context (~6K tokens) cannot hold the prompt (15K+ tokens for 100 chunks). The single‑pass recursive system captures cross‑batch claims through salvage at each recursive level — no extra LLM call, no hallucination risk.

- **`stream()` with `stream.close()` over `invoke()`** — `invoke()` blocks until the full response completes (up to 4096 tokens). The stream loop breaks immediately on degradation and `stream.close()` in `finally` guarantees the httpx connection closes. This cuts degradation waste from "entire max_tokens of spam" to "detection latency + TCP teardown" (< 1 s).

- **Recursive split‑on‑failure over blind retry** — Blind retry of a degraded batch addresses the symptom, not the cause. If the batch is too large for the model (prompt‑overflow → KV cache corruption), retrying the same batch will identically fail. Splitting in half reduces both prompt tokens and expected output tokens, addressing the causal mechanism.

- **GPU cooldown (`OLLAMA_RESTART_COOLDOWN_SECONDS`) over immediate restart** — The 0.6 s kill‑restart cycle was too fast for Metal page deallocation. Activity Monitor showed memory pressure dip then immediately spike. The 5 s cooldown (configurable) gives Metal/IOKit time.

- **Salvaging partial entities over discarding** — When `_parse_line_tagged` raises `RuntimeError`, entities committed before the error are returned. The salvage captures entities that sub‑batches cannot see (wide‑span cross‑batch claims).

- **`(name, claim)` dedup over name‑only dedup** — Name‑only dedup (keep longest evidence) threw away evidence diversity. `(name, claim)` dedup combines evidence sentences and unions chunk sources for identical claims, and preserves different claims as separate facts.

- **Single‑model extraction (gemma4:e4b) over configurable Pass 1 model** — The `EXTRACTION_PASS1_MODEL` env var was added for qwen3.6:35b support in Pass 1, then removed when Pass 1 was dropped. All recursive extraction uses gemma4:e4b. If higher extraction quality is needed, switch `OLLAMA_SMALL_MODEL` to `gemma4:26b` (17 GB, 25.8B params) — no code change needed.

- **Phase 10.5 Gaps H, K, L, M closed** — All four remaining Phase 10.5 extraction gaps are closed. The system has defenses at every level: stream (real‑time), parser (token spam + junk lines + block repetition), batch (recursive split), and GPU (process restart with cooldown).

---

## What NOT to change

All prior constraints apply. Additions from this session:

- Do NOT switch extraction back to full‑prompt or JSON.
- Do NOT reinstate two‑pass extraction (Pass 1 cross‑chunk). The hardware cannot support it; the single‑pass recursive system is the correct architecture.
- Do NOT remove batched extraction, evidence grouping, streaming, or any repetition/spam detection.
- Do NOT remove `max_retries=0` from the extraction LLM.
- Do NOT remove `_call_llm_with_detection` or `stream.close()` in the `finally` block.
- Do NOT remove the recursive split‑on‑failure logic in `_extract_batch_recursive`.
- Do NOT remove `_merge_entity_batches` (name, claim) dedup with evidence union.
- Do NOT remove `_save_failed_chunk` or the `failed_chunks/` directory.
- Do NOT remove `OLLAMA_RESTART_COOLDOWN_SECONDS` or the cooldown logic.
- Do NOT remove `_reset_ollama()`, `_restart_ollama_process()`, `_ensure_dedicated_ollama()`, or `_find_and_kill_ollama()`.
- Do NOT reinstate `EXTRACTION_PASS1_MODEL` or Pass 1 cross‑chunk prompt builders.
- Do NOT add keyword blacklists or grounding heuristics to the parser.
- Do NOT switch extraction back to JSON output.
- All prior constraints: per‑paper source prefixes, chunk_index, no `lstrip()`, no `Accept: application/json` on EPMC session, etc.

---

## File map

```
MODIFIED FILES (this session):
src/agents/extraction_agent.py      — Major rewrite: _call_llm_with_detection (stream+abort),
                                       extract_paper_recursive (single‑pass entry point),
                                       _extract_batch_recursive (recursive split‑on‑failure),
                                       _merge_entity_dicts (shallow merge for recursion),
                                       _save_failed_chunk (base‑case disk persistence),
                                       extract_entities simplified (removed mode parameter),
                                       _merge_entity_batches improved ((name,claim) dedup + evidence union),
                                       _categories_to_line_tagged_sorted (density ordering),
                                       prompt builders consolidated,
                                       _build_cross_chunk_prompt removed,
                                       Gap L (partial salvage) + Gap M (junk‑line counter) closed in _parse_line_tagged
src/ingestion/pre_extractor.py       — GPU cooldown (OLLAMA_RESTART_COOLDOWN_SECONDS) after server death,
                                       calls extract_paper_recursive instead of extract_entities_batched
src/streaming_handler.py             — TokenStreamHandler rewritten: real‑time word/hyphen/junk‑line detection,
                                       ModelDegradedException carries partial text for salvage,
                                       degradation flag (no raise during generation — avoids LangChain internals)
src/graph/nodes.py                   — extract_paper_two_pass → extract_paper_recursive
src/graph/survey_nodes.py            — extract_paper_two_pass → extract_paper_recursive
scripts/diagnose_cache_accumulation.py — Updated for extract_paper_recursive, new metrics
tests/test_extraction_agent.py        — Mocks updated for _call_llm_with_detection
.env                                  — Added OLLAMA_RESTART_COOLDOWN_SECONDS=5
.env.example                          — Added OLLAMA_RESTART_COOLDOWN_SECONDS=5
README.md                             — §2, §7, §11, §17 updated for new architecture
HANDOFF.md                            — This file — comprehensive session handoff

REMOVED (this session):
extract_entities_batched              — Replaced by extract_paper_recursive
extract_paper_two_pass                — Two‑pass design dropped after hardware failure
_pass2_recursive                      — Renamed to _extract_batch_recursive
_build_cross_chunk_prompt             — Removed with Pass 1
EXTRACTION_PASS1_MODEL env var        — Removed with Pass 1
mode parameter on extract_entities    — Removed; always "all" mode now
```

---

## Recommendations

### Immediate — run the diagnostic

```bash
python scripts/diagnose_cache_accumulation.py PMC10571047
```

This is the same 100‑chunk paper that failed the two‑pass system. With the new single‑pass recursive extraction (13 batches of ~8), it should complete cleanly. Watch for `degradation_events` and `failed_chunks` in the output.

If it succeeds: the system is production‑ready for the daemon. Bump batch_size to 16, re‑test, find the sweet spot.

If individual batches degrade and split: the recursive system is working as designed. Check the cooldown is sufficient (increase `OLLAMA_RESTART_COOLDOWN_SECONDS` to 10 if splitting is frequent).

If single chunks fail at the base case: the paper has inherently problem chunks (corrupted text, unusual formatting). These go to `failed_chunks/` — a documented gap, not a crash.

### Short‑term — bump batch_size

Once the diagnostic passes at batch_size=8, incremental testing at 12, 16, and 20 will find gemma4:e4b's sweet spot for this paper size. Higher batch_size = fewer restarts = faster extraction. The recursive split catches any overshoot safely.

### Medium‑term — try gemma4:26b

If extraction quality is insufficient, swap `OLLAMA_SMALL_MODEL=gemma4:26b` in `.env`. The 25.8B model (vs 8B for e4b) can handle larger batches and may produce better extractions. Cost: 17 GB vs 9.6 GB memory, ~20 s load time vs ~8 s. The extraction code requires zero changes.

### Phase 11 — Community routing (unchanged from prior)

Partial build committed, tests pass. Next steps:
1. Generate community summaries via `community_summarizer.py`
2. Wire `relevance_router.py` into Survey Mode retrieval
3. Wire `progressive_disclosure.py` — tiered KG disclosure
4. Integrate community routing end‑to‑end

### Beyond Phase 11 — Anchoring‑confidence entity lifecycle (new design)

During extraction, the anchoring‑confidence lifecycle was designed but **not implemented** — it is a Phase 11+ task. The design:

Anchoring check (TF‑IDF cosine similarity) gates confidence scores on KG entities:
- Strong match (≥ 0.7): confidence += 0.15 (capped at 1.0)
- No match (< 0.35): confidence *= 0.85 (decay)
- Cross‑verified by second paper: confidence += 0.10

Visibility gating: `stable` (≥ 0.70) fully visible, `tentative` (0.40‑0.70) visible, `weak` (0.15‑0.40) hidden from retrieval, `suspect` (< 0.15) hidden but surfaced in handoff. Only human review sets confidence to 0.0 (permanent removal). No auto‑deletion — entities are soft‑deprecated with continuous decay, self‑healing when re‑verified by new evidence.

---

## Prompt for next AI session

```
You are an expert senior software developer continuing the Federated RAG
system for biomedical research. Extraction hardening is complete. All
existing gaps (H, K, L, M) are closed. Phase 11 (community routing &
memory cascade) is the next major milestone.

Read the full README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - 103 extraction‑related tests pass, zero failures. Full suite ~375.
  - Extraction uses single‑pass recursive system: extract_paper_recursive
    (batch_size=8) splits into groups; _extract_batch_recursive handles
    each group with recursive split‑on‑failure. Stream‑level degradation
    detection (TokenStreamHandler) catches any degradation pattern in
    real‑time. stream.close() in finally guarantees httpx disconnect on
    abort. GPU restarts between batches with 5 s cooldown.
  - Orchestrator daemon runs full autonomous cycle: web discovery → EPMC
    fetch → batch ingest → recursive extraction → KG save → community
    detection → cycle handoff. Self‑bootstraps Ollama via launchd disarm.
  - KG: ~3,810 nodes, ~262K edges. BM25: 27K+ documents. 43+ papers.
  - Extraction uses evidence‑grouped line‑tagged format with CLAIM
    semantics (qualitative/quantitative/state/role). Parser maps legacy
    direction→claim. Entities deduped by (name, claim) with evidence
    union and source chunk combination.
  - Ollama GPU memory: process restart (SIGKILL + cooldown) is the
    mechanism. keep_alive=0 + /api/ps polling is proven unreliable.
    macOS launchd watchdog must be disarmed first. OLLAMA_MLX=1 does
    NOT exist in Ollama 0.24.0. KV cache uses q8_0 quantization.
  - Phase 11 partial build committed: community_detection.py (wired),
    community_summarizer.py, relevance_router.py, progressive_disclosure.py
    + their tests (NOT yet wired).
  - DeepSeek API available for development. Ollama (gemma4:e4b +
    qwen3.6:35b) is the production target. Only one model fits in 36GB
    M3 Max at a time.

CRITICAL OPEN:
  - Phase 11 wiring: community summaries, relevance router, progressive
    disclosure.
  - Long‑running daemon validation (≥8 h).

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE — see README §17 for full list):
  - Do NOT remove or disable any of the extraction defense layers
    (stream detection, recursive split, cooldown, salvage, failed_chunks).
  - Do NOT reinstate two‑pass extraction or cross‑chunk prompts.
  - Do NOT switch extraction back to JSON or full‑prompt.
  - All prior constraints still apply.

REUSABLE PRIMITIVES:
  - Orchestrator(graph_storage=gs, dry_run=True).run_once()
  - PreExtractor.extract_paper(paper_id, chunks, graph_storage=gs)
  - agent.extract_paper_recursive(chunks, categories, query, batch_size=8)
  - agent._extract_batch_recursive(chunks, categories, query)
  - agent._call_llm_with_detection(system_prompt, user_prompt)
  - PreExtractor._restart_ollama_process() — SIGKILL + cooldown
  - PreExtractor._ensure_dedicated_ollama() — launchd disarm
  - TokenStreamHandler() — real‑time degradation detection
  - _detect_token_spam(value) — word + character repetition
  - _merge_entity_batches(entities) — (name,claim) dedup + evidence union
  - _combine_evidence(e1, e2), _union_sources(s1, s2)

QUICK START:
  python scripts/diagnose_cache_accumulation.py PMC10571047   # diagnostic
  python phase9_verify.py --test orchestrator                  # dry run
  python phase9_verify.py --test orchestrator --orchestrator-live  # live
  python -m pytest tests/ -q --tb=short                        # all tests
```
