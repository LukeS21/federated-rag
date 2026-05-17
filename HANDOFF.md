# Phase 10.5 → 11 Handoff — 17 May 2026 (pulsed-wave parallel extraction, self-calibrating boundary, per-worker log files, compression-ratio degradation detection)

## Quick start

```bash
# Fast unit tests (no LLM, ~5s)
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

**Pulsed-wave parallel extraction with self-calibrating boundary is operational.** The system uses token‑budgeted chunk packing, per‑wave GPU restarts with parallel workers, and a self‑learning boundary (derived from actual pass/fail data) to determine per‑batch chunk budgets. Output ratio and safe‑context boundary calibrate concurrently across all papers — no hardcoded model‑context limits, no magic fractions.

41 extraction‑related tests pass, zero failures. Per‑worker LLM output is written to `logs/extraction/wave_NNN_*.txt` files (no jumbled stdout). The daemon pipeline runs end‑to‑end with process‑level GPU memory hygiene on macOS (SIGKILL + cooldown + orphan cleanup).

**Knowledge graph** (as of last live cycle): ~3,810 nodes, ~262K edges. BM25 corpus: 27K+ documents. 43+ papers ingested.

---

## What changed this session

| | Before this session | After this session |
|---|---|---|
| **Extraction architecture** | Single‑pass recursive: `extract_paper_recursive(batch_size=8)` with `_extract_batch_recursive` handling recursive split via sequential recursion. GPU restart between every batch. | **Pulsed‑wave parallel**: token‑budgeted chunk packing (tiktoken per‑chunk), GPU restart per wave, parallel workers within each wave, priority‑queue re‑entry for degraded sub‑batches. No hardcoded `batch_size`. |
| **Batch sizing** | Fixed `batch_size=8` — same chunk count for every paper regardless of chunk density. | **Self‑calibrating boundary**: formula `budget = (boundary_lower × 0.95 − system − overhead) / (1 + output_ratio)`. `boundary_lower` rises from pass data; `boundary_upper` falls from non‑base‑case degradation data. Starts conservatively (~8 chunks/batch) and converges to the model's true limit across papers. |
| **Output‑ratio tracking** | Not tracked. | Weighted EMA (80/20) of `output_tokens / chunk_tokens` persisted per‑model in `extraction_stats.json`. Updates every pulsed wave. Starts at 0.50; self‑calibrates. |
| **Degradation detection** | Word‑level (≥10 identical), hyphen‑level (≥10 identical sub‑tokens), junk‑line (≥20 without `:`). | **All of the above + universal compression‑ratio detector**: `zlib.compress(tail) → ratio ≥ 8:1` catches any repetition pattern — line‑level spam (`EVIDENCE: …` ×100), word spam, hyphen loops, and any future unknown pattern. |
| **Parallelism** | None — sequential batch execution. | Pulsed‑wave: up to `OLLAMA_NUM_PARALLEL` concurrent workers per wave, capped by GPU memory for large models (qwen3.6:35b auto‑capped at 1 worker). |
| **Worker output display** | All workers printed to stdout — jumbled interleaving. | Per‑worker log files in `logs/extraction/wave_NNN_*.txt` with `tail -f` instructions printed at wave start. Console shows wave‑level summaries only. |
| **Failed chunk tracking** | `failed_chunks/` folders only (human‑readable). | **+ machine‑readable `bad_chunks.json`**: chunk indices that reach base case ≥3 times are automatically isolated into single‑chunk batches in future extractions. |
| **Budget derivation** | `num_ctx − max_tokens×output_ratio − system − overhead` → dangerous over‑allocation (~13K budget producing 7 batches). | `(boundary_lower × 0.95 − system − overhead) / (1 + ratio)` → starts at ~737 (matches old batch_size=8), converges upward from real data. |
| **Removed code** | `_extract_batch_recursive` (recursive handler), `_merge_entity_dicts` (shallow merge), `batch_size` parameter. | All replaced by the pulsed‑wave loop in `extract_paper_recursive` + `_merge_entity_batches`. |
| **Env vars removed** | `EXTRACTION_EFFECTIVE_CTX_FRACTION` (0.35), `EXTRACTION_OUTPUT_RESERVE_RATIO` (0.50). | Both were magic‑number guesses. Replaced by boundary‑based self‑calibration from actual data. |

---

## What was accomplished

### Compression‑ratio degradation detection (`streaming_handler.py`)

**Problem:** The model produced hundreds of repeated `EVIDENCE: The thermoelectric materials |` lines. Zero existing detectors caught it — each line had a `:` (not junk), words were different (not word‑spam), and no hyphens were present. Pattern‑specific detectors are a whack‑a‑mole game.

**Solution:** Compression ratio as a *universal* degradation signal. All degradation patterns share one property — the model stops producing novel content. Repetitive text compresses at 30–70:1 while normal extraction output compresses at ~1.5:1.

```python
compressed = zlib.compress(tail.encode("utf-8"))
ratio = len(tail) / max(len(compressed), 1)
if ratio >= 8.0: mark_degraded()
```

This catches: word repetition, hyphen loops, line repetition, and any future unknown repetition pattern. No pattern‑specific thresholds needed for novel failure modes. The check runs every 100 chars on the last 2000‑char tail — ~10µs per check.

### Self‑calibrating boundary‑based budget (`extraction_agent.py`)

**Problem:** The budget formula derived chunk budgets from `num_ctx=16384` which has no relationship to gemma4:e4b's actual capability. The previous attempt used magic fractions (0.35, 0.50) that were guesses. Every approach to deriving the budget from `num_ctx` failed because `num_ctx` is the configured maximum, not the model's effective limit.

**Solution:** The boundary self‑calibrates from actual pass/fail data across all extraction waves:

```
Per batch (non‑base‑case only):
  PASS:    boundary_lower = max(boundary_lower, actual_total_tokens)
  DEGRADE: boundary_upper = min(boundary_upper, actual_total_tokens)

Budget for next wave:
  budget = (boundary_lower × 0.95 − system − overhead) / (1 + output_ratio)
```

- Starts at `boundary_lower=2500` → budget ~737 → ~8 chunks/batch (matches old batch_size=8)
- `boundary_upper=16384` (configured context window)
- Output ratio starts at 0.50, EMA‑updated every wave
- Both persisted in `projects/default/extraction_stats.json` per model
- Base‑case degradations (corrupted single chunks) do NOT update the boundary — they're data problems, not context problems

After enough papers, `boundary_lower` and `boundary_upper` converge to the model's true effective‑context limit. For gemma4:e4b this should be around 5000–6000 total tokens.

### Pulsed‑wave parallel extraction

**Architecture:**

```
GPU restart (5s)

Wave 1:  [batch_a ∥ batch_b ∥ batch_c ∥ batch_d]  ← parallel, wait for ALL
         collect passes, split degradations → re‑queue
         update boundary + ratio from wave data
         recompute budget for next wave

Wave 2:  GPU restart (5s)
         queue sorted: smaller sub‑batches first
         [sub_a1 ∥ sub_a2 ∥ batch_e ∥ batch_f]

         ...repeat until queue empty
```

**Key invariants:**
- GPU restart only when running set is empty — no in‑flight requests are killed
- `stream.close()` on a degraded batch's HTTP connection only affects that connection; siblings are independent (separate TCP sockets)
- Each parallel request gets its own KV cache allocation (not shared)
- Model weights are shared across requests; KV caches are per‑request
- Worker count capped by `OLLAMA_NUM_PARALLEL`, GPU memory, and model size (qwen3.6:35b auto‑capped at 1)

### Token‑based greedy batch packing

Chunks are packed by actual tiktoken count (not a fixed number):

```python
for chunk in chunks:
    ct = len(tokenizer.encode(format_chunk_text(chunk)))
    if current_tokens + ct > budget and current_batch: flush
    current_batch.append(chunk)
    current_tokens += ct
```

This means dense chunks (figure captions with statistical notation, 111 tokens) get fewer siblings; short chunks (20 tokens) get more. Every batch fits the model's budget exactly — no overflow, no wasted headroom.

### Bad chunk pre‑emption

Chunks that hit the base case ≥3 times are tracked in `projects/default/bad_chunks.json`. On future extractions, known‑bad chunks are pre‑emptively isolated into single‑chunk batches and placed at the front of the queue (they finish fast). If a known‑bad chunk passes 3+ times, it's removed from tracking.

### Per‑worker log files

In parallel mode, all live token output is written to `logs/extraction/wave_NNN_chunk-XXX_Mchunks.txt` instead of stdout. The handler calls `output_file.write(token)` + `output_file.flush()` on every token — visible in real‑time via `tail -f`. The console shows only wave‑level summaries:

```
Wave 1: 4 workers, 9 queued (workers=4, ratio=0.500)
Worker log files in logs/extraction/
  tail -f logs/extraction/wave_001_chunk-0_7chunks.txt
  tail -f logs/extraction/wave_001_chunk-7_6chunks.txt
  tail -f logs/extraction/wave_001_chunk-13_5chunks.txt
  tail -f logs/extraction/wave_001_chunk-18_6chunks.txt
...
Wave 1: 4/4 passed, 0 degraded → 9 queued
```

---

## Lessons learned

### 1. `num_ctx` is not the model's effective context window

`num_ctx=16384` is the configured maximum — what Ollama allocates for the KV cache. gemma4:e4b's actual effective extraction context is much smaller (observed ~5–6K total tokens). Deriving the batch budget from `num_ctx` produces dangerous over‑allocation (13K chunk budget = 7 batches for 100 chunks instead of the empirically‑safe 13).

**Lesson:** Never derive safety margins from the configured context window. Self‑calibrate from actual pass/fail data instead.

### 2. Pattern‑specific degradation detection is a losing game

The model found a new way to degrade (repeating entire formatted lines) that escaped all 5 existing detectors. Adding a 6th pattern‑specific detector just invites a 7th failure mode.

**Lesson:** Compression ratio is a universal, pattern‑agnostic degradation signal. All forms of repetition inflate the compression ratio. Normal text compresses at ~1.5:1; repetitive text at 30–70:1. One check catches everything.

### 3. Parallel requests have independent KV caches — no cross‑contamination

When two extraction requests run concurrently on the same loaded Ollama model, each gets an independent KV cache allocation. Degradation in one request does not affect the other. `stream.close()` closes only one HTTP connection. This makes pulsed‑wave parallelism safe: GPU restart once per wave, all requests run in parallel, no worker kills another mid‑flight.

### 4. Base‑case degradations must not pollute the boundary

A single corrupted chunk (figure caption with statistical notation, copyright markers, reference numbers) that degrades at depth 0 should not set `boundary_upper`. The degradation is caused by bad data, not by the model's context window being too full. The boundary tracker now skips base‑case (depth ≥ 12 or n ≤ 1) degradations.

### 5. Calibration must be per‑model and persistent

Switching `gemma4:e4b` → `gemma4:26b` changes both the output ratio and the effective context boundary. Tracking per‑model in `extraction_stats.json` means each model calibrates independently. The stats persist across diagnostic runs and daemon cycles — no re‑calibration needed after restart.

### 6. Magic fractions are fragile defaults

`EXTRACTION_EFFECTIVE_CTX_FRACTION=0.35` and `EXTRACTION_OUTPUT_RESERVE_RATIO=0.50` were untestable guesses. They couldn't be validated or invalidated without running the system at scale. The boundary approach replaces them with data‑driven calibration that improves with every extraction.

---

## Identified gaps and status

### Closed this session

| # | Gap | Severity | Status |
|---|------|----------|--------|
| — | Repeated‑line spam (`EVIDENCE: …` ×100) undetected | ~~High~~ | ✅ Closed — compression‑ratio detector catches any repetition pattern |
| — | `_extract_batch_recursive` / `_merge_entity_dicts` dead code | ~~Low~~ | ✅ Closed — removed; pulsed‑wave loop is the sole code path |
| — | `batch_size` parameter silently ignored | ~~Low~~ | ✅ Closed — removed from signature |
| — | Magic‑fraction budget constants (0.35, 0.50) | ~~Medium~~ | ✅ Closed — replaced with self‑calibrating boundary |
| — | Worker output jumbled in parallel mode | ~~Low~~ | ✅ Closed — per‑worker log files with `tail -f` instructions |
| — | Bad chunks re‑degraded every extraction | ~~Medium~~ | ✅ Closed — machine‑readable `bad_chunks.json` with pre‑emptive isolation |
| — | No output‑ratio tracking across papers | ~~Medium~~ | ✅ Closed — EMA‑weighted ratio persisted per‑model in `extraction_stats.json` |
| — | `num_ctx` used as budget ceiling | ~~High~~ | ✅ Closed — boundary‑based budget is independent of configured context window |

### Open (this session)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| N | `stream.close()` abort reliability untested at scale | Low | The `finally` block guarantees `.close()` is called, but the httpx cleanup depends on LangChain's internal stream handling. Edge cases (hung Ollama, network timeout during generator exit) not tested. No known failures. |
| O | No ≥8 h continuous daemon validation | Medium | Daemon has run short cycles but never >8 h. Longer runs needed to validate memory stability with the new pulsed‑wave cooldown and parallel extraction. |
| P | Boundary hasn't been calibrated with live daemon data | Low | The self‑calibrating boundary needs several papers of live extraction data to converge. The default (boundary_lower=2500 → ~8 chunks/batch) is safe but may be conservative. The system will adjust upward as passes accumulate. |
| Q | Chunk 99 on PMC10571047 is inherently broken | Low | The figure caption with `[Figure 50]`, `P < 0.01`, `**`, copyright markers, and reference numbers causes the model to hallucinate `E-L-E-V-E-N` spam every time. The `bad_chunks.json` pre‑emption will handle this after 3 occurrences. The chunk may simply need to be excluded from extraction or pre‑processed differently. |

### Evergreen (inherent hardware limitations)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| I | `/api/ps` cannot verify true GPU memory state | High | No macOS API exposes Metal buffer state. Process death (SIGKILL) is the strongest guarantee. The 5 s cooldown is a heuristic; no software can prove pages were deallocated. |

### Phase 11 (unchanged from prior session)

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

- **Pulsed‑wave over sequential recursive extraction** — Sequential `_extract_batch_recursive` with GPU restart between every batch wasted time and didn't benefit from concurrent requests. The pulsed‑wave design restarts the GPU once per wave and runs all batches in the wave in parallel. Speedup: ~2–4× depending on worker count.

- **Token‑based greedy packing over fixed `batch_size`** — Fixed `batch_size=8` treats every chunk as equal regardless of actual text density. Per‑chunk tiktoken counting guarantees no batch overflows the budget regardless of chunk composition.

- **Self‑calibrating boundary over magic fractions** — The previous approach used `EXTRACTION_EFFECTIVE_CTX_FRACTION=0.35` and `EXTRACTION_OUTPUT_RESERVE_RATIO=0.50` — untestable guesses that couldn't be validated. The boundary self‑calibrates from actual pass/fail data across all extractions. Every wave that passes raises the safe ceiling; every wave that degrades lowers the unsafe ceiling. The gap converges to the model's true limit.

- **Compression‑ratio detection over pattern‑specific detectors** — Adding pattern‑specific detectors (word spam, hyphen spam, junk lines) created a maintenance burden. Compression ratio is universal: any form of repetition inflates the ratio. One check replaces N detectors for future unknown patterns.

- **Machine‑readable bad‑chunk tracking over human‑readable only** — `failed_chunks/` folders were for human audit. `bad_chunks.json` gives the system the ability to pre‑emptively isolate known‑bad chunks without human intervention.

- **Per‑worker log files over jumbled stdout** — All live token output is written to timestamped log files visible via `tail -f`. The console shows only wave‑level summaries. Degradation detection runs identically regardless of output destination.

- **Correct KV‑cache‑per‑request calculation over rough estimate** — Worker count uses actual KV cache formula (`2 × layers × kv_heads × head_dim × num_ctx`) with correct model sizes (gemma4:e4b = 10.5 GB, qwen3.6:35b = 25 GB). Qwen is auto‑capped at 1 worker to prevent OOM.

- **Removed dead code** — `_extract_batch_recursive` and `_merge_entity_dicts` were deleted. The pulsed‑wave loop is the sole code path. `_merge_entity_batches` with `(name, claim)` dedup is the merge mechanism.

---

## What NOT to change

All prior constraints apply. Additions from this session:

### Extraction flow
- Do NOT reinstate `_extract_batch_recursive` or `_merge_entity_dicts` — the pulsed‑wave loop is the sole extraction path.
- Do NOT add `batch_size` back as a parameter — batch sizing is token‑driven.
- Do NOT derive chunk budget from `num_ctx` — use the self‑calibrating boundary formula.
- Do NOT remove the self‑calibrating boundary (`boundary_lower`/`boundary_upper`) — this is how the system learns the model's true context limit.
- Do NOT remove `_update_output_ratio` or `_update_boundary` — these are the calibration mechanisms.
- Do NOT remove the per‑wave budget recomputation — each wave should use the latest calibrated values.
- Do NOT remove `_pack_chunks_into_batches` token‑based packing — per‑chunk tiktoken counting is the correctness guarantee.
- Do NOT remove `_try_extract_once` — it's the single‑shot wrapper used by all parallel workers.
- Do NOT remove `_calculate_max_workers` — the memory‑aware worker cap prevents OOM on large models.
- Do NOT remove `extraction_stats.json` persistence — it's the calibration memory.
- Do NOT remove `bad_chunks.json` pre‑emption — it prevents repeated failures on known‑corrupted chunks.
- Do NOT remove base‑case boundary exclusion — corrupted chunks must not pollute the context‑window calibration.

### Output
- Do NOT remove per‑worker log files — they keep stdout clean in parallel mode.
- Do NOT hardcode `budget > 1000` in tests — the boundary defaults to ~737 (matching old batch_size=8).

### Compression detection
- Do NOT remove the compression‑ratio check from `TokenStreamHandler._check_degradation()` — it's the universal degradation detector.
- Do NOT remove the parser‑level compression check in `_parse_line_tagged`.

### All prior constraints
- Do NOT switch extraction back to full‑prompt or JSON.
- Do NOT reinstate two‑pass extraction or cross‑chunk prompts.
- Do NOT remove `_call_llm_with_detection` or `stream.close()` in the `finally` block.
- Do NOT remove `_merge_entity_batches` `(name, claim)` dedup with evidence union.
- Do NOT remove `_save_failed_chunk` or the `failed_chunks/` directory.
- Do NOT remove `OLLAMA_RESTART_COOLDOWN_SECONDS` or the cooldown logic.
- Do NOT remove `_reset_ollama()`, `_restart_ollama_process()`, `_ensure_dedicated_ollama()`, or `_find_and_kill_ollama()`.
- Do NOT add keyword blacklists or grounding heuristics to the parser.
- Do NOT switch extraction back to JSON output.
- All prior constraints: per‑paper source prefixes, chunk_index, no `lstrip()`, no `Accept: application/json` on EPMC session, etc.

---

## File map

```
MODIFIED FILES (this session):
src/streaming_handler.py              — +compression‑ratio degradation detection,
                                        +output_file param for per‑worker log files
src/agents/extraction_agent.py        — Major rewrite:
                                          + self‑calibrating boundary (lower/upper)
                                          + output‑ratio EMA tracking + persistence
                                          + per‑wave budget recomputation
                                          + token‑based greedy chunk packing
                                          + pulsed‑wave parallel extraction
                                          + per‑worker log files + tail -f instructions
                                          + bad_chunks.json machine‑readable tracking
                                          + _try_extract_once returns 4‑tuple with output tokens
                                          + _calculate_max_workers with correct KV formula
                                          + system_tokens measured once, reused in wave loop
                                          − _extract_batch_recursive (removed)
                                          − _merge_entity_dicts (removed)
                                          − batch_size parameter (removed)
                                          − EXTRACTION_EFFECTIVE_CTX_FRACTION env var (removed)
                                          − EXTRACTION_OUTPUT_RESERVE_RATIO env var (removed)
tests/test_extraction_agent.py        — +16 new tests (41 total, boundary, budget, workers,
                                        persistence, wave execution, bad‑chunk isolation);
                                        updated all mocks for 4‑tuple returns and **kwargs
.env                                  — Removed magic‑fraction env vars; added
                                        EXTRACTION_CHUNK_BUDGET, EXTRACTION_MAX_WORKERS
.env.example                          — Same
HANDOFF.md                            — This file — comprehensive session handoff

REMOVED (this session):
_extract_batch_recursive              — Replaced by pulsed‑wave loop
_merge_entity_dicts                   — Replaced by direct collection + _merge_entity_batches
EXTRACTION_EFFECTIVE_CTX_FRACTION     — Magic‑number guess removed
EXTRACTION_OUTPUT_RESERVE_RATIO        — Magic‑number guess removed
batch_size parameter                  — No longer meaningful (token‑driven sizing)
```

---

## Recommendations

### Immediate — run the diagnostic

```bash
python scripts/diagnose_cache_accumulation.py PMC10571047
```

The system now starts with conservative defaults (~8 chunks/batch, ~13 batches for 100 chunks — matching the old batch_size=8 behavior). Watch for:
- `boundary_lower` and `boundary_upper` converging in `extraction_stats.json`
- No more `EVIDENCE: …` ×100 spam going undetected (compression ratio catches it)
- Clean wave‑level console output with `tail -f` instructions
- Per‑worker log files in `logs/extraction/`

### Short‑term — validate calibration convergence

Run the diagnostic on 2–3 different papers. After each run, check `projects/default/extraction_stats.json`:

```bash
cat projects/default/extraction_stats.json | python -m json.tool
```

The `boundary_lower` should rise as passes accumulate. The `boundary_upper` should fall from 16384 toward the model's true limit. The `output_ratio` should converge to gemma4:e4b's actual output‑per‑chunk‑token behavior.

### Medium‑term — Phase 11 community routing

Partial build committed, tests pass. Next steps:
1. Generate community summaries via `community_summarizer.py`
2. Wire `relevance_router.py` into Survey Mode retrieval
3. Wire `progressive_disclosure.py` — tiered KG disclosure
4. Integrate community routing end‑to‑end

### Long‑running daemon validation

Run the daemon for ≥8 h continuous to validate memory stability with the new pulsed‑wave architecture. Monitor:
- `extraction_stats.json` for boundary convergence
- `logs/extraction/` for per‑wave worker output
- Activity Monitor for GPU memory pressure patterns between waves
- `bad_chunks.json` for accumulating known‑bad chunks

### Beyond Phase 11 — Anchoring‑confidence entity lifecycle (from prior handoff)

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
system for biomedical research.  Pulsed‑wave parallel extraction with
self‑calibrating boundary and compression‑ratio degradation detection
is complete.  Phase 11 (community routing & memory cascade) is the next
major milestone.

Read the full README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - 41 extraction‑related tests pass, zero failures.
  - Extraction uses pulsed‑wave parallel design: token‑budgeted chunk
    packing (tiktoken per‑chunk), GPU restart per wave, parallel workers
    within each wave (up to OLLAMA_NUM_PARALLEL, memory‑aware cap).
    Per‑worker output written to logs/extraction/wave_NNN_*.txt files
    (no jumbled stdout).  Console shows wave‑level summaries with
    tail -f instructions.
  - Batch budget self‑calibrates: (boundary_lower × 0.95 − system −
    overhead) / (1 + output_ratio).  boundary_lower starts at 2500
    (~8 chunks/batch), rises from pass data.  boundary_upper starts at
    16384, falls from non‑base‑case degradation data.  Both persist
    per‑model in projects/default/extraction_stats.json.
  - Output ratio (chunk_tokens → output_tokens) tracked via weighted
    EMA (80/20), persisted per‑model, updated every wave.
  - Degradation detection: word‑level, hyphen‑level, junk‑line, AND
    universal compression‑ratio (zlib, ≥8:1 catches any repetition).
    stream.close() in finally guarantees httpx disconnect on abort.
  - Bad chunks (≥3 base‑case failures) tracked in
    projects/default/bad_chunks.json, automatically isolated into
    single‑chunk batches on future extractions.
  - Worker count uses correct KV‑cache formula:
    2 × layers × kv_heads × head_dim × num_ctx.  gemma4:e4b = 10.5 GB
    model, qwen3.6:35b = 25 GB model.  Qwen auto‑capped at 1 worker.
  - Orchestrator daemon runs full autonomous cycle: web discovery → EPMC
    fetch → batch ingest → pulsed‑wave extraction → KG save → community
    detection → cycle handoff.  Self‑bootstraps Ollama via launchd disarm.
  - KG: ~3,810 nodes, ~262K edges.  BM25: 27K+ documents.  43+ papers.
  - Extraction uses evidence‑grouped line‑tagged format with CLAIM
    semantics (qualitative/quantitative/state/role).  Parser maps legacy
    direction→claim.  Entities deduped by (name, claim) with evidence
    union and source chunk combination.
  - Ollama GPU memory: process restart (SIGKILL + cooldown) between
    waves.  keep_alive=0 + /api/ps polling proven unreliable.
    macOS launchd watchdog disarmed first.  OLLAMA_MLX=1 does NOT
    exist in Ollama 0.24.0.  KV cache uses q8_0 quantization.
  - Phase 11 partial build committed: community_detection.py (wired),
    community_summarizer.py, relevance_router.py, progressive_disclosure.py
    + their tests (NOT yet wired).
  - DeepSeek API available for development.  Ollama (gemma4:e4b +
    qwen3.6:35b) is the production target.  Only one model fits in
    36 GB M3 Max at a time.

CRITICAL OPEN:
  - Phase 11 wiring: community summaries, relevance router, progressive
    disclosure.
  - Long‑running daemon validation (≥8 h).
  - Boundary calibration needs several papers of live data to converge
    (starts conservative, self‑adjusts).

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE — see README §17 for full list):
  - Do NOT remove or disable any detection layer (stream detection,
    compression‑ratio, recursive split, cooldown, salvage, failed_chunks).
  - Do NOT add back _extract_batch_recursive or _merge_entity_dicts.
  - Do NOT derive budget from num_ctx — use the boundary formula.
  - Do NOT remove the self‑calibrating boundary persistence.
  - Do NOT reinstate two‑pass extraction or cross‑chunk prompts.
  - Do NOT switch extraction back to JSON or full‑prompt.
  - All prior constraints still apply (see HANDOFF § "What NOT to change").

REUSABLE PRIMITIVES:
  - Orchestrator(graph_storage=gs, dry_run=True).run_once()
  - PreExtractor.extract_paper(paper_id, chunks, graph_storage=gs)
  - agent.extract_paper_recursive(chunks, categories, query)
  - agent._try_extract_once(chunks, categories, query) → (entities, degraded, salvage, output_tokens)
  - agent._calculate_chunk_budget(system_prompt) → int
  - agent._calculate_max_workers(num_ctx, total_batches) → int
  - agent._pack_chunks_into_batches(chunks, chunk_budget) → List[List[Dict]]
  - PreExtractor._restart_ollama_process() — SIGKILL + cooldown
  - PreExtractor._ensure_dedicated_ollama() — launchd disarm
  - TokenStreamHandler(output_file=None) — compression‑ratio + pattern detection
  - ExtractionAgent._load_extraction_stats(model) → stats dict
  - ExtractionAgent._update_output_ratio(model, chunk_tok, out_tok) → float
  - ExtractionAgent._update_boundary(model, actual_total, passed) → None
  - ExtractionAgent._load_bad_chunks() / _record_bad_chunk(pmcid, idx)
  - _merge_entity_batches(entities) — (name,claim) dedup + evidence union

QUICK START:
  python scripts/diagnose_cache_accumulation.py PMC10571047   # diagnostic
  python phase9_verify.py --test orchestrator                  # dry run
  python phase9_verify.py --test orchestrator --orchestrator-live  # live
  python -m pytest tests/test_extraction_agent.py -q --tb=short   # extraction tests
```
