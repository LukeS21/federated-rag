# Phase 10.5 → 11 Handoff — 17 May 2026 (boundary calibration, ratio-based repack, data-quality exemption, Terminal worker windows, OLLAMA_NUM_PARALLEL passthrough)

## Quick start

```bash
# Fast unit tests (no LLM, ~6s)
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

# Reset calibration to fresh defaults (if stale stats persist)
rm projects/default/extraction_stats.json
```

---

## Current project state

**Pulsed‑wave parallel extraction with self‑calibrating boundary, ratio‑based repack, and data‑quality exemption is operational.** The system uses token‑budgeted chunk packing with a self‑learning boundary derived from pass/fail data. Between waves, the queue repacks using the updated budget so ratio improvements actually increase batch density. Data‑quality degradations (depth‑0, ≤3 chunks) are excluded from calibration to prevent noise. Calibration logs appear in real‑time on the console.

41 extraction‑related tests pass, zero failures. Per‑worker LLM output is written to `logs/extraction/wave_NNN_*.txt` files with per‑wave Terminal windows that auto‑open/close. The daemon pipeline runs end‑to‑end with process‑level GPU memory hygiene on macOS (SIGKILL + cooldown + orphan cleanup). `OLLAMA_NUM_PARALLEL` is explicitly passed to the `ollama serve` subprocess to guarantee the server respects the parallelism cap.

**Knowledge graph** (as of last live cycle): ~3,810 nodes, ~262K edges. BM25 corpus: 27K+ documents. 43+ papers ingested.

---

## What changed this session

| | Before this session | After this session |
|---|---|---|
| **Initial boundary** | `boundary_lower=2500` → budget ~737 → 60 batches for 100 chunks (chunks averaging 900 tok each). Calibrated on tiny test chunks (~10 tok). | `boundary_lower=8000` → budget ~4220 → ~20 batches. Calibrated to real biomedical chunk density (~900 tok/chunk). Self‑calibration climbs from there. |
| **Boundary gap** | `boundary_lower` could exceed `boundary_upper` (`lower=29057 > upper=15941`) — no clamp. A pass at 29K from clean data proved the model could handle it, but a degradation at 16K from data‑quality issues created a contradictory state. | **Symmetric clamp**: PASS raises upper to match if lower exceeded it. DEGRADE lowers lower to match if upper dropped below it. The gap can never go negative. |
| **Data‑quality pollution** | Depth‑0 degradations (E‑L‑E‑V‑E‑N spam from corrupted chunks, `"of"`×100 word‑spam) updated `boundary_upper` and the output‑ratio EMA. Bad data skewed calibration. | **Data‑quality exemption**: depth‑0 AND n≤3 degradations are skipped for boundary updates AND ratio accumulation. The batches are small enough that context overflow is impossible — the degradation is from chunk content, not prompt length. |
| **Ratio‑based repack** | Budget was recomputed after each wave but never used. The initial batch packing was permanent; ratio improvements between waves had no effect. | **Live repack**: after each wave, fresh (depth‑0) batches in the queue are unpacked and re‑packed using the updated budget. Degraded sub‑batches are preserved as‑is (re‑merging could recreate the failure). Each wave's ratio/boundary improvements increase batch density. |
| **Calibration visibility** | Boundary and ratio updates happened silently. No console feedback on whether self‑calibration was working. | **Three realtime log lines**: (1) `Boundary (PASS/DEGRADE): model=X, lower A→B, upper=C, gap=D` — fires per batch. (2) `Output ratio: model=X, 0.500→0.483` — fires per wave when ratio changes. (3) Wave summary includes `(boundary=[L,U], ratio=R)` snapshot. |
| **OLLAMA_NUM_PARALLEL** | Read by Python code but never explicitly passed to the `ollama serve` subprocess. On macOS with `start_new_session=True`, launchctl‑scoped env vars could be lost. Ollama server defaulted to parallelism=1. | Both `_ensure_dedicated_ollama` and `_restart_ollama_process` now copy the parent env and `setdefault("OLLAMA_NUM_PARALLEL", "4")` and `setdefault("OLLAMA_MAX_QUEUE", "8")` before spawning `ollama serve` via `env=env`. Guarantees the server sees the parallelism setting. |
| **Worker output display** | Per‑worker log files with `tail -f` instructions printed to console. User had to manually open Terminal windows. | **Per‑wave Terminal windows**: at wave start (after log files open), `osascript` spawns N Terminal.app windows, each named `Wave{W} Chunk{X} ({N}ch)` and running `tail -f` on the worker log. At next wave start, all "Wave"‑named windows are closed. Clean lifecycle — no markers, no polling, no temp‑file pollution. |
| **Stale calibration** | `extraction_stats.json` from the prior 60‑batch run had an inflated `ratio=1.115` burned in by the EMA. Budgets were artificially constricted even after the boundary was fixed. | Stats file deleted to reset to fresh defaults. The system now starts each model from `boundary_lower=8000`, `ratio=0.50` and self‑calibrates clean from there. |
| **Community routing status** | HANDOFF listed `community_summarizer.py` and `relevance_router.py` as "not wired." | **Correction**: both ARE wired into `survey_community_route_node` in `survey_nodes.py` and tested in `test_phase11_integration.py`. Only `progressive_disclosure.py` remains unwired. |

---

## What was accomplished

### 1. Boundary calibration for real‑world chunks

**Problem:** `boundary_lower=2500` was calibrated using test chunks of ~10 tokens (`"Short chunk."`, `"Chunk number 5 with some extra words..."`). Real biomedical chunks average ~890 tokens (PMC10571047). The formula produced `budget ≈ 737`, fitting 1‑2 chunks per batch → 60 batches for 100 chunks. The HANDOFF claimed "~8 chunks/batch" but this was only true for test data.

**Solution:** Raised `boundary_lower` default from 2500 → 8000. At 8000: `safe_total = 7600, available = 7600−919−350 = 6331, budget = 6331/1.50 ≈ 4220`. With ~890‑tok chunks: ~4‑5 chunks/batch → ~20 batches. The self‑calibration pushes higher from there (observed budget reaching 17453 tok/batch by wave 5 after passes at 59‑chunk batches).

**Tradeoff:** Starting at 8000 is slightly more aggressive than the 2500 "safe floor" but is grounded in empirical data from live extraction (59‑chunk batches passing clean). If a future model has a tighter context window, the degradation → boundary_upper drop → repack cycle handles it.

### 2. Symmetric boundary clamp

**Problem:** A pass at 29057 total tokens (59‑chunk batch) raised `boundary_lower` to 29057. But `boundary_upper` was stuck at 15941 from a prior data‑quality degradation on a 6‑chunk batch. The gap was −13116 — contradictory state where the "safe" floor exceeded the "unsafe" ceiling. This allowed budget computation to climb unchecked.

**Solution:** After every `boundary_lower` or `boundary_upper` update, enforce `lower ≤ upper`:
- PASS with `lower > upper`: raise `upper` to match (the pass proves it's safe)
- DEGRADE with `upper < lower`: lower `lower` to match (the degradation proves it's unsafe)

This keeps the gap non‑negative and the calibration internally consistent.

### 3. Data‑quality degradation exemption

**Problem:** Corrupted chunks (figure captions with `P < 0.01`, copyright markers → E‑L‑E‑V‑E‑N spam) and dense 6‑chunk batches with word‑spam (`"of"` ×100) triggered boundary DEGRADE updates. These were NOT context‑overflow failures — at budgets of 7K‑17K tokens with 1‑6 chunks, total context was well within gemma4:e4b's ~6K effective limit. The degradation was from chunk content, not prompt size.

**Solution:** Degradations at `depth=0 AND n≤3` are classified as **data‑quality flukes**. They are excluded from:
- Boundary updates (`_update_boundary` not called)
- Ratio accumulation (`ct` and `output_tokens` not added to wave totals)

The batches still get split and re‑queued (so the system still processes them), but calibration only learns from real context‑limit events. Safety net: if misclassified, the re‑queued sub‑batches (depth > 0) properly calibrate on their next attempt.

### 4. Ratio‑based queue repack

**Problem:** `chunk_budget` was recomputed after each wave (`line 779`) but never consumed. The initial packing was permanent. Ratio improvements between waves (e.g., ratio dropping from 1.115 → 0.507 as the EMA converged) had zero effect on batch density during the current extraction.

**Solution:** After `chunk_budget` recomputation, separate the queue into fresh batches (depth=0) and degraded sub‑batches (depth>0). Unpack all fresh batch chunks, repack them using the updated budget, and reconstruct the queue. Degraded sub‑batches are NOT repacked — re‑merging could recreate the degradation. A `Repacked N chunks → M batches` log line confirms the repack.

**Result:** Observed in diagnostic run — 12 initial batches compressed to 2 final batches (16‑chunk + 59‑chunk) as ratio improved from 0.692 → 0.507. The system literally packed more chunks per batch as the model proved it could handle them.

### 5. Real‑time calibration logging

**Problem:** No console visibility into whether self‑calibration was functioning. Boundary and ratio updates were silent. Users couldn't tell if passes were raising the boundary or if the ratio was converging.

**Solution:** Three INFO‑level log additions:
- `_update_boundary`: fires `Boundary (PASS/DEGRADE): model=X, lower A→B, upper=C, gap=D` when values actually change. Silent on no‑ops.
- `_update_output_ratio`: fires `Output ratio: model=X, 0.500→0.483` when change exceeds 0.001.
- Wave summary: appended `(boundary=[L, U], ratio=R)` to the existing `Wave N: X/Y passed...` line.

These emit on every wave cycle, so calibration convergence is visible in real‑time.

### 6. OLLAMA_NUM_PARALLEL explicit passthrough

**Problem:** Ollama's server‑side parallelism cap (`OLLAMA_NUM_PARALLEL`) was never explicitly passed to the `ollama serve` subprocess spawned by `_ensure_dedicated_ollama` and `_restart_ollama_process`. On macOS with `start_new_session=True`, launchctl‑scoped env vars may not propagate to the child process. The Ollama server defaulted to `OLLAMA_NUM_PARALLEL=1`, meaning parallel extraction requests were queued and executed sequentially despite the Python code spawning `ThreadPoolExecutor` with 2 workers.

**Solution:** Both `subprocess.Popen(["ollama", "serve"], ...)` calls now copy `os.environ` and `setdefault("OLLAMA_NUM_PARALLEL", "4")` / `setdefault("OLLAMA_MAX_QUEUE", "8")` with `env=env` explicit passthrough. The Ollama server is guaranteed to see the parallelism settings regardless of how the parent Python process was started (shell, launchctl, or IDE).

**Model weights are shared** across parallel requests — only KV caches are duplicated (~1 GB/request for gemma4:e4b at 16K context). 2 concurrent workers = 10.5 GB model + 2×1 GB KV ≈ 12.5 GB, well within 36 GB M3 Max.

### 7. Per‑wave Terminal windows

**Problem:** Per‑worker log files required the user to manually copy/paste `tail -f` paths from the console into separate Terminal windows each wave. Tedious and error‑prone.

**Solution:** Each wave, after log files are opened:
1. **Close old windows**: `osascript` tells Terminal.app to close all windows with "Wave" in their name (safe — normal user windows are untouched).
2. **Open new windows**: For each worker, a tiny shell script is written to `/tmp/opencode_wave{W}_{idx}.sh` containing `printf '\033]0;Wave{W} Chunk{X} ({N}ch)\007' && tail -f /absolute/path/to/log.txt`. `osascript` tells Terminal.app to run the script. The ANSI escape sets the window title; the terminal window shows live output.
3. **Extraction end**: Remaining "Wave" windows are cleaned up.

No completion markers, no polling, no temp‑file accumulation (overwritten each wave). The window lifecycle maps 1:1 to the wave lifecycle. If Python crashes, windows stay open as forensics.

---

## Lessons learned

### 7. Test‑data calibration produces dangerously wrong initial values

`boundary_lower=2500` was "calibrated so the first wave produces ~8‑chunk batches." This was true for test chunks (`"Short chunk."` ≈ 4 tokens) but catastrophic for real biomedical chunks (~890 tokens). The initial budget of 737 produced 60 single‑chunk batches instead of the expected 13.

**Lesson:** Calibration constants must be back‑tested against real‑world data. If test data is unrealistic (as it often is for token‑counting tests), document the discrepancy and use real‑world averages for defaults. The system self‑calibrates upward, but starting too low causes excessive fragmentation on the first several papers.

### 8. `OLLAMA_NUM_PARALLEL` is server‑side and must be explicitly propagated

The Python code reading `os.getenv("OLLAMA_NUM_PARALLEL")` only controls the client‑side worker cap. The actual Ollama server parallelism is an independent setting. On macOS with `start_new_session=True`, launchctl‑scoped env vars may not propagate. The only reliable approach is `env=os.environ.copy()` with explicit `setdefault()` in the `subprocess.Popen` call.

### 9. Calibration feedback loops need both directions

The initial design only used budget recomputation for future waves (which was dead code — the initial packing was permanent). Adding ratio‑based repack between waves completes the feedback loop: the system measures actual output/chunk ratio, adjusts the budget, and repacks accordingly. Both ratio increases AND decreases are handled — the repack works symmetrically.

### 10. Data‑quality failures are categorically different from context‑limit failures

A degradation on a 1‑chunk batch at a 7K‑token budget (E‑L‑E‑V‑E‑N spam from a figure caption) is NOT a context‑overflow event. The total context (~5K tokens) is well within the model's effective window. Updating `boundary_upper` from this event feeds noise into the calibration. The `depth=0, n≤3` heuristic cleanly separates data‑quality problems from real context limits. Degraded sub‑batches (depth>0) are never excluded — they always calibrate.

### 11. Model‑key mismatch causes silent calibration fragmentation

The diagnostic script hardcodes `ExtractionAgent(model="deepseek-chat")` even when running Ollama with `gemma4:e4b`. This means extraction stats are keyed to the wrong model name. The daemon may use yet another key. Each code path tracks separate calibrations that never cross‑pollinate. Deleting `extraction_stats.json` resets all keys to defaults. Future work should unify the model key to match the actual Ollama model in use.

---

## Identified gaps and status

### Closed this session

| # | Gap | Severity | Status |
|---|------|----------|--------|
| — | `boundary_lower=2500` producing 60 batches instead of 13 | ~~High~~ | ✅ Closed — raised to 8000; calibrated to real chunk density (~890 tok/chunk) |
| — | `OLLAMA_NUM_PARALLEL` not reaching Ollama server | ~~High~~ | ✅ Closed — explicit `env=env` in both Popen calls with `setdefault("OLLAMA_NUM_PARALLEL", "4")` |
| — | Ratio‑based repack not wired (dead budget recomputation) | ~~Medium~~ | ✅ Closed — fresh batches repacked between waves using updated budget |
| — | Calibration progress invisible (no console feedback) | ~~Low~~ | ✅ Closed — three INFO‑level log additions: boundary updates, ratio updates, wave snapshots |
| — | Data‑quality degradations polluting boundary + ratio | ~~Medium~~ | ✅ Closed — depth‑0/n≤3 exemption skips boundary update and ratio accumulation |
| — | `boundary_lower > boundary_upper` negative gap | ~~Medium~~ | ✅ Closed — symmetric clamp keeps lower ≤ upper |
| — | Per‑worker Terminal windows | ~~Low~~ | ✅ Closed — osascript‑driven, wave‑lifecycle‑tied windows with `tail -f` |
| — | `extraction_stats.json` stale ratio (1.115 from 60‑batch run) | ~~Low~~ | ✅ Closed — stats file deleted; fresh defaults (8000, 0.50) |

### Open (this session)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| A | ProgressiveDisclosure not wired | High | `progressive_disclosure.py` fully built + tested (10 unit, 5 integration) but never instantiated in production. Needs wiring into `survey_community_route_node` (`survey_nodes.py:298`) where all required data (`community_data`, `community_summaries`, `relevant_communities`) is already computed. Then replace ad‑hoc community section in `survey_scrub_node` (~line 1089) with `disclosure_map["tier1_system_overview"]`. Tiered context access for Survey Mode synthesis. |
| B | SPECTER2 `paper_similarity_search()` not built | Medium | `spector2_cache.json` has 768‑dim embeddings for 8 papers but no consumer. Need `Spector2Cache.find_similar(doi, min_score=0.6)` — cosine‑similarity‑ranked paper discovery. Graceful degradation (empty list if DOI not cached). Configurable threshold via `SPECTOR2_SIMILARITY_THRESHOLD` env var. Wiring into orchestrator's discovery loop is a follow‑up. |
| C | No ≥8 h continuous daemon validation | Medium | Daemon has run short cycles but never >8 h. Longer runs needed to validate memory stability with the pulsed‑wave cooldown, parallel extraction, and boundary convergence over multiple papers. |
| D | Model‑key mismatch in extraction stats | Low | Diagnostic script uses `model="deepseek-chat"` but Ollama runs `gemma4:e4b`. Multiple code paths may use different keys, fragmenting calibration. Needs investigation and unification. |

### Evergreen (inherent hardware/architecture limitations)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| E | `/api/ps` cannot verify true GPU memory state | High | No macOS API exposes Metal buffer state. Process death (SIGKILL) is the strongest guarantee. The 5 s cooldown is a heuristic; no software can prove pages were deallocated. |
| F | `stream.close()` abort reliability untested at scale | Low | The `finally` block guarantees `.close()` is called, but the httpx cleanup depends on LangChain's internal stream handling. Edge cases (hung Ollama, network timeout during generator exit) not tested. No known failures. |

### Phase 11 (unchanged layout — status annotation only)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| G | Community summaries | ~~High~~ → ✅ Wired | `community_summarizer.py` is wired into `survey_community_route_node` (`survey_nodes.py:281`) and tested. Prior HANDOFF was stale. |
| H | Relevance router | ~~High~~ → ✅ Wired | `relevance_router.py` is wired into `survey_community_route_node` (`survey_nodes.py:291`) and tested. Prior HANDOFF was stale. |
| I | Progressive disclosure not integrated | High | See Gap A above — fully built + tested, not wired. Last Phase 11 code gap. |
| J | SPECTER2 embeddings unused | Medium | See Gap B above — cache exists, no consumer function. |

---

## Key architectural decisions (DO NOT UNDO)

### Carried forward from prior sessions

All Phase 4–10.5 constraints still apply. See README §17.

### New decisions (this session)

- **`boundary_lower=8000` over `2500`** — Calibrated to real biomedical chunks (~890 tok/chunk), not tiny test data. Produces ~20 batches for 100 chunks instead of 60. The self‑calibration pushes higher from real passes. Backed by observed 59‑chunk batch passing clean at 29K total context.
- **Symmetric boundary clamp** — After every PASS/DEGRADE update, enforce `lower ≤ upper`. PASS raises upper to match; DEGRADE lowers lower to match. The gap can never go negative. Prevents contradictory calibration states (`lower=29057 > upper=15941`).
- **Data‑quality degradation exemption (depth=0, n≤3)** — Small‑batch degradations that can't be context‑overflow (well within the model's effective window) are excluded from boundary AND ratio calibration. The batch still gets split and re‑queued. Genuine context‑overflow degradations (depth>0 or n>3) always calibrate.
- **Ratio‑based queue repack between waves** — Fresh (depth‑0) batches in the queue are repacked with the updated budget after each wave. Degraded sub‑batches are preserved as‑is. The initial packing is no longer permanent; ratio improvements increase batch density mid‑extraction.
- **Realtime calibration console logging** — Three INFO‑level additions: boundary updates, ratio updates, and wave‑summary calibration snapshots. Calibration progress is visible without tailing debug logs.
- **Explicit `OLLAMA_NUM_PARALLEL` + `OLLAMA_MAX_QUEUE` in `ollama serve` subprocess** — `env=os.environ.copy()` with `setdefault()` guarantees the server respects the parallelism cap regardless of how the parent process was launched.
- **Per‑wave Terminal.app windows tied to wave lifecycle** — `osascript` opens `tail -f` windows at wave start, closes them at next wave start. Uses ANSI escape window titles filtered by "Wave" pattern — safe, no user windows affected. Shell scripts written to `/tmp/` to avoid Python→AppleScript→bash escaping ambiguity.
- **Stats reset on calibration mismatch** — Deleting `extraction_stats.json` resets all model keys to fresh defaults (`boundary_lower=8000, ratio=0.50`). Recommended when migrating from mis‑calibrated prior runs.

---

## What NOT to change

All prior constraints apply. Additions from this session:

### Boundary & calibration
- Do NOT lower `boundary_lower` back to 2500 — it was calibrated on unrealistically small test chunks and produces excessive fragmentation on real biomedical papers.
- Do NOT remove the symmetric clamp (`if lower > upper: ...`) — it prevents contradictory calibration states.
- Do NOT remove the data‑quality exemption (`depth==0 and n<=3` for degradation boundary/ratio skip) — bad data must not pollute the calibration.
- Do NOT remove the ratio‑based queue repack between waves — it makes calibration improvements actionable during extraction.
- Do NOT remove the realtime calibration logging in `_update_boundary`, `_update_output_ratio`, or the wave summary line — it's the only visibility into self‑calibration.
- Do NOT remove `extraction_stats.json` persistence — it's the calibration memory across runs.

### Ollama parallelism
- Do NOT remove the `env=env` passthrough in `_ensure_dedicated_ollama` or `_restart_ollama_process` — it's the only guarantee `OLLAMA_NUM_PARALLEL` reaches the server.
- Do NOT hardcode `OLLAMA_NUM_PARALLEL` — always use `setdefault("OLLAMA_NUM_PARALLEL", "4")` so the parent env takes precedence.

### Worker display
- Do NOT remove the per‑wave Terminal window spawning — it provides live per‑worker output without console bloat.
- Do NOT change the "Wave" window‑name filter — it's the only safety mechanism preventing accidental closure of user windows.

### All prior constraints
- Do NOT reinstate `_extract_batch_recursive` or `_merge_entity_dicts`.
- Do NOT add `batch_size` back as a parameter — batch sizing is token‑driven.
- Do NOT derive chunk budget from `num_ctx` — use the self‑calibrating boundary formula.
- Do NOT remove `_update_boundary`, `_update_output_ratio`, `_pack_chunks_into_batches`, `_try_extract_once`, `_calculate_max_workers`, `_merge_entity_batches`, per‑worker log files, compression‑ratio detection, base‑case boundary exclusion, `bad_chunks.json` pre‑emption, or any GPU‑memory‑management method (`_reset_ollama`, `_restart_ollama_process`, `_ensure_dedicated_ollama`, `_find_and_kill_ollama`, orphan cleanup, cooldown).
- Do NOT switch extraction back to JSON or full‑prompt. Do NOT reinstate two‑pass extraction or cross‑chunk prompts.
- All prior constraints: per‑paper source prefixes, `chunk_index`, no `lstrip()`, no `Accept: application/json` on EPMC session, etc.

---

## File map

```
MODIFIED FILES (this session):
src/agents/extraction_agent.py        — + boundary_lower 2500→8000
                                         + symmetric clamp in _update_boundary
                                         + data‑quality exemption (depth=0, n≤3)
                                         + ratio‑based queue repack between waves
                                         + realtime calibration INFO logging
                                         + per‑wave Terminal window spawning (osascript)
                                         + subprocess + shlex imports
                                         + wave‑start window cleanup
                                         + extraction‑end window cleanup
src/ingestion/pre_extractor.py        — + env=os.environ.copy() with setdefault
                                         (OLLAMA_NUM_PARALLEL, OLLAMA_MAX_QUEUE)
                                         in _ensure_dedicated_ollama Popen
                                         + same in _restart_ollama_process Popen
tests/test_extraction_agent.py        — Updated 6 tests for boundary_lower=8000:
                                         test_calculate_chunk_budget_positive
                                         test_boundary_defaults
                                         test_boundary_update_pass_raises_lower
                                         test_boundary_update_degrade_lowers_upper
                                         test_calculate_chunk_budget_calibrated
                                         test_boundary_persistence_survives_ratio_update
                                         (new clamp behavior documented in assertions)
.env                                  — OLLAMA_NUM_PARALLEL + OLLAMA_MAX_QUEUE
                                         documented as server‑side; passed via env= fix
projects/default/extraction_stats.json — DELETED to reset stale ratio=1.115
                                         (fresh defaults: boundary=8000, ratio=0.50)
HANDOFF.md                            — This file — comprehensive session handoff

UNMODIFIED BUT RELEVANT (existing Phase 11 components):
src/graph/progressive_disclosure.py   — Fully built + tested, NOT wired (Gap A)
src/utils/spector2_cache.py           — 8 papers cached, no find_similar() (Gap B)
src/graph/survey_nodes.py             — survey_community_route_node already has
                                         community_data, summaries, routing (lines 248-333)
src/state.py                          — Has Phase 11 state fields, needs disclosure_map
```

---

## Recommendations

### Immediate — wire ProgressiveDisclosure (Gap A)

3‑file change, ~30 lines of code:

1. `src/state.py` (~line 54): add `disclosure_map: NotRequired[Dict]`
2. `src/graph/survey_nodes.py` (~line 298, inside `survey_community_route_node`): after routing, instantiate `ProgressiveDisclosure(graph_storage, community_data, summaries)` and call `build_disclosure_map(relevant_communities=relevant, query=query)`, store in `updates["disclosure_map"]`
3. `src/graph/survey_nodes.py` (~lines 1089‑1101, inside `survey_scrub_node`): replace the ad‑hoc `# RESEARCH COMMUNITIES` section builder with `disclosure_map.get("tier1_system_overview", "")`

Zero new tests needed — `ProgressiveDisclosure` already has 10 unit + 5 integration tests. Validate with:
```bash
python -m pytest tests/test_progressive_disclosure.py tests/test_phase11_integration.py -q --tb=short
```

### Short‑term — build SPECTER2 paper_similarity_search (Gap B)

2‑file change, ~30 lines:

1. `src/utils/spector2_cache.py`: add `find_similar(self, doi: str, *, min_score: float = None) → List[Dict]`. Default `min_score` from `SPECTOR2_SIMILARITY_THRESHOLD` env var (default 0.6). Computes cosine similarity via `numpy` (already available via `sentence‑transformers` dep). Returns `[{doi, s2_paper_id, score}, ...]` sorted descending, filtered to `score ≥ min_score`. Returns `[]` if DOI not cached or no matches.

2. `tests/test_spector2_cache.py`: add 3 tests — returns results, DOI not cached, respects threshold.

### Short‑term — run diagnostic to verify calibration

```bash
python scripts/diagnose_cache_accumulation.py PMC10571047
```

Watch for:
- `Packed 100 chunks → ~20 batches (budget=~4220 tok/batch)` at wave 1
- `Repacked N chunks → M batches` between waves (ratio‑based repacking)
- `Boundary (PASS): model=deepseek-chat, lower 8000→N, upper=16384, gap=M` after each clean batch
- `Output ratio: model=deepseek-chat, 0.500→0.4xx` after each wave
- Terminal windows opening/closing with live output
- No `boundary=[29057, 15941]` negative‑gap states

### Medium‑term — 8h daemon validation (Gap C)

Run the daemon for ≥8 h continuous. Monitor:
- `projects/default/extraction_stats.json` for boundary convergence per model
- Console for calibration log progression
- Activity Monitor for GPU memory pressure between waves
- `projects/default/bad_chunks.json` for accumulating known‑bad chunks
- No OOM, no orphaned GPU runners, no memory creep

### Beyond — model‑key unification (Gap D)

Investigate and unify the model key used by extraction stats across diagnostic scripts, daemon, and survey mode. All should use the actual Ollama model name (`gemma4:e4b`) rather than hardcoded strings (`deepseek-chat`).

### Phase 12 — Skills & Experiential Memory

After Phase 11 is fully closed (ProgressiveDisclosure wired + SPECTER2 built), move to Phase 12:
- Skill library from agent trajectories
- JSONL trajectory logging
- Experiential memory system
- A/B skill evaluation before deployment

---

## Prompt for next AI session

```
You are an expert senior software developer continuing the Federated RAG
system for biomedical research.  Pulsed‑wave parallel extraction with
self‑calibrating boundary, ratio‑based repack, data‑quality exemption,
and per‑wave Terminal windows is complete.  Phase 11 (community routing
& memory cascade) has one remaining code gap — ProgressiveDisclosure
wiring — plus a SPECTER2 paper_similarity_search function to build.

Read the full README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - 41 extraction‑related tests pass, zero failures.
  - Extraction uses pulsed‑wave parallel design: token‑budgeted chunk
    packing (tiktoken per‑chunk), GPU restart per wave, parallel workers
    within each wave (up to OLLAMA_NUM_PARALLEL, memory‑aware cap).
    Per‑worker output written to logs/extraction/wave_NNN_*.txt files
    with per‑wave Terminal.app windows (auto‑open/close via osascript).
  - Batch budget self‑calibrates: (boundary_lower × 0.95 − system −
    overhead) / (1 + output_ratio).  boundary_lower starts at 8000
    (~20 batches for 100 chunks at ~900 tok/chunk), rises from pass data.
    boundary_upper starts at 16384, falls from real (non‑data‑quality)
    degradation data.  Symmetric clamp prevents negative gap.
  - Data‑quality exemption: depth‑0 AND n≤3 degradations skip boundary
    AND ratio updates — these are chunk‑content problems, not context
    overflow.  Sub‑batches still split and re‑queued.
  - Ratio‑based repack: after each wave, fresh (depth‑0) batches in the
    queue are repacked with the updated budget.  Degraded sub‑batches
    preserved as‑is.  Ratio improvements increase batch density mid‑run.
  - Realtime calibration logging: Boundary (PASS/DEGRADE), Output ratio,
    and wave‑summary boundary/ratio snapshots emitted INFO‑level.
  - OLLAMA_NUM_PARALLEL explicitly passed to ollama serve subprocess via
    env=os.environ.copy() with setdefault().  Server respects the cap.
  - Output ratio (chunk_tokens → output_tokens) tracked via weighted
    EMA (80/20), persisted per‑model, updated every wave.
  - Degradation detection: word‑level, hyphen‑level, junk‑line, AND
    universal compression‑ratio (zlib, ≥8:1 catches any repetition).
    stream.close() in finally guarantees httpx disconnect on abort.
  - Bad chunks (≥3 base‑case failures) tracked in
    projects/default/bad_chunks.json, automatically isolated.
  - Worker count uses correct KV‑cache formula; gemma4:e4b = 10.5 GB,
    qwen3.6:35b = 25 GB.  Qwen auto‑capped at 1 worker.
  - Orchestrator daemon runs full autonomous cycle: web discovery → EPMC
    fetch → batch ingest → pulsed‑wave extraction → KG save → community
    detection → cycle handoff.  Self‑bootstraps Ollama via launchd disarm.
  - KG: ~3,810 nodes, ~262K edges.  BM25: 27K+ documents.  43+ papers.
  - Extraction uses evidence‑grouped line‑tagged format with CLAIM
    semantics.  Entities deduped by (name, claim) with evidence union.
  - Community routing: detection (wired), summarizer (wired),
    relevance router (wired), progressive disclosure (NOT wired).
  - SPECTER2 cache: 8 papers with 768‑dim embeddings.  No consumer
    function (paper_similarity_search not built).

CRITICAL OPEN:
  - Wire ProgressiveDisclosure into survey_community_route_node (3 files,
    ~30 lines).  See HANDOFF § "Recommendations → Immediate."
  - Build SPECTER2 paper_similarity_search (2 files, ~30 lines).  See
    HANDOFF § "Recommendations → Short‑term."
  - Long‑running daemon validation (≥8 h).

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE — see README §17 + HANDOFF §"What NOT to change"):
  - Do NOT lower boundary_lower below 8000.
  - Do NOT remove symmetric clamp, data‑quality exemption, ratio‑based
    repack, or realtime calibration logging.
  - Do NOT remove the env=env OLLAMA_NUM_PARALLEL passthrough.
  - Do NOT remove per‑wave Terminal windows.
  - Do NOT remove or disable any detection layer.
  - Do NOT add back _extract_batch_recursive or _merge_entity_dicts.
  - Do NOT derive budget from num_ctx — use the boundary formula.
  - All prior constraints still apply.

REUSABLE PRIMITIVES:
  - Orchestrator(graph_storage=gs, dry_run=True).run_once()
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
  - ProgressiveDisclosure(graph_storage, community_data, summaries)
    .build_disclosure_map(relevant_communities, query) → Dict
  - Spector2Cache() — .get(doi), .put(doi, s2_id, emb), .stats()

QUICK START:
  python scripts/diagnose_cache_accumulation.py PMC10571047   # diagnostic
  python phase9_verify.py --test orchestrator                  # dry run
  python phase9_verify.py --test orchestrator --orchestrator-live  # live
  python -m pytest tests/test_extraction_agent.py -q --tb=short   # extraction tests
  python -m pytest tests/test_progressive_disclosure.py tests/test_phase11_integration.py tests/test_spector2_cache.py -q
```
