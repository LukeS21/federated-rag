# Phase 11 → 12 Handoff — 18 May 2026 (ProgressiveDisclosure wired, SPECTER2 built, Planner/Reflector/Executor architecture designed)

## Quick start

```bash
# Fast unit tests — extraction (no LLM, ~6s)
python -m pytest tests/test_extraction_agent.py -q --tb=short

# Progressive disclosure + SPECTER2 + community routing tests
python -m pytest tests/test_progressive_disclosure.py tests/test_phase11_integration.py tests/test_spector2_cache.py -q --tb=short

# Diagnostic: test extraction on a real paper (live Ollama)
python scripts/diagnose_cache_accumulation.py PMC10571047

# Full daemon cycle (live — self‑bootstraps Ollama via launchd disarm)
python phase9_verify.py --test orchestrator --orchestrator-live

# Dry run (see what WOULD happen, no API spend)
python phase9_verify.py --test orchestrator

# SPECTER2 similarity search (new)
python -c "
from src.utils.spector2_cache import Spector2Cache
c = Spector2Cache()
print(c.find_similar('10.1016/j.bioactmat.2021.01.030', min_score=0.5))
"

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

**Phase 11 is complete.** ProgressiveDisclosure is wired into production Survey Mode. SPECTER2 `paper_similarity_search` is built with 3 tests passing. All previously documented extraction hardening (pulsed‑wave parallel extraction, self‑calibrating boundary, ratio‑based repack, data‑quality exemption, per‑wave Terminal windows, OLLAMA_NUM_PARALLEL passthrough) remains operational.

**87 tests pass, zero failures:** 41 extraction‑related + 10 ProgressiveDisclosure unit + 5 ProgressiveDisclosure integration + 14 SPECTER2 cache (11 existing + 3 new) + 10 community routing integration + 7 orchestrator‑integration.

**This session was also a major architectural redesign.** We designed the Planner/Reflector/Executor 3‑role cognitive architecture for the orchestrator — the blueprint for long‑running autonomous biomedical research. This includes tiered planning (strategic → tactical → execution), plan trees with revision tracking, the handoff box as a memory bridge across GPU‑reset cycles, per‑task‑type boundary calibration (generalizing the extraction pattern), dependency gates to prevent writing on incomplete knowledge, circuit breakers with natural‑language health alerts, map‑reduce decomposition for large‑context tasks, a self‑tuning staleness detector, and a meta‑cognition layer (Reflector) that audits plans against the KG before execution.

**Knowledge graph** (from prior live cycles): ~3,810 nodes, ~262K edges. BM25 corpus: 27K+ documents. 43+ papers ingested.

---

## What changed this session

| | Before this session | After this session |
|---|---|---|
| **ProgressiveDisclosure** | Fully built + tested (10 unit, 5 integration) but never instantiated in production. `survey_scrub_node` built an ad‑hoc community section from raw state data. | Wired into `survey_community_route_node` (`survey_nodes.py:311‑321`). `disclosure_map` stored in state. `survey_scrub_node` consumes `disclosure_map["tier1_system_overview"]`. Three‑tier context access operational in Survey Mode. |
| **SPECTER2 cache** | 8 papers cached with 768‑dim embeddings. No consumer function. Dead data. | `find_similar(doi, min_score=0.6)` method built. Cosine‑similarity‑ranked paper discovery via numpy. Graceful `[]` return on missing DOI. Configurable threshold via `SPECTOR2_SIMILARITY_THRESHOLD` env var. 3 new tests pass. |
| **Cache entry format** | `put()` stored `{s2_paper_id, embedding, fetched_at}`. No DOI field in entry. | `put()` now stores `{doi, s2_paper_id, embedding, fetched_at}`. `find_similar()` returns original DOI (not lowercased key). |
| **Orchestrator architecture** | Single agent model. No explicit planning, reflecting, or meta‑cognition roles. Handoffs were descriptive log files, not prescriptive memory bridges. No tiered planning or calibration beyond extraction. | **3‑role cognitive architecture designed:** Planner (mission + decomposition), Executor (calibrated specialists per task type), Reflector (audit gate). Tiered planning (strategic T3 → tactical T2 → execution T1). Plan trees with revision tracking. Handoff box as memory bridge across GPU‑reset cycles. Per‑task‑type boundary/ratio calibration generalizing the extraction pattern. Dependency gates for write‑on‑complete‑knowledge guarantee. Map‑reduce decomposition for large‑context tasks. Circuit breaker with natural‑language health alerts. Self‑tuning staleness detector. Goal graveyard. |
| **Novel approaches documented** | Per‑task‑type calibration existed only in extraction. No formal architecture for cross‑cycle consciousness. | Per‑task‑type calibration generalized to planner + reflector calls. Handoff box as memory bridge pattern. Plan trees with living revision. Adaptive staleness via plan‑revision count (not cycle count or wall‑time). |

---

## What was accomplished

### Code: ProgressiveDisclosure wired (Gap A)

**3 files changed, ~30 lines.**

1. `src/state.py:55` — added `disclosure_map: NotRequired[Dict]` to `AgentState`
2. `src/graph/survey_nodes.py:311‑321` — after relevance routing in `survey_community_route_node`, instantiates `ProgressiveDisclosure(graph_storage, community_data, summaries)`, calls `build_disclosure_map(relevant_communities=relevant, query=query)`, stores in `updates["disclosure_map"]` with INFO‑level logging. Wrapped in try/except matching existing error patterns.
3. `src/graph/survey_nodes.py:1101‑1107` — replaced the ad‑hoc `# RESEARCH COMMUNITIES` section builder (~13 lines) with `disclosure_map.get("tier1_system_overview", "")`.

**Rationale:** The previous approach read `community_data`, `relevant`, and `community_summaries` directly from state and manually iterated over communities to build a markdown section. This was fragile, duplicated the disclosure logic, and couldn't use the 3‑tier hierarchy already built and tested in `ProgressiveDisclosure`. The wiring leverages existing tested code, reduces duplication, and provides the system overview (Tier 1), community details (Tier 2), and paper‑community mapping (Tier 3) for synthesis prompts.

### Code: SPECTER2 find_similar built (Gap B)

**2 files changed, ~65 lines.**

1. `src/utils/spector2_cache.py:76` — `put()` now stores `"doi": doi` in the cache entry alongside `s2_paper_id`, `embedding`, `fetched_at`. Previously only the lowercased key was stored.
2. `src/utils/spector2_cache.py:86‑130` — new method `find_similar(self, doi, *, min_score=None) → List[Dict]`. Computes cosine similarity between the query paper's 768‑dim embedding and every other cached embedding using numpy. Filters to `score ≥ min_score`. Returns `[{doi, s2_paper_id, score}, ...]` sorted descending. Default `min_score` from `SPECTOR2_SIMILARITY_THRESHOLD` env var (default 0.6). Graceful degradation: returns `[]` if DOI not cached, embedding is zero‑norm, or no matches.
3. `tests/test_spector2_cache.py` — 3 new tests: `test_find_similar_returns_results` (correctly returns similar papers above threshold and not the query paper itself), `test_find_similar_doi_not_cached` (returns empty list), `test_find_similar_respects_threshold` (higher threshold returns fewer results). Imported `numpy as np` at module level.

**Rationale:** SPECTER2 embeddings (768‑dim from Semantic Scholar) provide paper‑level semantic similarity. The cache has 8 papers and grows with each daemon cycle. `find_similar()` is the consumer function — it enables cosine‑similarity‑ranked paper discovery. Wiring into the orchestrator's discovery loop is a follow‑up task: when a new paper is ingested, find the N most similar already‑cached papers and use their community assignments as priors for the new paper's community membership.

### Architecture: Planner/Reflector/Executor 3‑Role Design

This was the bulk of the session — a complete re‑thinking of the orchestrator's cognitive architecture. The following was designed (not yet implemented):

**The Problem:** The current orchestrator executes a fixed pipeline per cycle (web discovery → EPMC → extract → KG → communities). It has no planner, no quality control, no meta‑cognition, and no mechanism for task decomposition when context limits are hit. Large research tasks (write a paper, explore IL‑6 contradictions, generate hypotheses) can't be expressed or executed. The system has no self‑direction.

**The Solution — Three cognitive roles:**

| Role | Owns | Produces | Context |
|------|------|----------|---------|
| **Planner** | The mission, goals, priorities, plan | Decomposed plan tree (phases → goals → steps). Handoff write. | Tier 1–2 KG overview + goals + new data. ~2000–3000 tokens. |
| **Executor** | Individual step execution | Structured output per step. Self‑calibrates per task type. | Task‑specific: chunks (extraction), entities (contradiction), community detail (analysis). Bounded by learned per‑task‑type calibration. |
| **Reflector** | Quality and correctness auditing | Validation flags, staleness alerts, context pressure handling | Plan + executor output + KG for fact‑checking. ~2000 tokens. |

**Key architectural decisions made this session:**

- **Tiered planning, not 3 separate agents.** The planner is ONE LLM that thinks at three different scope levels: Tier 1 (immediate execution, every cycle), Tier 2 (priority rebalancing, when new data arrives), Tier 3 (strategic direction review, every ~10 cycles). Not three separate calls per cycle — typically 1 call per cycle.
- **Plan trees with revision tracking.** Plans are nested (phase → goal → step). Each node tracks its revision ID. The planner can reopen completed phases when new data reveals gaps. Dependency gates prevent Phase 4 (writing) from executing until Phase 2 (gaps) confirms Phase 1 (extraction) is complete.
- **Handoff box as memory bridge.** The planner does NOT have persistent consciousness across cycles (GPU is reset between thinking sessions). Instead, a "handoff box" — a structured JSON/markdown state package containing mission, goals, plan tree, KG overview, calibration data, open questions, and hypotheses — is read at cycle start and written at cycle end. This IS the planner's memory. The GPU reset is sleep; the box is memory.
- **Per‑task‑type calibration generalized from extraction.** The proven extraction pattern (boundary_lower/upper learned from pass/fail, output_ratio EMA, tiktoken‑measured budgets) is applied to ALL LLM calls — planner, reflector, and every executor specialist. Each task type tracks its own calibration in a shared stats file. Pass/fail signals differ by task type (compression ratio for extraction, plan parseability + executor success for planner, planner acceptance of flags for reflector) but the calibration math is identical.
- **Dependency gates — no writing on incomplete knowledge.** A phase can only execute when its dependencies are satisfied. If Phase 2 discovers Phase 1 missed gut microbiome data, Phase 1 reopens, Phase 2 pauses, Phase 4 is blocked. The system never produces output on incomplete knowledge. Honesty: gaps are documented, not papered over.
- **Map‑reduce decomposition — preserve quality, don't truncate.** When a task exceeds the per‑task‑type boundary, it decomposes recursively: map (analyze independently in bounded sub‑contexts) → reduce (synthesize analysis results, which are compressed, into unified output). No truncation, no information loss. Each leaf has full context for its scope. The reduce step sees everything through compressed summaries. Base case is always small enough (e.g., "Is this single entity's claim correct?") that failure indicates a model problem, not a sizing problem.
- **Circuit breaker with natural‑language health alerts.** If ALL task types show declining boundaries over N cycles, the system detects systemic model degradation (Ollama bug, quantization drift). It resets calibration to conservative defaults, recommends model re‑pull, and writes a human‑readable markdown health alert with: what happened, what it means, what was done, what the user should do, and when the system self‑resolves.
- **Self‑tuning staleness.** Goal staleness is measured by plan‑revision count (how many times has the plan been revised since this goal was last updated?), not cycle count or wall‑time. The threshold self‑tunes based on reflector false‑positive/false‑negative rates. Corroboration‑count‑aware: goals with 15 corroborations and no recent updates are "settled science," not "stale."
- **User input as priority zero.** The planner treats live user queries (via Streamlit) as priority‑0 goals that preempt autonomous work. Mode switch: interactive (short steps, fast response) vs autonomous (full cycles when user is idle).
- **"Any experience level user" vision.** The system decomposes high‑level tasks ("write an obesity paper for my lab") into phases, identifies knowledge gaps, discovers literature to fill them, asks clarifying questions only when genuinely needed, and can push back if evidence contradicts the user's assumptions — like a PhD assistant. Plans adapt as new information is discovered.

---

## Lessons learned

### 1. The extraction calibration pattern generalizes to any LLM call

The `boundary_lower/upper` + `output_ratio` EMA + tiktoken measurement system is not extraction‑specific. Any LLM call with measurable input tokens, measurable output tokens, and a binary pass/fail signal can self‑calibrate. The only difference across task types is the pass/fail signal definition — not the calibration math.

### 2. Chunk overlap is a text‑level solution to a domain‑level non‑problem in this architecture

ChromaDB's chunk‑overlap pattern preserves semantic continuity across arbitrary text‑length splits. In our architecture, atomic units (chunks from the PMC parser, entities from extraction, goals in the plan tree) are already semantically complete. The synthesis layer (entity dedup, map‑reduce analysis, plan tree revision) provides cross‑unit continuity. Chunk overlap would add 20% token overhead with negligible benefit.

### 3. Plans are living documents, not scripts

The planner doesn't execute a fixed decomposition from cycle 1. Every cycle, it loads the handoff box and asks: "Given what I now know, is the plan still optimal?" New discoveries trigger replanning. Completed phases can be reopened. Stale goals are archived. The plan tree is the system's cognitive map — revisable, self‑critical, adapting to new evidence.

### 4. The handoff box decouples consciousness from GPU lifetime

The planner doesn't need persistent GPU memory across cycles. Its "memory" is a structured state package in the filesystem — the handoff box. Each cycle starts with a fresh GPU and a full state load. This enables: crash resilience (state survives GPU death), long‑horizon reasoning (memory persists indefinitely), and instance chaining (any future version of the planner can read the box).

### 5. JSON is for the system. Markdown is for the LLM

JSON has ~4:1 token overhead for LLM consumption (keys, brackets, quotes). The canonical store should be JSON (Python reads/writes deterministically). But the LLM should receive markdown (bullet lists, section headers) generated from JSON at prompt‑build time. Dual‑format storage is the efficient pattern.

### 6. Degraded output must not pollute calibration

When an LLM call degrades (truncation, repetition, hallucination), the boundary can be updated (learn what's too much), but the output_ratio MUST NOT be updated with garbage tokens. Only healthy output tokens (up to the degradation point, measured by the detection system) contribute to the ratio EMA. This is already implemented in extraction (data‑quality exemption) — the same rule applies to all task types.

### 7. The Reflector's calibration is confounded if measured by planner acceptance

If the reflector's "pass" signal is "planner accepted my flag," a degraded planner that accepts everything produces misleading reflector calibration. The fix: calibrate the reflector on executor outcome. If the executor succeeds despite the reflector's flag, the reflector was wrong (false positive). If the executor fails and the reflector flagged it, the reflector was right. The ground truth is executor success, not planner agreement.

### 8. No hardcoded thresholds survive real‑world variability

"Analyze top 10 entities," "stale after 5 cycles," "expected 100‑300 tokens for abstract" — all of these fail in real usage. The architecture uses learned calibration (boundary, ratio), adaptive triggers (plan‑revision count, not cycle count), and tiktoken measurement (not LLM‑estimated tokens). The only acceptable hardcoded value is the initial default, which self‑calibrates away.

---

## Identified gaps and status

### Closed this session

| # | Gap | Severity | Status |
|---|------|----------|--------|
| A | ProgressiveDisclosure not wired | ~~High~~ | ✅ Closed — wired into `survey_community_route_node` + `survey_scrub_node`. 3 files, ~30 lines. |
| B | SPECTER2 `paper_similarity_search()` not built | ~~Medium~~ | ✅ Closed — `Spector2Cache.find_similar()` method with 3 tests. 2 files, ~65 lines. |

### Open (implementation — code to build)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| C | Planner not built | High | The planner (tiered decomposition, plan tree management, priority rebalancing, handoff box read/write, dependency gate enforcement) is fully designed but not implemented. |
| D | Reflector not built | High | The reflector (deterministic plan validation via regex/topology/lookup, LLM‑based output quality auditing, staleness detection, context pressure response, circuit breaker triggering) is fully designed but not implemented. |
| E | Handoff box format not built | High | The structured state package (mission, goals, plan tree, calibration, open questions, hypotheses, crash log) needs a concrete JSON + markdown dual‑format specification and read/write library. |
| F | Per‑task‑type calibration for planner/reflector | High | The extraction calibration system needs to be generalized: shared stats file (`orchestration_stats.json`), per‑task‑type keys, and per‑task‑type pass/fail signal definitions applied to planner and reflector LLM calls. |
| G | Map‑reduce decomposition for large‑context execution steps | Medium | When an executor step exceeds its per‑task‑type boundary, it needs automatic decomposition (map into sub‑steps) and synthesis (reduce sub‑results). Design is complete; implementation is not. |
| H | Circuit breaker & health alerts | Medium | System‑level health monitoring across task types. Detects uniform calibration decline (model degradation). Resets to conservative defaults, writes natural‑language health alert. |
| I | Capability registry | Low | Each executor specialist advertises its current boundary, input format, and minimum entity count. Planner reads this before assigning steps. Prevents impossible‑task assignments. |

### Open (validation & infrastructure)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| J | No ≥8 h continuous daemon validation | Medium | Daemon has run short cycles but never >8 h. Longer runs needed to validate memory stability with the pulsed‑wave cooldown, parallel extraction, and boundary convergence over multiple papers. |
| K | Model‑key mismatch in extraction stats | Low | Diagnostic script uses `model="deepseek-chat"` but Ollama runs `gemma4:e4b`. Multiple code paths may use different keys, fragmenting calibration. |
| L | Wave log rotation not implemented | Low | `logs/extraction/wave_*.txt` and `/tmp/opencode_wave*.sh` files accumulate unbounded. Need per‑paper cleanup. |
| M | `atexit` GPU cleanup not implemented | Low | No atexit/signal handler. If Python crashes mid‑extraction, Ollama stays loaded. Launchd disarm is permanent — no re‑arm on exit. |

### Evergreen (inherent hardware/architecture limitations)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| N | `/api/ps` cannot verify true GPU memory state | High | No macOS API exposes Metal buffer state. Process death (SIGKILL) is the strongest guarantee. The 5 s cooldown is a heuristic tuned for DDR5 unified memory at 100 GB/s. |
| O | `stream.close()` abort reliability untested at scale | Low | The `finally` block guarantees `.close()` is called, but the httpx cleanup depends on LangChain's internal stream handling. Edge cases (hung Ollama, network timeout during generator exit) not tested. No known failures. |

---

## Key architectural decisions (DO NOT UNDO)

### Carried forward from prior sessions

All Phase 4–10.5 constraints still apply. See README §17 and prior HANDOFF §"What NOT to change."

### New code decisions (this session)

- **`disclosure_map` in state** — `AgentState` now carries the full `ProgressiveDisclosure.build_disclosure_map()` output. `survey_scrub_node` consumes `tier1_system_overview` for the community context section. The disclosure map provides bounded context access (Tier 1 system overview, Tier 2 community details, Tier 3 paper‑community map) for Survey Mode synthesis.
- **`find_similar()` min_score default from env** — `SPECTOR2_SIMILARITY_THRESHOLD` env var (default 0.6) controls cosine‑similarity threshold. Respects user override via `setdefault` pattern.
- **Cache `doi` field** — `put()` now stores `"doi": doi` in the cache entry so `find_similar()` can return the original DOI (not the lowercased dictionary key). Backward‑compatible: existing cache entries without `doi` fall back to the dictionary key.
- **No hardcoded N for similarity results** — `find_similar()` returns all matches above threshold, not "top 5." Filtering is the consumer's responsibility.

### New architecture decisions (this session — design only, not yet implemented)

- **Planner/Reflector/Executor 3‑role separation** — Planner owns mission and plan. Executor owns per‑step execution with per‑task‑type calibration. Reflector owns quality auditing. Roles are separate; no role edits another's domain.
- **Tiered planning, not 3 separate agents** — One planner LLM thinks at three scope levels (strategic → tactical → execution). Not three agents. Not three calls per cycle.
- **Handoff box as memory bridge** — The planner's continuity across GPU‑reset cycles is a structured state package (JSON + markdown), not LLM memory. The planner loads the box, thinks, plans, executes, and writes the updated box. The GPU reset is sleep; the box is memory.
- **Plan trees with revision tracking** — Nested plan structure (phase → goal → step). Each node tracks revision ID. Completed phases can be reopened when new data reveals gaps. Dependency gates enforce execution order.
- **Per‑task‑type calibration generalized** — The extraction `boundary_lower/upper` + `output_ratio` EMA pattern applies to ALL LLM calls. Per‑task‑type stats in a shared file. Pass/fail signals differ; calibration math is identical.
- **Map‑reduce decomposition for large‑context tasks** — When a step exceeds its per‑task‑type boundary, decompose recursively into map (bounded analysis) → reduce (synthesize into unified output). No truncation; no information loss.
- **No hardcoded thresholds** — All sizing is learned (calibration), all triggers are adaptive (plan‑revision count, not cycle count), all measurement is objective (tiktoken, not LLM estimates).
- **User input is priority zero** — Live user queries preempt autonomous work. Mode switch between interactive (short steps, fast response) and autonomous (full cycles) on user activity.
- **Dependency gates — no writing on incomplete knowledge** — A phase executes only when its dependencies are satisfied. Gaps are documented, not papered over. Honesty over user satisfaction.

---

## What NOT to change

All prior constraints apply. Additions from this session:

### Code constraints
- Do NOT remove `disclosure_map` from `AgentState` — it provides Tier 1–3 bounded context for Survey Mode synthesis.
- Do NOT revert `survey_scrub_node` to the ad‑hoc community section builder — use `disclosure_map["tier1_system_overview"]`.
- Do NOT remove `find_similar()` from `Spector2Cache` — it's the consumer for the paper‑similarity discovery pipeline.
- Do NOT remove the `doi` field from cache entries — `find_similar()` depends on it for returning original DOIs.
- Do NOT remove the `setdefault` env pattern for `SPECTOR2_SIMILARITY_THRESHOLD`.

### Architecture constraints (design — enforce when building)
- Do NOT collapse Planner + Reflector into one role — independent audit is the point of the architecture.
- Do NOT make the planner's execution plan a fixed script — plans are living documents, revisable every cycle.
- Do NOT hardcode thresholds for staleness, batch size, entity count, or token limits — use learned calibration and adaptive triggers.
- Do NOT derive context budgets from `num_ctx` — use the per‑task‑type boundary formula.
- Do NOT feed JSON to LLMs — canonical store is JSON, LLM receives generated markdown.
- Do NOT allow writing/output generation on incomplete knowledge — dependency gates must block execution when gaps exist.
- Do NOT degrade degraded‑output calibration — garbage tokens must never update the output_ratio EMA. Only healthy tokens (up to degradation point) contribute.
- Do NOT make the reflector's calibration depend on planner acceptance — calibrate on executor outcome (the ground truth), not planner agreement.

### All prior constraints
- Do NOT reinstate `_extract_batch_recursive` or `_merge_entity_dicts`.
- Do NOT add `batch_size` back as a parameter — batch sizing is token‑driven.
- Do NOT derive chunk budget from `num_ctx` — use the self‑calibrating boundary formula.
- Do NOT remove any extraction calibration, detection, or GPU‑management method.
- Do NOT lower `boundary_lower` below 8000.
- Do NOT remove the symmetric clamp, data‑quality exemption, ratio‑based repack, realtime logging, or `env=env` OLLAMA passthrough.
- All prior constraints: per‑paper source prefixes, `chunk_index`, no `lstrip()`, no `Accept: application/json` on EPMC session, etc.

---

## File map

```
MODIFIED FILES (this session — code):
src/state.py                           — + disclosure_map: NotRequired[Dict] (line 55)
src/graph/survey_nodes.py              — + ProgressiveDisclosure instantiation in 
                                          survey_community_route_node (lines 311-321)
                                        — + survey_scrub_node now consumes disclosure_map
                                          instead of ad‑hoc community section (lines 1101-1107)
src/utils/spector2_cache.py            — + doi field in put() entry (line 76)
                                        — + find_similar() method (lines 86-130)
tests/test_spector2_cache.py           — + 3 new tests: find_similar_returns_results,
                                          find_similar_doi_not_cached, find_similar_respects_threshold
                                        — + numpy import at module level
HANDOFF.md                             — This file — comprehensive session handoff

MODIFIED FILES (this session — documentation):
README.md                              — Updated Phase 11 status, architecture redesign,
                                          novel approaches, revised planned capabilities

UNMODIFIED BUT RELEVANT (existing components):
src/graph/progressive_disclosure.py    — Fully built + tested, NOW WIRED (Gap A closed)
src/graph/community_detection.py       — Louvain community detection (used by disclosure)
src/agents/community_summarizer.py     — Community LLM summarization (wired)
src/agents/relevance_router.py         — Query → community routing (wired)
src/agents/extraction_agent.py         — Pulsed‑wave extraction with self‑calibration
src/agents/orchestrator.py             — Current daemon cycle (to be replaced by planner architecture)
src/ingestion/pre_extractor.py         — GPU management, OLLAMA_NUM_PARALLEL passthrough
src/streaming_handler.py               — Compression‑ratio degradation detection
src/graph/base_graph.py                — KG abstract interface

FILES TO BUILD (Phase 12 implementation):
src/agents/planner.py                  — Tiered planner (T1/T2/T3), plan tree management
src/agents/reflector.py                — Deterministic + LLM‑based plan/executor auditing
src/utils/handoff_box.py               — JSON + markdown dual‑format state persistence
src/utils/calibration.py               — Generalized per‑task‑type boundary/ratio system
src/utils/capability_registry.py       — Executor boundary + input format advertisement
```

---

## Recommendations

### Immediate (next session) — implement the architecture foundation

1. **Build per‑task‑type calibration system** — Generalize `extraction_stats.json` pattern. Shared file `projects/default/orchestration_stats.json` keyed by `{task_type}.{model}`. Same `boundary_lower/upper`, `output_ratio`, `total_chunk_tokens`, `total_output_tokens` fields. Pass/fail signal definitions per task type.

2. **Build the handoff box format** — Structured JSON state package: `{mission, goals: [{id, priority, status, corroboration_count, last_revision, ...}], plan_tree: {phases: {id → {goals: {id → {steps: [...]}}}}}, calibration: {per_task_type_stats}, open_questions: [...], hypotheses: [{claim, confidence, supporting_papers, created_cycle}], crash_log: [...]}`. Markdown generation for LLM consumption.

3. **Build the Planner** — Tier 1 (immediate execution plan from top‑priority goals). Tier 2 (priority rebalancing when new data arrives). Tier 3 (strategic review ~every 10 cycles). Plan tree with revision tracking. Dependency gate enforcement. Handoff box read/write.

4. **Build the Reflector** — Deterministic checks: regex plan parseability, topological sort for circular dependencies, resource existence lookup, output truncation detection. LLM‑based checks: executor output quality against plan step, plan‑goal alignment (periodic), staleness detection for ambiguous cases. Calibration on executor outcome, not planner acceptance.

### Short‑term

5. **Wire SPECTER2 similarity into discovery** — When a new paper is ingested, call `find_similar(doi)` to find the N most similar cached papers. Use their community assignments as priors for the new paper's community detection. This reduces cold‑start cost for Louvain and improves community coherence.

6. **≥8 h daemon validation** — Run `orchestrator --live` for ≥8 h continuous. Monitor: boundary convergence in `extraction_stats.json`, GPU memory pressure between waves, `bad_chunks.json` accumulation, orphaned `ollama runner` processes. Track oscillation amplitude — boundary should converge, not oscillate.

7. **atexit GPU cleanup** — Register `atexit` handler: SIGKILL Ollama subprocess, optionally re‑arm launchd plist. Catch SIGTERM/SIGINT too. Write `orchestrator_state.json` as `"stopped"` (not `"crashed"`) so next startup knows the exit was clean.

8. **Wave log rotation** — Delete `logs/extraction/wave_*.txt` from previous paper at the start of each new `extract_paper_recursive()` call. Delete `/tmp/opencode_wave*.sh` immediately after `osascript` fires (the Terminal.app has already read the script).

### Medium‑term

9. **Map‑reduce decomposition for executor steps** — When a step exceeds its calibrated boundary, automatically decompose (recursive map) and synthesize (reduce). Base case is always trivially bounded.

10. **Circuit breaker with health alerts** — Monitor calibration trends across all task types. If N consecutive cycles show declining boundaries across all types, trigger: reset calibration, recommend model re‑pull, write human‑readable health alert to handoff box.

### Beyond

11. **KG consolidation (Phase 13)** — Embedding‑based entity dedup using existing `all‑MiniLM‑L6‑v2` model (already loaded by relevance router). LLM‑based community consolidation with e4b. RAPTOR‑style hierarchical summarization integrated with ProgressiveDisclosure tiers.

12. **UI/UX for research co‑pilot** — Progress bars, estimated time remaining, real‑time status updates, clarifying‑question UI, contradiction‑flag alerts, plan‑revision history viewer. Streamlit upgrade.

---

## Prompt for next AI session

```
You are an expert senior software developer continuing the Federated RAG
system for autonomous biomedical research.  Pulsed‑wave parallel extraction
with self‑calibrating boundary is operational (41 tests).  Phase 11 is
fully closed: ProgressiveDisclosure is wired into Survey Mode, and
SPECTER2 paper similarity search (find_similar) is built (87 tests total).

This session completed a major architectural redesign: the orchestrator's
cognitive architecture has been designed as a Planner/Reflector/Executor
3‑role system with tiered planning, plan trees with revision tracking,
handoff box as memory bridge, per‑task‑type calibration generalized from
extraction, dependency gates, map‑reduce decomposition, circuit breakers
with natural‑language health alerts, and a self‑tuning staleness detector.
See HANDOFF.md for the full design.

Read HANDOFF.md and README.md carefully before making changes.

CURRENT STATE:
  - 87 tests pass, zero failures (41 extraction + 46 phase11/spector2).
  - Extraction uses pulsed‑wave parallel design with self‑calibrating
    boundary, ratio‑based repack, data‑quality exemption, per‑wave
    Terminal windows, and OLLAMA_NUM_PARALLEL passthrough.
  - ProgressiveDisclosure 3‑tier hierarchy wired into Survey Mode.
  - SPECTER2 find_similar() built with cosine‑similarity paper discovery.
  - Orchestrator daemon runs full autonomous cycle: web discovery → EPMC
    fetch → batch ingest → pulsed‑wave extraction → KG save → community
    detection → cycle handoff.  Self‑bootstraps Ollama via launchd disarm.
  - KG: ~3,810 nodes, ~262K edges.  BM25: 27K+ documents.  43+ papers.

CRITICAL OPEN (Phase 12 implementation):
  - Build per‑task‑type calibration system (generalize extraction_stats.json)
  - Build handoff box format (JSON canonical + markdown for LLM consumption)
  - Build Planner (Tier 1/2/3, plan trees, revision tracking, dependency gates)
  - Build Reflector (deterministic checks: regex, topology, resource lookup;
    LLM checks: output quality, plan alignment, staleness)
  - Build capability registry for executor boundary advertisement
  - Wire SPECTER2 find_similar into orchestrator discovery loop

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE — see README §17 + HANDOFF):
  - Do NOT collapse Planner + Reflector into one role.
  - Do NOT hardcode thresholds — use learned calibration and adaptive triggers.
  - Do NOT feed JSON to LLMs — canonical JSON, LLM receives markdown.
  - Do NOT allow output generation on incomplete knowledge — dependency gates.
  - Do NOT let degraded output pollute calibration — only healthy tokens count.
  - Do NOT calibrate Reflector on planner acceptance — use executor outcome.
  - All prior extraction + GPU + OLLAMA constraints still apply.

REUSABLE PRIMITIVES:
  - ExtractionAgent with _calculate_chunk_budget, _update_boundary, _update_output_ratio
  - ProgressiveDisclosure(graph_storage, community_data, summaries).build_disclosure_map(...)
  - Spector2Cache().find_similar(doi, min_score=0.6)
  - PreExtractor._restart_ollama_process() — SIGKILL + cooldown
  - PreExtractor._ensure_dedicated_ollama() — launchd disarm
  - TokenStreamHandler — compression‑ratio + pattern detection
  - Orchestrator(graph_storage=gs, dry_run=True).run_once()
  - Scheduler(cooldown_seconds=10) — chain‑based daemon loop

QUICK START:
  python -m pytest tests/test_extraction_agent.py -q --tb=short   # 41 tests, ~6s
  python -m pytest tests/test_progressive_disclosure.py tests/test_phase11_integration.py tests/test_spector2_cache.py -q  # 46 tests
  python phase9_verify.py --test orchestrator                  # dry run
  python phase9_verify.py --test orchestrator --orchestrator-live  # live cycle
```
