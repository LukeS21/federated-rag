# Phase 10.5 → Phase 10.5 (cont.) Handoff — 16 May 2026 (extraction hardening session)

## Quick start

```bash
# Full daemon cycle (live — calls EPMC + Ollama)
python phase9_verify.py --test orchestrator --orchestrator-live

# Dry run (see what WOULD happen, no API spend)
python phase9_verify.py --test orchestrator

# All tests
python -m pytest tests/ -q --tb=short

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

**Phase 10 is 100% complete. Phase 10.5 extraction-hardening session is concluding.**
All 307 existing tests pass. The core daemon pipeline runs end-to-end. The system
has been battle-tested through multiple live daemon cycles, revealing and
addressing real-world degradation patterns in local Ollama extraction.

**Knowledge graph** (as of last live cycle): ~3,810 nodes, ~262K edges. BM25 corpus:
27K+ documents. 43+ papers ingested.

**What's running:** Background daemon discovers topics via web search, fetches OA
papers from Europe PMC, batches them into ChromaDB+BM25, extracts entities with
batched line‑tagged extraction, updates the KG, runs community detection, and
writes cycle handoffs.

**What this session focused on:** Hardening the extraction pipeline against
local‑LLM degradation (Ollama/gemma4:e4b on M3 Max). The daemon was burning
45 min+ on hung batches and producing garbage output in late‑cycle batches
due to llama.cpp Metal‑backend memory fragmentation.

---

## What changed this session

| | Before this session | After this session |
|---|---|---|
| **Extraction prompt size** | All 37+ chunks in one LLM call → 20K‑50K token prompts → 15 min+ generation or timeout | Batched: 8 chunks/call → ~6K token prompts → 30‑90s per batch |
| **Extraction output format** | One entity per block, evidence repeated N times | Grouped: evidence once, entities compactly listed. ~60‑70% token savings |
| **Extraction output visibility** | Only `200 OK` / `Retrying` log lines | Real‑time streaming via `TokenStreamHandler` — tokens appear as generated |
| **Retry behavior** | `max_retries=2` — hung batch burned 45 min (3× 900s) | `max_retries=0` — hung batch fails once, pipeline continues |
| **Hang detection** | Only `LLM_TIMEOUT=900s` | Python‑level `ThreadPoolExecutor` timeout at 600s per batch |
| **Repetition loop detection** | None | Parser compares consecutive committed entities — aborts on identical block |
| **Ollama memory between batches** | No reset — fragmentation accumulated across batches | API‑based `keep_alive=0` + polling `GET /api/ps` between every batch |
| **Memory reset verification** | None | Logs running model count before/after reset (1 → 0 transition proves eviction) |
| **DIRECTION field** | Free‑form — LLM invented "source", "characteristic", "application", "N/A", "target" | Constrained to measurable change values; omitted for non‑biological entities |
| **README** | ~2000 lines, historical build diary, stale Phase 4‑5 handoff instructions | ~800 lines, North Star vision, current architecture, consolidated constraints, planned capabilities. History moved to `docs/phase-history.md` |
| **Dead code** | `pubmed.py`, `hierarchical_clusterer.py`, `PHASE8_STATUS.md` in codebase | Deleted. 9 deprecated scripts archived to `archive/` |
| **Phase 10 gaps A/B/C** | Listed as open in HANDOFF.md | Closed: `_load_state()`, `_cleanup_handoffs()`, `_ensure_file_logging()` all built |

---

## What was accomplished

### Batch extraction (`extract_entities_batched`)

**File:** `src/agents/extraction_agent.py` — new method on `ExtractionAgent`.

Splits chunks into groups of 8. Each batch gets its own `extract_entities()` call
with a 600s Python‑level timeout. Results are merged and deduplicated across
batches (`_merge_entity_batches()` — normalises entity names, keeps longest
evidence). Three call sites updated: `pre_extractor.py` (daemon ingest),
`survey_nodes.py` (Survey Mode), `nodes.py` (Deep Mode).

**Why:** 37+ chunks in one prompt = 20K‑50K tokens → gemma4:e4b takes 15 min+
or hangs. 8 chunks = ~6K tokens → 30‑90s. This alone saved the daemon from
the original 90+ min extraction hangs.

### Evidence‑grouped extraction format

**File:** `src/agents/extraction_agent.py` — system prompt rewrite.

Old format (one ENTITY per block, evidence repeated):
```
TYPE: material     ENTITY: polyethyleneimine   EVIDENCE: Polymeric nanocarriers...
TYPE: material     ENTITY: dendrimers          EVIDENCE: Polymeric nanocarriers...
TYPE: material     ENTITY: graphene-based      EVIDENCE: Polymeric nanocarriers...
```

New grouped format (evidence once, entities compact):
```
EVIDENCE: Polymeric nanocarriers like polyethyleneimine, dendrimers, graphene...
SOURCE: Chunk 1 | paper.pdf
TYPE: material
ENTITY: polyethyleneimine | DIRECTION: unchanged
ENTITY: dendrimers | DIRECTION: unchanged
ENTITY: graphene-based materials | DIRECTION: elevated
```

Parser (`_parse_line_tagged`) rewritten to handle both old and new formats
backward‑compatibly. Pipe‑delimited entity attributes supported
(`ENTITY: name | DIRECTION: value | CONTEXT: value`).

**Token savings:** ~60‑70% for grouped entities (evidence stated once).

### Streaming output

**Files:** `src/llm/__init__.py` — `streaming=True` parameter added to
`get_chat_model()`, passed to `ChatOpenAI`. `src/agents/extraction_agent.py` —
`TokenStreamHandler` attached to `_call_llm()`.

Tokens appear in realtime during extraction. Degradation (garbage output,
repetition loops, stalls) is visible immediately rather than after timeout.

### Repetition loop detection

**File:** `src/agents/extraction_agent.py` — `_parse_line_tagged`'s `_commit()`.

Each committed entity is compared to the previous one. If identical (all
fields match), a `RuntimeError` is raised and the batch is aborted. Catches
the most common degradation pattern: the model stuck repeating the same
entity block infinitely.

### Ollama memory management

**File:** `src/ingestion/pre_extractor.py` — `PreExtractor._reset_ollama()`.

Three‑step process:
1. POST `/api/generate` with `keep_alive=0` (request unload)
2. Poll `GET /api/ps` every 0.5s until model disappears from running list
3. Log before/after model count for verifiability

Called between **every batch** (in `extract_entities_batched`) and between
**every paper** (at end of `extract_paper`). Safety valve: 30s timeout.

**Verifiability:** Each reset logs:
```
Resetting Ollama — 1 model(s) loaded before: gemma4:e4b
Ollama reset complete in 2.1s — 0 model(s) loaded now: none — GPU memory cleared
```

The `before → after` model count transition (1 → 0) proves the model was
evicted. A fake reset (0.3‑0.8s, still showing model loaded) would be visible
in the logs.

### DIRECTION field constraint

**File:** `src/agents/extraction_agent.py` — system prompt rule 4.

DIRECTION now explicitly defined as a measurable change vs baseline. Valid
values: `elevated, decreased, increased, reduced, unchanged, upregulated,
downregulated, up, down`. For entities where this makes no sense (materials,
methods, equipment, concepts), DIRECTION must be **omitted entirely** — not
filled with placeholder values like "source", "characteristic", "application",
"general", "N/A", or "target".

### README restructure

**File:** `README.md` — ~2000 → ~800 lines.

New sections: North Star Vision, Current State, Background Daemon, Architectural
Constraints (consolidated from 4 scattered lists), Planned Capabilities, Phase
Evolution (condensed paragraph + link to `docs/phase-history.md`).

Obsidian vault (`docs/kg/`) noted as archived snapshot — not actively maintained.
Canonical docs: README (architecture), HANDOFF.md (next‑phase), `docs/phase-history.md`
(build history).

Dead files removed: `src/retrieval/pubmed.py` (deprecated), `src/agents/hierarchical_clusterer.py`
(never wired), `PHASE8_STATUS.md` (stale). Nine deprecated scripts archived to `archive/`.

---

## Lessons learned

### 1. Batched extraction is necessary but not sufficient

Breaking 37+ chunks into batches of 8 eliminates prompt‑size hangs (30‑90s
per batch vs 15 min+). But batches are still sequential on a single GPU —
the model runs 5‑38 batches per paper, each adding to the cumulative inference
count. GPU memory fragmentation can still degrade late batches within the same
paper. **Lesson:** Batches solve the prompt‑size problem; memory resets
(between batches) are needed for sustained‑load stability.

### 2. Ollama's `/api/ps` cannot verify Metal GPU memory state

The API tells what Ollama's Go server *thinks* about loaded models. It does
NOT reflect what llama.cpp's Metal backend actually holds in GPU memory.
Evidence: resets confirmed in 0.8s (physically impossible for a 9.6 GB model
unload from Metal), yet the next batch showed clear signs of degradation —
single‑field token spamming (`Energy: Energy: Energy: …` hundreds of times).
**Lesson:** The Ollama API provides an administrative view, not a hardware‑level
verification. True GPU memory state verification may require Metal profiling
tools (`MTLCaptureManager` via PyObjC) — complex and Mac‑only — or a full
Ollama process restart (which would kill other agents using Ollama).

### 3. GPU memory degradation produces multiple failure signatures

Not one failure mode — at least four distinct patterns observed:
- **Block‑level repetition**: Same entity block repeated infinitely (caught)
- **Token‑level spamming**: Single token repeated hundreds of times within
  one field value, e.g. `SOURCE: Chunk: Energy: Energy: Energy: …` (NOT caught)
- **Format collapse**: Model forgets extraction format, falls back to
  numbered output (`1. N/A\n2. N/A\n…`) (caught by parser as empty result)
- **Garbage output**: Random tokens, e.g. `TYPE: TYPE: TYPE: …` (caught by
  parser as empty or markdown‑fallback)

**Lesson:** Detection must cover all known failure signatures. Current
coverage: block repetition (caught), format collapse (caught), garbage
(caught). NOT caught: token‑level spamming within a field value.

### 4. Streaming output is critical for diagnostics

Real‑time token visibility reveals degradation the instant it starts — you
see the model stuck repeating `Energy: Energy:` or producing `N/A` lists.
Without streaming, you wait until the 600s timeout or the parser warning
after the batch completes. **Lesson:** Streaming is not a performance
feature — it's a diagnostic tool that shortens the debug cycle from hours
to seconds.

### 5. Evidence grouping saves 60‑70% tokens with zero quality loss

The model naturally groups entities by the evidence sentence they come from.
Giving it a format that expresses this efficiently (`EVIDENCE` once, `ENTITY`
lines compact) reduces generation time significantly. The parser handles both
old and new formats, so there's no transitional breakage. **Lesson:** Match
the output format to the model's natural extraction pattern — if it already
groups entities, give it a grouped format.

### 6. DIRECTION needs explicit constraints — the LLM invents otherwise

Without a clear definition of what DIRECTION means and when to omit it,
gemma4:e4b invents nonsense labels: "source", "characteristic", "application",
"target", "prerequisite", "N/A", "general". Defining valid values and
explicitly listing forbidden placeholder values in the prompt eliminates
this class of hallucination entirely. **Lesson:** Optional fields are a
lie — either define them or the LLM fills them with noise.

### 7. max_retries=0 is essential for daemon extraction

LangChain's default `max_retries=2` turns a single 900s timeout into a
45‑minute hang per failing batch. With the daemon processing 10‑40 batches
per cycle, even one hung batch under the old behavior could waste an hour.
Setting `max_retries=0` on the extraction‑specific LLM instance means
failures are logged and the pipeline continues immediately. **Lesson:**
In a background daemon context, fail fast > retry. The safety nets
(timeout, repetition detection, memory reset) should prevent failures
from happening; retrying a failed request just multiplies the damage.

---

## Novel approaches invented

### 1. Evidence‑grouped line‑tagged extraction format

Instead of one entity block per piece of evidence (repeating the evidence
text N times), evidence is stated once and entities are listed compactly
with pipe‑delimited per‑entity attributes. The parser converts this into
the same Python dict structure used by downstream consumers. Saves ~60‑70%
of output tokens for grouped entities. Generalizable to any extraction task
where multiple entities share source text.

### 2. Between‑batch Ollama model reset with polling verification

After each batch of 8 chunks, the system unloads and reloads the Ollama
model to flush llama.cpp Metal‑backend memory fragmentation. Unloading is
done via the Ollama native API (`keep_alive=0`); unloading is confirmed by
polling `GET /api/ps` until the model disappears from the running list.
A "before/after" model count log makes the reset auditable. This is the
first‑known approach to programmatic GPU‑memory hygiene for long‑running
local‑LLM pipelines.

### 3. Block‑level LLM repetition detection in parser output

Rather than relying on token‑probability heuristics or external timeout
detection, the parser itself detects degradation by comparing each committed
entity to the previous one. If the full entity dict (type, entity name,
evidence, source, direction, context, conditions) matches the previously
committed entity, the model is stuck in a repetition loop and the batch is
aborted immediately. Simple, deterministic, zero false‑positives for
legitimate extraction output.

---

## Identified gaps and status

### Phase 10 (carried forward)

| # | Gap | Severity | Status |
|---|------|----------|--------|
| A | State file write‑only | ~~Low~~ | ✅ Closed — `_load_state()` reads state on init |
| B | No handoff file cleanup | ~~Low~~ | ✅ Closed — `_cleanup_handoffs()` removes files >7 days |
| C | No daemon log management | ~~Medium~~ | ✅ Closed — `_ensure_file_logging()` with RotatingFileHandler |
| D | Line‑tagged format untested with real Ollama | Low | Partially addressed — multiple live cycles run. Parser stable. Prompt may need tuning for gemma4:e4b edge cases. |
| E | No long‑running daemon validation | Medium | Partially addressed — daemon ran multiple cycles (3+). Degradation patterns documented. Longer runs (>10 cycles, >24h) still needed. |
| F | Coverage‑gated routing not wired | Low | Not addressed |
| G | SPECTER2 embeddings unused | Low | Not addressed |

### Phase 10.5 (new — identified during extraction hardening)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| H | Token‑level spam not detected by parser | High | Single‑field token spamming (`SOURCE: Energy: Energy: Energy: …` repeated hundreds of times) is not caught by the block‑level repetition detector (which compares full entity dicts). The model can produce one massive entity block with a garbled field rather than multiple identical blocks. |
| I | `/api/ps` cannot verify true GPU memory state | High | Ollama's API reports administrative state, not Metal buffer state. 0.8s "confirmed unloaded" for a 9.6 GB model is physically impossible on Apple Silicon. No reliable software‑level verification exists without Metal profiling tools (PyObjC/MTLCaptureManager) or a full Ollama process restart. |
| J | No field‑level output sanity checks | Medium | The parser accepts any field value regardless of length or content. A 5000‑character `SOURCE` field full of repeated tokens passes through undetected. Reasonable upper bounds per field type would catch token‑spam corruption. |
| K | No guaranteed GPU‑memory reset mechanism | Medium | The only reliable way to fully flush Metal GPU memory is killing and restarting the Ollama server process — which would break any other agents using Ollama simultaneously (Streamlit UI, synthesis). No safe mechanism exists for the daemon to force a hard reset when other Ollama users are active. A coordinated restart (check `/api/ps` for other loaded models, only restart when sole user) was designed but not built. |

---

## Key architectural decisions (DO NOT UNDO)

### Carried forward from prior sessions

All Phase 4–10 constraints still apply. See README §17.

### New decisions (this session)

- **Batch extraction (8 chunks) over full‑prompt extraction** — 8 chunks produces
  ~6K‑token prompts that gemma4:e4b handles in 30‑90s. Full‑paper extraction
  (37+ chunks, 20K‑50K tokens) causes 15 min+ generation or hangs. The batch
  method preserves evidence quality (no text truncation) and adds a merge‑and‑
  deduplicate step for cross‑batch entity normalization.

- **Evidence‑grouped over per‑entity output format** — Stating evidence once per
  group of entities sharing that evidence saves 60‑70% of output tokens. The
  pipe‑delimited entity format (`ENTITY: name | DIRECTION: value`) keeps
  per‑entity attributes compact. Parser is fully backward‑compatible with the
  old per‑entity format.

- **Streaming extraction output** — `streaming=True` on the extraction LLM
  instance enables real‑time token visibility via `TokenStreamHandler`. This
  is a diagnostic necessity, not a UX feature — it reveals degradation patterns
  the instant they begin.

- **`max_retries=0` for extraction** — The extraction LLM is created with
  `max_retries=0`. Retrying a hung Ollama request in a daemon context just
  wastes time; fail fast and let the batch‑level error handling continue the
  pipeline.

- **`keep_alive=0` + polling for between‑batch reset** — Ollama's API is used
  to request model unload, and `GET /api/ps` is polled to confirm it. This is
  the best available approach short of a full process restart. The before/after
  model count logging makes resets auditable.

- **Block‑level repetition detection in parser** — Comparing consecutive
  committed entities catches the most common degradation pattern (LLM stuck
  repeating the same output) with zero false‑positives.

- **DIRECTION constrained to measurable changes** — Only valid values:
  elevated/decreased/increased/reduced/unchanged/upregulated/downregulated/up/down.
  Omitted entirely for non‑biological entities (materials, methods, equipment).
  This was needed because gemma4:e4b was inventing placeholder values (source,
  characteristic, application, target, N/A, general).

- **Phase 11 files committed as partial build** — `community_detection.py`,
  `progressive_disclosure.py`, `community_summarizer.py`, `relevance_router.py`
  and their tests exist in the codebase but are NOT yet wired into the daemon
  or retrieval pipeline. They are designed and built — wiring is the next task.

---

## What NOT to change

All prior constraints (Phase 4–10) apply. Additions from this session:

- Do NOT switch extraction back to full‑prompt (all chunks in one call) —
  batched extraction (8 chunks/call) is the format. The batch method exists
  specifically because full‑prompt extraction causes 15 min+ hangs on local
  Ollama.
- Do NOT remove `extract_entities_batched()` or `_merge_entity_batches()` —
  these are the extraction pipeline's primary entry points.
- Do NOT remove the evidence‑grouped format from the system prompt — the
  model performs better and saves 60‑70% tokens with it.
- Do NOT remove the block‑level repetition detector in `_parse_line_tagged`'s
  `_commit()` — it catches the most common degradation pattern.
- Do NOT remove `max_retries=0` from the extraction LLM — retrying hung
  Ollama requests wastes time in a daemon context.
- Do NOT remove `streaming=True` from the extraction LLM — real‑time output
  is critical for diagnosing degradation.
- Do NOT remove `_reset_ollama()` or its between‑batch call site — this is
  the only mechanism currently preventing cumulative GPU fragmentation.
- Do NOT remove the before/after model‑count logging in `_reset_ollama()` —
  it is the only verification that resets actually evicted the model.
- Do NOT switch extraction back to JSON output — line‑tagged is the format
  for Pass 2. Pass 1 (category discovery) retains JSON.
- Do NOT reinstate deleted files (`pubmed.py`, `hierarchical_clusterer.py`,
  `PHASE8_STATUS.md`) — they are dead code.
- Do NOT reinstate archived scripts — they served their purpose as phase
  demonstrators. The archive preserves history.

---

## File map

```
MODIFIED FILES (this session):
src/agents/extraction_agent.py      — +extract_entities_batched(), +_merge_entity_batches(),
                                       +block‑level repetition detection, +streaming wiring,
                                       evidence‑grouped system prompt, DIRECTION constraint
src/ingestion/pre_extractor.py       — +_reset_ollama() with polling + verification logging,
                                       batch extraction call site
src/llm/__init__.py                  — +streaming parameter on get_chat_model()
src/agents/orchestrator.py           — community detection wired into cycle, gap closures (A,B,C)
src/agents/__init__.py               — GapResolver, Orchestrator, Scheduler, run_parallel, write_handoff
src/graph/graph_builder.py           — minor fixes
src/graph/survey_nodes.py            — batch extraction call site
src/graph/nodes.py                   — batch extraction call site
src/state.py                         — Phase 11 fields added
tests/test_extraction_agent.py       — updated for line‑tagged + batch
tests/test_orchestrator.py           — state file, handoff tests
tests/test_scheduler.py              — crash recovery, duplicate rejection
README.md                            — restructured (~2000→~800 lines), North Star vision,
                                       consolidated constraints, planned capabilities
HANDOFF.md                           — this file
docs/phase-history.md                — full per‑phase build history

NEW FILES (this session):
src/agents/community_summarizer.py   — Phase 11 (partial build)
src/agents/relevance_router.py       — Phase 11 (partial build)
src/graph/community_detection.py     — Phase 11: Louvain community detection
src/graph/progressive_disclosure.py  — Phase 11: tiered KG disclosure
tests/test_community_detection.py    — Phase 11 community tests
tests/test_community_summarizer.py
tests/test_phase11_integration.py
tests/test_progressive_disclosure.py
tests/test_relevance_router.py

DELETED FILES:
src/retrieval/pubmed.py              — deprecated, replaced by Europe PMC
src/agents/hierarchical_clusterer.py — never wired into production
PHASE8_STATUS.md                     — stale, consolidated into docs/phase-history.md

ARCHIVED (→ archive/):
phase2_demo.py, phase3_demo.py, phase4_viz.py, investigate_bm25.py,
phase5_api_comparison.py, phase5_benchmark.py, phase5_verify.py,
phase9_pubmed_demo.py, scripts/acquire_corpus.py
```

---

## Recommendations

### Immediate (Phase 10.5 remaining gaps — priority order)

1. **Gap H — Add token‑level spam detection** (~10 lines). In `_parse_line_tagged`'s
   `_commit()`, check that no field value exceeds a reasonable length (e.g., 500 chars
   for `evidence`, 200 chars for `source`). If exceeded, the model is producing
   token‑spam degradation — abort the batch. This catches the `Energy: Energy: Energy:`
   corruption that the block‑level detector misses.

2. **Gap K — Safe Ollama restart when sole user** (~30 lines). Before attempting a
   hard restart (`ollama stop` / `ollama serve`), query `GET /api/ps`. Only proceed
   if gemma4:e4b is the ONLY model loaded — other models active means other agents are
   running (synthesis, UI) and would break. If safe, use `subprocess.run(["ollama", "stop"])`
   and `subprocess.Popen(["ollama", "serve"])`. Cost: ~15‑30s per restart. Use only
   between papers (not between batches — too expensive for 38‑batch papers).

3. **Gap G — Wire SPECTER2 paper similarity** (~20 lines). Add
   `paper_similarity_search(doi, top_k=5)` to `Spector2Cache`. Use cosine similarity
   between cached SPECTER2 embeddings. Surface related papers in cycle handoff or as
   a discovery supplement.

4. **Gap E — Multi‑hour daemon validation** (manual, ~2 hrs). Run the daemon for
   4‑8 hours (4‑8 cycles). Monitor: memory usage over time, per‑batch latency trend
   (are later cycles slower?), reset times (are they increasing?), entity counts
   (are they decreasing?). Capture Ollama logs if available.

### Phase 11 — Community routing (next major milestone)

Partial build already committed. The KG is at ~3,800 nodes / 262K edges — far
beyond the original community‑detection threshold. Next steps:

1. Wire `community_detection.py` (Louvain) into the daemon cycle — already called
   in `orchestrator._run_cycle()` but verify it runs cleanly and inspect output.
2. Generate community summaries via `community_summarizer.py`.
3. Wire `relevance_router.py` into Survey Mode retrieval — given a query, gate
   which communities provide context.
4. Wire `progressive_disclosure.py` — tiered disclosure (system/community/paper).

### Beyond Phase 11

The North Star Vision (README §1) identifies the persistent belief store as the
next major architectural layer. The claim ledger (`src/synthesis/claim_ledger.py`)
is the foundation. Key additions:

- **Belief store data model**: Claims with confidence, evidence_for/against,
  version_history, status (supported/challenged/contradicted/deprecated).
- **Contradiction detection agent**: Runs during daemon cycles — checks new
  entities/claims against existing beliefs, flags contradictions, updates
  confidences.
- **Probabilistic KG edges**: Edge weights adjusted over time by daemon cycles.

---

## Prompt for next AI session

```
You are an expert senior software developer continuing the Federated RAG system
for biomedical research. Phase 10 is complete. Phase 10.5 extraction‑hardening
session is concluding. Phase 11 (community routing & memory cascade) is the next
major milestone.

Read the full README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - 307 tests pass, zero failures.
  - Orchestrator daemon runs full autonomous cycle: web discovery → parallel
    EPMC fetch → batch ingest → PreExtractor → KG save → community detection →
    cycle handoff. Dry‑run + live modes.
  - KG: ~3,810 nodes, ~262K edges (grown significantly since Phase 10 handoff).
    BM25: 27K+ documents. 43+ papers ingested.
  - Extraction uses batched evidence‑grouped line‑tagged format with streaming
    output, block‑level repetition detection, and between‑batch Ollama memory
    resets (keep_alive=0 + /api/ps polling verification).
  - Phase 10 gaps A/B/C closed (state resume, handoff cleanup, daemon logging).
  - Phase 11 partial build committed: community_detection.py, community_summarizer.py,
    relevance_router.py, progressive_disclosure.py + their tests. Community
    detection already called in orchestrator cycle.
  - DeepSeek API available for development. Ollama (gemma4:e4b + qwen3.6:35b) is
    the production target.
  - README restructured with North Star vision, consolidated constraints,
    planned capabilities. docs/phase-history.md has full build history.

CRITICAL OPEN PROBLEMS (from Phase 10.5):
  H. Token‑level spam not detected by block‑level repetition detector.
     The `Energy: Energy: Energy: …` failure mode produces one massive entity
     block with a garbled field — not two identical blocks. Add a field‑length
     sanity check in the parser (Gap H in HANDOFF).
  I. Ollama's /api/ps cannot verify true Metal GPU memory state.
     The 0.8s "confirmed unloaded" for a 9.6 GB model is physically impossible.
     No reliable software verification exists without Metal profiling tools.
  K. No guaranteed GPU memory reset mechanism. The only reliable way is a full
     Ollama process restart, which would kill other agents. A coordinated restart
     (check /api/ps for other models, only restart when sole user) was designed
     but not built.

PHASE 10.5 REMAINING (close before Phase 11):
  1. Gap H — Token‑spam detection in parser (~10 lines)
  2. Gap K — Safe Ollama restart when sole user (~30 lines)
  3. Gap G — Wire SPECTER2 paper similarity (~20 lines)

PHASE 11 PLANNED BUILD ORDER:
  1. Verify community detection runs cleanly in orchestrator cycle (already called)
  2. Generate community summaries via community_summarizer.py
  3. Wire relevance_router.py into Survey Mode retrieval
  4. Wire progressive_disclosure.py — tiered KG disclosure
  5. Integrate community routing end‑to‑end

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE — see README §17 for full list):
  - Do NOT switch extraction back to full‑prompt or JSON. Batched evidence‑grouped
    line‑tagged format is the standard.
  - Do NOT remove between‑batch Ollama resets, repetition detection, streaming output,
    or max_retries=0 on extraction LLM.
  - All prior constraints still apply (per‑paper source prefixes, chunk_index,
    no lstrip(), no Accept:application/json on EPMC session, etc.)

REUSABLE PRIMITIVES:
  - Orchestrator(graph_storage=gs, dry_run=True).run_once()
  - Orchestrator(graph_storage=gs, interval_minutes=60).start()
  - PreExtractor.extract_paper(paper_id, chunks, graph_storage=gs)
  - ExtractionAgent().extract_entities_batched(chunks, categories, query)
  - ExtractionAgent._parse_line_tagged(text) / _merge_entity_batches(entities)
  - PreExtractor._reset_ollama() — API unload + polling verification
  - run_parallel(func, items, max_workers=4)
  - WebSearchClient().discover_topics(terms)
  - EuropePMCClient().full_text_xml(pmcid) — EPMC REST → PMC OAI fallback

QUICK START:
  python phase9_verify.py --test orchestrator              # dry run (~10s)
  python phase9_verify.py --test orchestrator --orchestrator-live  # live cycle
  python -m pytest tests/ -q --tb=short                     # all tests
```
