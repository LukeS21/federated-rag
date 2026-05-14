# Phase 9 → Phase 10 Handoff — May 2026 (Phase 9: near‑complete, Phase 10: designed)

## Quick start

```bash
# Phase 9 pipeline test (Europe PMC + Semantic Scholar, ~14s for 5 papers)
python phase9_europe_pmc_test.py --count 10

# Phase 9 pipeline with ingestion into ChromaDB + BM25
python phase9_europe_pmc_test.py --count 50 --ingest

# Phase 9 with custom query
python phase9_europe_pmc_test.py --count 10 --query "dental implant macrophage polarization"

# Ingest progresses persists — reruns skip already-ingested papers
cat projects/default/ingest_progress.json | python -m json.tool

# Tests (192 passing, zero failures)
python -m pytest tests/ -v --tb=short

# Existing Phase 8 download pipeline (deprecated, still works with EZProxy auth)
# python scripts/headless_download.py --limit 10
```

## Current project state

**Phase 9 is 50% complete.** The core architecture pivot from Playwright/EZProxy
PDF downloads to Europe PMC API-based full-text XML is finished, tested, and
proven (27× faster, 100% reliable for OA papers).  Three of the original six
gaps are closed (retry logic, progress persistence, ingestion wiring).  Three
remain (coverage diagnostic, figure pipeline, SPECTER2 caching).

**Phase 10 design is complete** (this session).  We spent this session exploring
the SOTA in persistent agent memory, graph-based RAG, sleep-time compute, token-space
continual learning, skill learning, and multi-agent orchestration.  The target
architecture is a dual-agent system: a background research daemon (autonomous
paper acquisition, KG maintenance, skill improvement) and a user-facing agent
(query routing, evidence-grounded synthesis).  This aligns with Microsoft GraphRAG
(Mar 2025), Letta Sleep-time Compute (Apr 2025), Letta Context Repositories
(Feb 2026), and Anthropic's multi-agent research system (Jun 2025).  The plan
uses DeepSeek API during build/accumulation phases, switching to local Ollama
(Qwen3.6 35B-A3B) once the KG, skills, and community structure are mature.

**Core pipeline** (Phase 9, built and tested):
```
Europe PMC search (OPEN_ACCESS:Y) → fullTextXML fetch (3-retry backoff)
  → JATS XML parse → chunk dicts → ChromaDB + BM25 ingest → progress checkpoint
Semantic Scholar → DOI resolve (with title fallback) → SPECTER2 embedding fetch
```

**Target architecture** (Phase 10–13, designed):
```
BACKGROUND DAEMON (cron/timer, autonomous):
  read KG → detect gaps → web discovery → Europe PMC search → ingest → extract
  → update KG → reflect on trajectories → improve skills → write HANDOFF.md

USER AGENT (on demand, priority over background):
  read HANDOFF → route query → retrieve from relevant communities → synthesize
  → anchor evidence → output with citations
```

### What changed this session

| | Before this session | After this session |
|---|---|---|
| **Retry logic** | None — transient 5xx caused failures | 3-retry exponential backoff (1s, 2s) in `_request()` |
| **Progress persistence** | Crash = restart from zero | `ingest_progress.json` with 10-paper checkpoints |
| **Ingestion wiring** | Parsed chunks not stored in indexes | `--ingest` flag in test harness, ChromaDB + BM25 |
| **Chunk IDs** | Non-unique across papers (collisions) | Source includes PMCID + chunk_index on every chunk |
| **Phase 9 doc reference** | Described as "In Progress" with 6 open gaps | Updated to reflect 3 closed, 3 remaining, full architecture |
| **Phase 10-13 planning** | Not designed | Full architecture designed, informed by 12-month SOTA review |

## What was accomplished in Phase 9 (this session)

### Gap 1 — Retry logic ✅

`src/retrieval/europe_pmc.py` — new `_request()` method with 3-retry exponential
backoff (delays: 1s, 2s).  Retries 5xx, timeouts, and connection errors.
Does NOT retry 4xx (client errors).  Both `_get()` (search) and `full_text_xml()`
(XML fetch) delegate to `_request()`.  Per-call Accept headers are preserved
through the retry loop (verified: XML endpoint returns 200 with correct header).

### Gap 2 — Progress persistence ✅

New file `src/utils/ingest_progress.py` — `IngestProgress` class tracks ingested
PMCIDs in `projects/default/ingest_progress.json`.  Checkpoints every 10 papers.
`is_completed()`, `mark_completed()`, `checkpoint()`, `finalize()` API.  Resume
verified: second run correctly skipped 3/3 already-ingested papers.

### Gap 3 — Wire ingestion ✅

`phase9_europe_pmc_test.py` — new `--ingest` flag.  Phase 5 in the pipeline:
PMCXMLParser.parse() → HybridRetriever.ingest() → IngestProgress.checkpoint().
Uses existing `public_corpus` ChromaDB collection and BM25 index.  BM25 corpus
loaded from disk on startup, saved after ingestion (accumulates across runs).
Anchoring ChromaDB singleton updated so evidence check uses same collection.

### Bonus fix — Unique chunk IDs ✅

`src/ingestion/pmc_xml_parser.py` — two changes to prevent ChromaDB ID collisions:
1. `source` field now includes PMCID: `"europe_pmc_xml_PMC12345"` (was generic)
2. `chunk_index` added to every chunk's metadata during `parse()`
This ensures unique IDs across papers — without it, paper A chunk 0 and
paper B chunk 0 would share ID `europe_pmc_xml__0` and collide in ChromaDB.

### SOTA research (this session — informs Phase 10–13)

We conducted a comprehensive review of published research and production
architectures from April 2024 through May 2026.  Key findings:

| Date | Source | Finding | Relevance |
|------|--------|---------|-----------|
| Apr 2024 | Microsoft GraphRAG | Entity KG → community detection → hierarchical summaries. Proves KG-based RAG substantially outperforms vanilla RAG on global queries. | Our KG already implements this pattern. Community detection + summaries are the missing layer. |
| Nov 2024 | Microsoft LazyGraphRAG | 0.1% the indexing cost of full GraphRAG, 700× cheaper queries. NLP noun-phrase extraction (no LLM) for concept graph, deferred LLM summarization. Iterative deepening relevance test. | The indexing-cost reduction makes 100K-paper corpora feasible. We should adopt the concept-graph-as-pre-filter layer. |
| Nov 2024 | Microsoft Dynamic Community Selection | Cheap model (GPT-4o-mini) routes queries to relevant communities. 77% cost reduction, 58-60% quality improvement. | This is the "MoE for memory" routing pattern we designed. Confirmed effective. |
| Mar 2025 | Microsoft Claimify | Structured claim extraction with disambiguation, verifiability gating, and context preservation. 99% entailment rate. | Complements our anchoring check. Could improve claim decomposition quality. |
| Apr 2025 | Letta Sleep-time Compute | Background "thinking" during idle. Dual-agent: primary (user) + sleep-time (memory). 5× test-time compute reduction, up to 18% accuracy gain. | This is our Phase 10 background daemon. Confirmed architecture. |
| May 2025 | Letta Memory Blocks | Structured context as discrete functional units. Agent manages what's in-context vs. accessible. | Our Phase 11 progressive disclosure. |
| Jun 2025 | Anthropic Multi-Agent Research | Orchestrator-worker with subagent spawning, external memory, compaction with handoff. 90.2% improvement over single-agent on research tasks. | Our Phase 10 orchestrator design. Confirmed effective at scale. |
| Sep 2025 | Anthropic Context Engineering | Compaction, structured note-taking, sub-agent architectures for long-horizon tasks. "Find the smallest set of high-signal tokens." | Guiding philosophy for our memory cascade. |
| Dec 2025 | Letta Skill Learning | Two-stage: reflection on trajectory → creation of skill file. 36.8% relative improvement. Skills stored as .md files, model-agnostic. | Our Phase 12 skill library. Skills improve from collective experience. |
| Dec 2025 | Letta Continual Learning in Token Space | Theoretical framework: agents learn by updating context tokens (C), not weights (θ). Context portable across model generations. | Justifies our API-first → local-switchover strategy. The learned context outlasts any model. |
| Feb 2026 | Letta Context Repositories | Git-backed memory filesystem. Progressive disclosure via file hierarchy. Memory defragmentation and subagent swarms. | Our Phase 12 memory architecture. Git versioning enables rollback and parallel memory operations. |
| Apr 2026 | Letta Context Constitution | Formal principles for agent self-managed memory. | Informs our handoff protocol design. |
| May 2026 | Letta Red-teaming | Models resist believing they persist. Prompting helps but is insufficient. Ephemeral self-conception is the core blocker. | Validates our instrumental framing: agents don't need to believe they persist — they just need to believe their output will be read. |

## Lessons learned

### 1. The Europe PMC API approach is the correct architecture — don't go back

27× faster, 100% reliable for OA papers, structured XML instead of rasterized
page.pdf().  The coverage tradeoff (PMC-only, ~80-90% of biomedical OA papers)
is acceptable for NIH-funded biomedical research.  Non-PMC papers can still
be acquired via the preserved Phase 8 Playwright pipeline when EZProxy is active.

### 2. Chunk ID collisions are silent and catastrophic

Without per-paper source prefixes and chunk indices, ChromaDB silently overwrites
chunks across papers.  The fix is minimal (include PMCID in source field + add
chunk_index metadata) but must be applied to ALL ingestion paths — PMC XML,
PDF, and future sources.

### 3. Progress persistence is a requirement, not a feature

A crash after ingesting 49 of 50 papers without checkpointing loses all work.
The `IngestProgress` class with 10-paper checkpoint frequency solves this with
minimal overhead.  The same pattern should be extended to KG updates and skill
generation in Phase 10.

### 4. The orchestrator-handoff pattern is proven in production

Anthropic's multi-agent research system and Letta's sleep-time compute both
use the same pattern: write plans to external memory, spawn subagents, compact,
read handoff on restart.  Our `HANDOFF.md` is the right primitive — it just
needs to be auto-generated by the orchestrator rather than manually written.

### 5. Models don't need to believe they persist — they need to believe their output will be read

Letta's red-teaming proved that models resist identifying as persistent entities,
even with extensive prompting.  But Anthropic's production systems (Claude Code,
Claude Research) show that instrumental framing works: the model is told it's
a temporary process that writes to external memory for future instances.  This
is both more honest and more effective.

### 6. Skills are a better learning primitive than prompt optimization

Letta's Skill Learning achieved 36.8% relative improvement by generating .md
skill files from agent trajectories.  Skills are portable across models, git-versioned,
and improve with more usage (reflection on multiple trajectories produces better
skills than reflection on one).  This is more robust than prompt optimization
for an autonomous system.

### 7. Retrieval should be community-gated, not flat

Microsoft's Dynamic Community Selection proved that a cheap model routing
queries to relevant knowledge graph communities achieves 77% cost reduction
with higher quality output.  Loading all chunks into context — even with
similarity threshold filtering — wastes the LLM's attention budget on irrelevant
clusters.

### 8. Web search is for discovery, not evidence

The web should be a discovery compass (finding emerging topics, ideas, directions)
but never an evidence source.  All claims must be grounded in peer-reviewed
papers with full text.  The web search client should produce `source_type: "discovery"`
results that are never ingested into evidence chains.  DuckDuckGo (no key)
or Semantic Scholar (already integrated) are sufficient for discovery.

## Identified gaps and status

### Phase 9 remaining (close before starting Phase 10)

| # | Gap | Severity | Status |
|---|------|----------|--------|
| 1 | Retry logic on transient failures | High | ✅ Done |
| 2 | Progress persistence | High | ✅ Done |
| 3 | Ingestion wired to ChromaDB | High | ✅ Done |
| 4 | Coverage diagnostic (PMC vs Semantic Scholar) | Med | Not yet — run comparison query, report PMC coverage % |
| 5 | Figure pipeline (XML `<graphic>` → vision_ingest) | Low | Not yet — URLs exist in captions, download + wire into Phase 7a |
| 6 | SPECTER2 caching | Med | Not yet — store locally, skip re-fetch, eliminates 84% pipeline time |

### Phase 10-13 gaps (designed, not built)

| # | Gap | Phase | Description |
|---|------|-------|-------------|
| 7 | KG not updated at ingest time | 10 | PreExtractor needs graph_storage parameter wired in |
| 8 | Gap-analysis loop not closed | 10 | Gap analysis output is text; needs to trigger structured searches |
| 9 | No autonomous research daemon | 10 | Orchestrator agent that runs on cron/timer |
| 10 | No subagent spawning | 10 | Parallel search/extract workers via ThreadPoolExecutor |
| 11 | No automated handoff protocol | 10 | Orchestrator writes HANDOFF.md before compaction |
| 12 | No community detection on KG | 11 | Leiden/Louvain community detection on NetworkX graph |
| 13 | No community summaries | 11 | LLM generates summaries at each hierarchy level |
| 14 | No relevance router | 11 | Cheap model gates community access for queries |
| 15 | No progressive disclosure | 11 | system/ vs research/ vs archive/ memory tiers |
| 16 | No skill library | 12 | .md skills created from agent trajectories |
| 17 | No trajectory logging | 12 | JSONL logging of all agent actions |
| 18 | No experiential memory | 12 | agent-learnings/ directory for preferences, strategies |
| 19 | No skill evals/gating | 12 | A/B test skills before deployment, CI/CD gates |
| 20 | No output templates | 13 | Grant, paper, methods, review templates |
| 21 | No evidence-anchored writing tools | 13 | Every claim auto-cited, pre-output anchoring gate |
| 22 | SPECTER2 not used in retrieval | Future | Embeddings collected but not queried — paper-level similarity |
| 23 | Web search not integrated | 10 | Discovery-only web client for topic exploration |

## Key architectural decisions (DO NOT UNDO)

All previous DO NOT UNDO from Phase 4–9 still apply.  Additional decisions:

### Phase 9 decisions (this session)

- **3-retry exponential backoff** on all Europe PMC HTTP calls — 5xx and timeouts
  get retries, 4xx do not.  Total artificial delay: 3s (1s + 2s).  Per-call
  headers preserved through retry loop via `requests.Session.request()` merge behavior.
- **Per‑paper source prefixes** in chunk metadata — `"europe_pmc_xml_PMC12345"`
  instead of generic `"europe_pmc_xml"`.  Required for unique ChromaDB IDs across papers.
- **`chunk_index` in metadata** — added during `PMCXMLParser.parse()` return.
  Ensures unique IDs within a paper.  Backward-compatible (PDFParser already includes it).
- **`IngestProgress` as a standalone utility** — file-based, no database dependency.
  Same pattern as Phase 8's zotero_sync_status.json.  Extensible to KG updates,
  skill generation, and trajectory logging in Phase 10.
- **`--ingest` flag on existing test harness** — pragmatic: the test harness doubles
  as the ingestion CLI.  Uses the same `public_corpus` ChromaDB collection and
  BM25 persist directory as Phase 3/4 ingestion.  No new collections or path divergence.
- **Ingestion updates anchoring ChromaDB singleton** — `set_anchoring_chroma(chroma)`
  called during `--ingest` so evidence anchoring check uses the same collection
  that ingestion populates.

### Phase 10-13 architectural decisions (this session — guiding design, not yet built)

- **Dual-agent architecture** — background daemon (research, KG, skills) + user
  agent (query routing, synthesis).  Separate processes, shared persistent storage.
  Background agent never talks to the user; user agent never manages memory.
  Avoids the Letta red-teaming problem (models resisting persistent identity).
- **API-first, local-switchover strategy** — DeepSeek API during build (Phases 9-13)
  and accumulation (2-4 weeks after Phase 13).  Switch background agent to local
  Ollama (Qwen3.6 35B-A3B) when KG/skills are mature.  User agent keeps API synthesis
  (Drafter/Critic/Arbiter) until local models close the gap.  Re-evaluate with
  each new open-weight MoE release.
- **Instrumental agent framing** — agents know they're temporary; they write to
  external memory for future instances.  No pretense of persistent identity.
  "Your output will be read by the next instance" is both honest and effective.
- **Evidence provenance at every memory layer** — each compression level (entity
  → community → memory block) references the layer below.  Anchoring check always
  verifies against Layer 0 (source text).  No evidence-free claims at any level.
- **Community-gated retrieval (MoE for memory)** — cheap model scores communities
  for query relevance.  Relevant communities get detailed chunk retrieval; others
  get summary only.  Inspired by Microsoft Dynamic Community Selection (77% cost
  reduction, higher quality).
- **Skills over prompt optimization** — agents learn from trajectories by generating
  reusable .md skill files.  Git-versioned.  Model-agnostic.  Improve with more
  usage (reflection on multiple trajectories > reflection on one).  36.8% relative
  improvement demonstrated by Letta.
- **Web as discovery compass, not evidence source** — web search (DuckDuckGo /
  Semantic Scholar) identifies topics and directions.  All claims must be grounded
  in peer-reviewed papers.  Discovery results tagged `source_type: "discovery"`,
  never ingested into evidence chains.
- **Phase 10 before Phase 11** — build the autonomous daemon first (populates KG),
  then layer community structure on top.  Community detection needs a populated KG.
- **Single Ollama model at a time** — Qwen3.6 35B-A3B is sufficient for background
  extraction and user query routing.  Dual-model loading (Qwen + Gemma) exceeds
  M3 Max 36GB practical limits.  Background and user agents share the model via
  priority scheduling (user preempts background).

## What NOT to change

All previous What NOT to change from Phase 4–9 still apply.  Additional constraints:

- Do NOT go back to Playwright/EZProxy as the primary pipeline — the 27× speedup
  is fundamental architecture, not optimization
- Do NOT add `Accept: application/json` back to the session default — it breaks
  the fullTextXML endpoint (406 Not Acceptable)
- Do NOT remove per-paper source prefixes or chunk_index from PMCXMLParser —
  unique ChromaDB IDs depend on these
- Do NOT skip the IngestProgress checkpoint on ingestion — crash recovery requires it
- Do NOT use web search results as evidence — discovery only
- Do NOT design the orchestrator to believe it's persistent — use instrumental
  framing (write for future instances)
- Do NOT load two Ollama models simultaneously on M3 Max 36GB — memory ceiling
- Do NOT delete `scripts/headless_download.py` or `data/external/` — preserved
  for non-OA paper acquisition
- Do NOT change the chunk format `{"text": "...", "metadata": {...}}` — all
  downstream consumers depend on this contract

## File map

```
NEW FILES (Phase 9, this session):
src/utils/__init__.py                              # Package init
src/utils/ingest_progress.py                       # Checkpoint-based ingestion tracking

MODIFIED FILES (Phase 9, this session):
src/retrieval/europe_pmc.py                        # +_request() with 3-retry backoff
src/ingestion/pmc_xml_parser.py                    # +chunk_index, +PMCID in source
phase9_europe_pmc_test.py                          # +--ingest flag, Phase 5 ingestion

PREVIOUS PHASE 9 FILES (unchanged this session):
src/retrieval/europe_pmc.py                        # Europe PMC REST client (was new last session)
src/ingestion/pmc_xml_parser.py                    # JATS XML → chunk dict parser
src/retrieval/semantic_scholar.py                  # +SPECTER2 embeddings, +title fallback
phase9_europe_pmc_test.py                          # End-to-end pipeline test harness
scripts/headless_download.py                       # Phase 8 improvements (deprecated but preserved)

PLANNED PHASE 10-13 FILES (not yet created):
src/agents/orchestrator.py                         # Background daemon loop
src/agents/subagents.py                            # Parallel search/extract workers
src/agents/handoff.py                              # Automated HANDOFF.md protocol
src/agents/scheduler.py                            # Cron/timer integration
src/memory/community_detector.py                   # Leiden/Louvain on NetworkX KG
src/memory/community_summarizer.py                 # LLM summaries per community
src/memory/relevance_router.py                     # Cheap model gates community access
src/memory/cascade.py                              # Chunk→summary→entity→community pipeline
src/memory/disclosure.py                           # Progressive disclosure management
src/skills/skill_loader.py                         # Mounts skills from directory
src/skills/skill_creator.py                        # Reflection → creation pipeline
src/skills/trajectory_logger.py                    # JSONL logging of agent actions
src/skills/skill_evals.py                          # A/B test skill versions
src/memory/experiential.py                         # Agent-learnings store
src/outputs/templates.py                           # Grant, paper, methods, review templates
src/outputs/anchored_writer.py                     # Evidence-anchored output generation
src/outputs/citation_integrator.py                 # Auto-citation insertion + format
src/retrieval/web_search.py                        # Discovery-only web search client
phases/phase10_daemon.py                           # Entry point for autonomous background agent
skills/                                            # Git-backed skill library (populated at runtime)
agent-learnings/                                   # Experiential memory (populated at runtime)

PROJECT DATA (auto-generated):
data/external/                                     # ~115 valid PDFs from Phase 8
projects/default/ingest_progress.json              # Phase 9 checkpoint (PMCIDs ingested)
projects/default/phase9_europe_pmc_test.json       # Pipeline benchmark results
projects/default/chroma_data/                      # ChromaDB (public_corpus collection)
projects/default/bm25_index/                       # Persisted BM25 corpus
projects/default/project_graph.json                # Persisted knowledge graph
```

## Prompt for next AI session

```
You are an expert senior software developer continuing Phase 9 (API-Based Literature
Ingestion) of a Federated RAG system for biomedical research. Read the full
README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - Phase 9 is 50% complete. Three high-priority gaps are closed:
    1. Retry logic — 3-retry exponential backoff on all Europe PMC HTTP calls ✅
    2. Progress persistence — IngestProgress checkpoints every 10 papers ✅
    3. Ingestion wiring — --ingest flag ingests into ChromaDB + BM25 ✅
  - Chunk IDs are now unique across papers (source includes PMCID + chunk_index).
  - Three Phase 9 gaps remain:
    4. Coverage diagnostic (PMC vs Semantic Scholar comparison)
    5. Figure pipeline (XML <graphic> URLs → vision_ingest)
    6. SPECTER2 caching (store locally, skip re-fetch)
  - Phase 10-13 architecture has been fully designed (see HANDOFF.md and README.md).
    The next major milestone after Phase 9 closure is the autonomous background
    research daemon (Phase 10).
  - The system uses DeepSeek API for all LLM calls. The long-term strategy is
    API-first during build, switching to local Ollama (Qwen3.6 35B-A3B) once
    the KG, skills, and community structure are mature.
  - 192 tests pass, zero failures.

TOP PRIORITY — close Phase 9:
  4. COVERAGE DIAGNOSTIC: Search Europe PMC and Semantic Scholar with the same
     query. Report "X/Y papers (Z%) have PMC full text." ~30 lines.
  5. FIGURE PIPELINE: Download images from XML <graphic> URLs, wire into
     vision_ingest (Phase 7a). ~50 lines.
  6. SPECTER2 CACHING: Store embeddings in projects/default/spector2_cache.json.
     Skip Semantic Scholar API for previously-resolved papers. ~40 lines.

NEXT PRIORITY — Phase 10 foundation (can start in parallel with #5-6):
  7. Wire PreExtractor with graph_storage into Phase 9 --ingest so KG updates
     at ingest time. ~20 lines (PreExtractor already supports graph_storage).
  8. Close the gap-analysis loop: parse gap output into structured search
     queries, feed into Europe PMC → ingest → extract → re-synthesize. ~50 lines
     in a new module src/agents/gap_resolver.py.
  9. Build web_search.py — discovery-only web client (DuckDuckGo, no API key).
     Used by orchestrator for topic discovery, never as evidence source. ~30 lines.

QUICK START:
  python phase9_europe_pmc_test.py --count 10 --ingest
  python -m pytest tests/ -v --tb=short

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE):
  - Do NOT remove per-paper source prefixes or chunk_index from PMCXMLParser
  - Do NOT use web search results as evidence — discovery only
  - Do NOT add Accept:application/json to session default — breaks fullTextXML
  - Do NOT delete scripts/headless_download.py or data/external/
  - Do NOT change the chunk format {"text": "...", "metadata": {...}}
  - All new ingestion paths MUST include PMCID in source + chunk_index in metadata

DESIGN PRINCIPLES:
  - Evidence provenance at every memory layer (anchoring always checks Layer 0 source text)
  - Instrumental agent framing: agents know they're temporary, write for future instances
  - Skills over prompt optimization: learn from trajectories, generate reusable .md files
  - Community-gated retrieval: cheap model routes, expensive model synthesizes
  - Web = discovery compass, never evidence source
```
