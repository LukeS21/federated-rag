# Phase 9 → Phase 10 Handoff — 15 May 2026 (Phase 9: complete, Phase 10: foundation built)

## Quick start

```bash
# Full verification demo (all Phase 9 features + Phase 10 foundations)
python phase9_verify.py --fresh --skip-ingest --skip-figures

# Quick pipeline test with coverage diagnostic
python phase9_europe_pmc_test.py --count 10 --coverage

# Ingest into ChromaDB + BM25 + Knowledge Graph
python phase9_europe_pmc_test.py --count 10 --ingest --graph

# Ingest with figure downloads + KG updates + coverage
python phase9_europe_pmc_test.py --count 10 --ingest --graph --figures --coverage

# All tests (246 passing, 0 failures)
python -m pytest tests/ -q --tb=short

# Phase 8 pipeline (deprecated, preserved for non-OA papers)
# python scripts/headless_download.py --limit 10
```

## Current project state

**Phase 9 is 100% complete.** All 6 original gaps are closed: retry logic,
progress persistence, ingestion wiring, coverage diagnostic, figure pipeline,
and SPECTER2 caching. Three Phase 10 foundation pieces are also built:
PreExtractor+graph_storage wiring, gap_resolver.py, and web_search.py.

**The Europe PMC `fullTextXML` REST endpoint is returning 404 for all PMCIDs**
(server-side outage as of 15 May 2026). A transparent PMC OAI-PMH fallback was
added to `full_text_xml()` — same JATS XML content, different transport. The
code tries EPMC REST first, falls back to NCBI's OAI endpoint automatically.
Content quality is identical; only the XML envelope differs.

**246 tests pass, zero failures** (up from 186 when Phase 9 was last handed off).
54 new tests cover all Phase 9 additions and the Phase 10 foundation modules.
The 6 pre-existing synthesis agent mock failures are now resolved.

**Core pipeline** (Phase 9, built and tested):
```
Europe PMC search (OPEN_ACCESS:Y) → fullTextXML fetch (EPMC REST → PMC OAI fallback)
  → JATS XML parse → chunk dicts → ChromaDB + BM25 ingest → progress checkpoint
  → Figure pipeline (XML <graphic> URLs → download → vision_ingest)
  → PreExtractor KG update (if --graph)
Semantic Scholar → DOI resolve (with title fallback) → SPECTER2 embedding fetch
  → Spector2Cache (DOI-keyed, skip re-fetch)
Coverage diagnostic: EPMC ∩ S2 overlap → "X/Y papers (Z%) have PMC full text"
Gap analyser: gap text → structured queries → EPMC search → ingest → re-synthesize
Web discovery: DuckDuckGo → discovery-tagged results (never evidence)
```

**Target architecture** (Phase 10–13, designed):
```
BACKGROUND DAEMON (cron/timer, autonomous):
  read KG → detect gaps → web discovery → EPMC search → ingest → extract
  → update KG → reflect on trajectories → improve skills → write HANDOFF.md

USER AGENT (on demand, priority over background):
  read HANDOFF → route query → retrieve from relevant communities → synthesize
  → anchor evidence → output with citations
```

## What changed this session

| | Before this session | After this session |
|---|---|---|
| **Phase 9 status** | 50% (3/6 gaps) | 100% (6/6 gaps) |
| **Coverage diagnostic** | Did not exist | `src/retrieval/coverage.py` — DOI/title matching, 40% coverage demonstrated |
| **Figure pipeline** | XML URLs recorded but never downloaded | `vision_ingest_figure_url()` + `vision_ingest_xml_figures()`, describe=True tested with Ollama |
| **SPECTER2 caching** | Re-fetched from API every run | `src/utils/spector2_cache.py` — DOI-keyed JSON cache, hit verification 0.0000s |
| **EPMC fullTextXML** | Worked (200 responses) | Returns 404 for all PMCIDs as of 15 May 2026 — PMC OAI fallback added |
| **S2 rate limits** | 429s killed the pipeline | 429 retry with 10→20→40s exponential backoff |
| **ChromaDB duplicates** | Warnings on re-ingest | `ChromaClient.add_documents_deduped()` + `get_existing_ids()` |
| **Gap parser** | Did not exist | `src/agents/gap_resolver.py` — structured parsing, false-positive filtering, 9-word-boundary patterns |
| **Web search** | Did not exist | `src/retrieval/web_search.py` — ddgs primary, DDG API fallback, all results tagged `discovery` |
| **KG at ingest** | Not wired | `--graph` flag connects PreExtractor to graph_storage |
| **XML namespace stripping** | Missed self-closing prefixed tags | Fixed to handle `<mml:mtr/>` etc. (`[\s/>]` pattern) |
| **Tests** | 186 passing, 0 failures | **246 passing, 0 failures** (54 new tests) |

## What was accomplished

### Gap 1 — Retry logic ✅ (prior session)
3-retry exponential backoff on Europe PMC HTTP calls in `_request()`.

### Gap 2 — Progress persistence ✅ (prior session)
`IngestProgress` class with 10-paper checkpoints, `completed_count()`, `get_completed()`.

### Gap 3 — Wire ingestion ✅ (prior session)
`--ingest` flag on test harness, ChromaDB + BM25, anchoring singleton updated.

### Gap 4 — Coverage diagnostic ✅
`src/retrieval/coverage.py` — `run_coverage_diagnostic()` searches EPMC and S2 with the
same query, matches by DOI (exact then URL-prefix-stripped) and title fuzzy (≥0.6
threshold). Reports X/Y papers (Z%) with PMC full text. Demonstrated 40% coverage
for broad queries, 0% for niche OA/paywalled-divergent queries — both valid results.

Matching strategy (in order): DOI exact → DOI clean (strip `https://doi.org/`)
→ title fuzzy (SequenceMatcher + word-set Jaccard, threshold 0.6).

### Gap 5 — Figure pipeline ✅
`vision_ingest_figure_url()` in `src/vision/vision_ingest.py`:
- Downloads image from URL (`http://`, `https://`, or `file://`)
- Options: caption-as-description (zero LLM) or `describe=True` (Ollama vision model)
- Embeds via `FigureEmbedder` with source + figure_index in metadata
- `vision_ingest_xml_figures()` batch adapter scans parsed chunks for `figure_image_url`

Tested `describe=True` with gemma4:e4b via Ollama — bar chart correctly identified
("M2 highest, Treg medium, M1 lowest"). Real PMC figure URLs were absent during
EPMC outage but code path is identical.

### Gap 6 — SPECTER2 caching ✅
`src/utils/spector2_cache.py` — `Spector2Cache` class, DOI-keyed JSON persistence
at `projects/default/spector2_cache.json`. Validates 768-dim vectors. Handles
corrupted JSON gracefully (wipes). Proven 0.0000s cache-hit retrieval.

### Phase 10 #7 — PreExtractor + graph_storage ✅
`--graph` flag on `phase9_europe_pmc_test.py --ingest` creates `graph_storage` via
`create_graph_storage()`, loads existing KG, passes to `PreExtractor.extract_paper()`
for each ingested paper. KG accumulates across runs.

### Phase 10 #8 — Gap resolver ✅
`src/agents/gap_resolver.py` — `_parse_gaps_to_queries()` with:
- 9 word-boundary gap patterns (`\bno\s+[\w\s-]{0,40}?\bdata\b`, etc.)
- False-positive filter for null findings (`\bno\s+significant\s+difference\b`, etc.)
- `lstrip("the ")` bug fixed (was stripping individual characters)
- Hyphens in compound words (IL-17A, Ti-6Al-4V) not treated as bullet markers
- `GapResolver` class: parse → search → fetch → ingest loop with graph_storage support

### Phase 10 #9 — Web search ✅
`src/retrieval/web_search.py` — `WebSearchClient` using `ddgs` library (primary)
with DDG Instant Answer API fallback. All results tagged `source_type: "discovery"`.
`discover_topics()` for multi-term parallel discovery.

### Bonus fixes — 9 bugs found and fixed during verification

| Bug | Severity | File | Fix |
|-----|----------|------|-----|
| Coverage PMID↔PMCID mismatch | High | `coverage.py` | Replaced with DOI exact→clean→title fuzzy |
| Coverage `s2_doi` dead code | Low | `coverage.py` | Removed |
| Gap parser `lstrip("the ")` | High | `gap_resolver.py` | `removeprefix("the ")` |
| Gap parser `\bno\s+\w+\s+data\b` | Med | `gap_resolver.py` | `[\w\s-]{0,40}?` flexible quantifier |
| Gap parser false-positive scope | Med | `gap_resolver.py` | Check `gap_title` only, not whole block |
| Gap parser hyphen bullet split | Med | `gap_resolver.py` | Line-start-only regex |
| S2 double rate-limiting | Med | `semantic_scholar.py` | Removed redundant `_rate_limit()` |
| S2 429 not retried | High | `semantic_scholar.py` | 10→20→40s exponential backoff |
| XML `<mml:mtr/>` not stripped | Med | `pmc_xml_parser.py` | `[\s/>]` pattern |

## External API status & diagnostic guide

Knowing whether a failure is "our code" or "their server" is critical for
debugging. Below is a timeline, current status, and diagnostic procedure for
each external dependency.

### Europe PMC `fullTextXML` — returns 404 (server-side outage)

**Timeline:**
- 13 May 2026: Endpoint returned 200. `phase9_europe_pmc_test.py` logged
  "Fetched 5 XMLs (0 empty)". Pipeline worked end-to-end.
- 15 May 2026 13:07 UTC: Endpoint returned 404 for ALL tested PMCIDs
  (PMC5506916, PMC5512621, PMC6677551, PMC7876544, PMC8221428, PMC8866424,
  PMC4302049, PMC12900525). No intermediate 429s — direct 404. The papers
  still exist in PMC (search confirms `inPMC: Y`).

**What we checked:**
- EPMC search API still works (`200`). Search for `PMCID:PMC5506916` returns
  metadata with `inPMC: Y`.
- The `fullTextXML` endpoint returns `404` with zero-length body (not a
  content-negotiation error, not a redirect, not a rate limit).
- Direct `curl` without Accept header: `404`.
- Direct `curl` with `Accept: text/xml`: `404` (was previously needed to
  avoid `406`).
- Alternative endpoint pattern `/fullTextXML?format=xml`: `404`.
- Alternative endpoint on `europepmc.org`: returns HTML page, not XML.

**When to suspect it's an "us" error:**
- If the endpoint returns `406 Not Acceptable` — the Accept header is missing
  or wrong. The code sends `Accept: text/xml, application/xml, */*`. If this
  header was accidentally changed, the `406` would return instead of `404`.
- If the endpoint returns `429 Too Many Requests` — rate-limiting, not outage.
- If only some PMCIDs return 404 but others work — the specific papers may
  have been removed from PMC.
- If EPMC search also fails — the entire EPMC REST API is down.

**Current status (15 May 2026):** EPMC REST endpoint is down. PMC OAI-PMH
fallback is active and working (280KB JATS XML per paper, identical content).
The code tries EPMC REST first; if it fails, it transparently uses OAI.
**No code change is needed when EPMC REST comes back up** — the primary path
succeeds and the fallback is never invoked.

### Semantic Scholar — 429 rate limits (quota exhaustion)

**Timeline:**
- Throughout Phase 9 testing: S2 returned 200 when called with sufficient
  spacing. SPECTER2 embeddings were fetched, coverage diagnostics returned
  5 results.
- 15 May 2026: When running multiple tests back-to-back (SPECTER2 cache test →
  coverage test → SPECTER2 retry), S2 began returning `429 Too Many Requests`.
  Even with `_min_interval=3.0s` per-call spacing, the **hourly quota** was
  exhausted by sequential test runs.

**What we checked:**
- S2 search with API key in direct `curl`: `200` after 30s cooldown.
  Confirms the API key is valid and the endpoint works.
- S2 search without API key: `429` (free tier limit).
- The code sends `x-api-key` header correctly when `S2_API_KEY` is set in `.env`.
- The `_min_interval=3.0s` works per-call but doesn't account for hourly quota
  (likely 100-500 requests/hr for free tier with API key).

**When to suspect it's an "us" error:**
- If the API key is missing/expired — check `S2_API_KEY` in `.env`. The code
  logs S2 errors with the HTTP status. 429 with a valid key means quota
  exhausted; 429 without a key means free tier limit.
- If the `_min_interval` was accidentally increased (e.g., to 30s) — S2 would
  work but be unnecessarily slow.
- If the `_request()` retry loop has a bug — check that 429 → sleep → retry
  works. A trace with 3 consecutive 429s means the backoff isn't sleeping
  long enough.

**Current status (15 May 2026):** S2 works with API key and adequate cooldown.
The 429 backoff (10→20→40s) prevents cascading failures. Between test runs,
wait 30-60s for the hourly quota bucket to refill. For the Phase 10 daemon
(runs once per hour), the quota is sufficient.

## Lessons learned

### 1. Verification tests would have caught 6 of 9 bugs before merge

The coverage matching bug (PMID↔PMCID), gap parser false-positive scope, S2 rate
limit handling, and XML namespace issues all passed initial "it works" tests but
failed under systematic verification. Running `phase9_verify.py` and the new
unit tests exposed every one. New code should ship with its own verification.

### 2. External API resiliency needs at least two paths

The EPMC outage would have completely broken the pipeline without the OAI fallback.
Every critical external dependency should have a secondary resolution path — not
for performance, but for availability. The EPMC REST + PMC OAI dual-path pattern
should be extended to Semantic Scholar (e.g., OpenAlex as fallback for paper
resolution).

### 3. Regex-based HTML/XML processing has a long tail of edge cases

The namespace-stripping regex handled opening tags, closing tags, and attributes,
but missed self-closing tags (`<mml:mtr/>`). The gap parser's `lstrip("the ")`
stripped individual characters, not the phrase. Regex is fast but fragile — each
new data source (OAI XML vs EPMC REST XML) reveals a new edge case. For Phase 10,
fuzz-test the parser against a diverse set of XML samples.

### 4. Rate limiting needs both per-call spacing and aggregate backoff

The S2 client's `_min_interval=3.0s` worked per-call but sequential test runs
exhausted the hourly quota. Adding 429-specific exponential backoff (10→20→40s)
alongside per-call spacing prevents both transient and aggregate quota failures.
This same pattern should apply to any rate-limited API (ddgs, EPMC).

### 5. Coverage diagnostic is valuable even at 0%

A 0% coverage result is not a failure — it's data. It tells the orchestrator that
the most relevant S2 papers are paywalled and can only be acquired via EZProxy
(Phase 8 path). A coverage-gated routing decision ("route to EZProxy if coverage
< 30%") would make the background daemon more adaptive.

### 6. Caching at the right granularity eliminates the dominant cost

SPECTER2 embedding fetching was 84% of pipeline time on re-runs. A 90-line cache
module reduced it to 0.0000s. The same pattern should apply to: (a) EPMC search
results (query → paper list cache, 24h TTL), (b) full-text XML (PMCID → XML cache,
permanent), (c) coverage diagnostic results (query → coverage report, 7-day TTL).

### 7. The gap parser's false-positive filter must scope to the actual gap sentence

Checking the entire text block for false-positive patterns caused valid gaps to
be discarded when the block also contained a null finding. Scoping the check to
the first sentence (`gap_title`) fixed this. This principle generalizes: validation
filters should operate on the smallest relevant unit, not the full context window.

## Novel approaches invented this session

1. **PMC OAI-PMH as fullTextXML fallback** — Using the OAI-PMH protocol (designed
   for bulk harvesting) as a transparent single-article retrieval fallback when
   the REST endpoint is down. Zero code changes for callers.

2. **Hierarchical DOI matching** — Three-tier matching (DOI exact → DOI clean
   → title fuzzy) with progressive degrace. Each tier catches a different class
   of API format inconsistency without requiring per-API normalization.

3. **Word-boundary-aware gap detection with false-positive filtering** — Instead
   of substring keyword matching (which catches "no significant difference" as a
   gap), regex patterns with `\b` boundaries + a dedicated false-positive exclusion
   list correctly distinguish real research gaps from null findings.

4. **Dedup-before-add in ChromaDB** — Checking existing IDs via `collection.get(ids=…)`
   before `collection.add()` eliminates duplicate-entry warnings on re-ingest
   without requiring collection-level dedup configuration.

## Identified gaps and status

### Phase 9 (all closed ✅)

| # | Gap | Status |
|---|------|--------|
| 1 | Retry logic | ✅ Done |
| 2 | Progress persistence | ✅ Done |
| 3 | Ingestion wiring | ✅ Done |
| 4 | Coverage diagnostic | ✅ Done |
| 5 | Figure pipeline | ✅ Done |
| 6 | SPECTER2 caching | ✅ Done |

### Phase 10 foundation (all built ✅)

| # | Item | File | Status |
|---|------|------|--------|
| 7 | PreExtractor + graph_storage | `phase9_europe_pmc_test.py` (`--graph`) | ✅ Done |
| 8 | Gap resolver | `src/agents/gap_resolver.py` | ✅ Done |
| 9 | Web search (discovery) | `src/retrieval/web_search.py` | ✅ Done |

### Phase 10 core (designed, not built)

| # | Gap | File | Description |
|---|------|------|-------------|
| 10 | No autonomous daemon | `src/agents/orchestrator.py` | Background loop: detect gaps → discover → search → ingest → extract → KG → handoff |
| 11 | No subagent spawning | `src/agents/subagents.py` | ThreadPoolExecutor for parallel search/extract |
| 12 | No automated handoff | `src/agents/handoff.py` | Orchestrator writes HANDOFF.md before compaction |
| 13 | No scheduler | `src/agents/scheduler.py` | Cron/timer integration |

### Phase 11–13 (designed, not built)

| # | Gap | Phase | Description |
|---|------|-------|-------------|
| 14 | No community detection | 11 | Leiden/Louvain on NetworkX KG |
| 15 | No community summaries | 11 | LLM summaries per hierarchy level |
| 16 | No relevance router | 11 | Cheap model gates community access |
| 17 | No progressive disclosure | 11 | system/ vs research/ vs archive/ tiers |
| 18 | No skill library | 12 | .md skills from agent trajectories |
| 19 | No trajectory logging | 12 | JSONL logging of agent actions |
| 20 | No experiential memory | 12 | agent-learnings/ directory |
| 21 | No skill evals | 12 | A/B test before deployment |
| 22 | No output templates | 13 | Grant, paper, methods, review |
| 23 | No evidence-anchored writing | 13 | Auto-citation, pre-output anchoring gate |
| 24 | SPECTER2 not in retrieval | Future | Embeddings collected, not queried |

## Key architectural decisions (DO NOT UNDO)

### Carried forward from prior sessions

- **3-retry exponential backoff** on EPMC HTTP calls (5xx retried, 4xx not)
- **Per-paper source prefixes** (`"europe_pmc_xml_PMC12345"`)
- **`chunk_index` in metadata** on every chunk
- **`IngestProgress` file-based** (no DB dependency)
- **`--ingest` uses existing `public_corpus`** ChromaDB + BM25 paths
- **Anchoring singleton updated** on ingest (`set_anchoring_chroma`)
- **Dual-agent architecture** (background daemon + user agent, separate processes)
- **API-first, local-switchover** strategy (DeepSeek now → Ollama Qwen3.6 35B-A3B later)
- **Instrumental agent framing** (temporary, writes for future instances)
- **Evidence provenance** at every memory layer (anchoring always checks Layer 0)
- **Community-gated retrieval** (MoE for memory)
- **Skills over prompt optimization** (.md files, git-versioned)

### New decisions (this session)

- **PMC OAI-PMH as transparent `fullTextXML` fallback** — same JATS content, different transport.
  Tried first: EPMC REST. Falls back: NCBI OAI. Callers unaware of path used.
- **DOI-keyed SPECTER2 cache** — DOI is more stable than S2 paper_id across API versions.
  JSON file at `projects/default/spector2_cache.json`. 768-dim validation on store.
- **ChromaDB dedup-before-add** — `ChromaClient.add_documents_deduped()` checks existing
  IDs before `collection.add()`. Eliminates duplicate-entry warnings on re-ingest.
- **Coverage matching: DOI exact → DOI clean → title fuzzy** — three-tier progressive matching
  handles API format inconsistencies without per-source normalization tables.
- **Gap parser: word-boundary patterns + per-sentence false-positive filter** — regex
  with `\b` boundaries, 9 gap patterns, 6 false-positive patterns, scoped to first sentence.
- **S2 client: per-call spacing + aggregate 429 backoff** — `_min_interval=3.0s` for
  normal operation, 10→20→40s exponential backoff on 429. Both necessary for free-tier quotas.
- **`--graph`, `--coverage`, `--figures` flags on test harness** — pragmatic: single CLI
  exercises all Phase 9 features without a separate demo.

## What NOT to change

All prior constraints apply. Additions from this session:

- Do NOT remove the PMC OAI fallback from `full_text_xml()` — it's the working path while EPMC REST is down
- Do NOT change the gap parser's regex patterns without running the 18 unit tests
- Do NOT switch SPECTER2 cache key from DOI to S2 paper_id — DOIs are more stable
- Do NOT increase S2 `_min_interval` beyond 3.0s without testing (3.0s works with API key)
- Do NOT remove `ChromaClient.add_documents_deduped()` — re-ingest safety depends on it
- Do NOT use `lstrip()` for prefix removal — use `removeprefix()` or explicit check
- Do NOT remove `file://` URL support from `vision_ingest_figure_url()` — used for testing
- Do NOT skip the `_is_false_positive_gap()` check — gap quality depends on it
- Do NOT add `Accept: application/json` back to the EPMC session default — breaks the OAI endpoint
- All previous NOT TO CHANGE rules from Phase 4–9 still apply

## File map

```
NEW FILES (this session):
src/utils/spector2_cache.py                         # SPECTER2 embedding cache (DOI-keyed JSON)
src/retrieval/coverage.py                           # Coverage diagnostic (EPMC vs S2)
src/retrieval/web_search.py                         # Discovery-only web search (ddgs + DDG API)
src/agents/gap_resolver.py                          # Gap parsing + resolution loop
phase9_verify.py                                    # Comprehensive verification demo
tests/test_spector2_cache.py                        # 13 tests: cache put/get/persist/edge cases
tests/test_gap_resolver.py                          # 18 tests: parser, false positives, patterns
tests/test_coverage.py                              # 14 tests: matching, title overlap, DOI variants
tests/test_phase9_phase10_integration.py             # 9 tests: end-to-end pipeline validation

MODIFIED FILES (this session):
phase9_europe_pmc_test.py                           # +--coverage, --figures, --graph flags; cache integration
src/retrieval/europe_pmc.py                         # +PMC OAI-PMH fallback in full_text_xml()
src/retrieval/semantic_scholar.py                   # +_request() with 429 backoff; all methods use it
src/ingestion/pmc_xml_parser.py                     # +self-closing tag namespace stripping [\s/>]
src/retrieval/chroma_client.py                      # +add_documents_deduped(), get_existing_ids()
src/retrieval/hybrid_retriever.py                   # +dedup on ingest via add_documents_deduped()
src/utils/ingest_progress.py                        # +completed_count(), get_completed()
src/vision/vision_ingest.py                         # +vision_ingest_figure_url(), vision_ingest_xml_figures()

PHASE 10 PLANNED FILES (not yet created):
src/agents/orchestrator.py                          # Background daemon loop
src/agents/subagents.py                             # Parallel search/extract workers
src/agents/handoff.py                               # Automated HANDOFF.md protocol
src/agents/scheduler.py                             # Cron/timer integration

PHASE 11-13 PLANNED (not yet created):
src/memory/community_detector.py                    # Leiden/Louvain on KG
src/memory/community_summarizer.py                  # LLM summaries per community
src/memory/relevance_router.py                      # Cheap model gates community access
src/memory/cascade.py                               # Chunk→summary→entity→community
src/memory/disclosure.py                            # Progressive disclosure tiers
src/skills/skill_loader.py                          # Mount skills from directory
src/skills/skill_creator.py                         # Reflection → creation pipeline
src/skills/trajectory_logger.py                     # JSONL agent action logging
src/skills/skill_evals.py                           # A/B test skill versions
src/memory/experiential.py                          # Agent-learnings store
src/outputs/templates.py                            # Grant, paper, methods, review
src/outputs/anchored_writer.py                      # Evidence-anchored output
src/outputs/citation_integrator.py                  # Auto-citation insertion

PROJECT DATA (auto-generated — not committed):
projects/default/spector2_cache.json                # SPECTER2 cache (DOI → embedding)
projects/default/ingest_progress.json               # Ingested PMCIDs checkpoint
projects/default/phase9_europe_pmc_test.json        # Pipeline benchmark results
projects/default/phase9_verify_results.json         # Demo verification results
projects/default/chroma_data/                       # ChromaDB (public_corpus)
projects/default/bm25_index/                        # Persisted BM25 corpus
projects/default/extractions/                       # PreExtractor entity cache
projects/default/embeddings/                        # Paper embeddings
projects/default/project_graph.json                 # Knowledge graph
```

## Recommendations for Phase 10

### Immediate — build order
1. **`src/agents/scheduler.py`** (~30 lines) — cron/timer skeleton. Simplest piece, unblocks daemon.
2. **`src/agents/subagents.py`** (~50 lines) — `ThreadPoolExecutor` wrapper for parallel EPMC
   search + XML fetch. Reuses existing `EuropePMCClient` and `PMCXMLParser`.
3. **`src/agents/orchestrator.py`** (~200 lines) — main daemon loop calling: web discovery
   → gap resolver → EPMC search → subagent ingest → PreExtractor → graph_storage.save().
4. **`src/agents/handoff.py`** (~80 lines) — reads current state, writes HANDOFF.md for next
   instance. Consumes `IngestProgress`, `Spector2Cache.stats()`, KG node/edge counts.

### Architecture notes
- The `gap_resolver.resolve_gaps()` can already be called from the orchestrator — it handles
  search → fetch → parse → ingest → PreExtractor. Pass `graph_storage` and `ingest=True`.
- The coverage diagnostic can gate EZProxy routing: if coverage < 30%, queue papers for
  Phase 8 EZProxy acquisition instead of XML-only ingestion.
- SPECTER2 embeddings are cached but not yet queried. A `paper_similarity_search()`
  method on `Spector2Cache` would unlock paper-level recommendation for the orchestrator.
- The `vision_ingest_xml_figures()` function works with `describe=False` (caption-only,
  zero LLM cost). Deferred description via `describe_queued_figures()` can run overnight.
- `web_search.discover_topics()` returns structured results already — the orchestrator
  can use these directly as seed queries for the gap resolver.

### Test before building Phase 10
```bash
python phase9_verify.py --fresh                        # Full verification
python -m pytest tests/ -q --tb=short                   # 246 tests must pass
python phase9_europe_pmc_test.py --count 5 --coverage   # Coverage must return structured data
```

---

## Prompt for next AI session

```
You are an expert senior software developer continuing the Federated RAG system
for biomedical research. Phase 9 is complete. Phase 10 core build begins now.

Read the full README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - Phase 9 is 100% complete. All 6 gaps closed.
  - Three Phase 10 foundation pieces built: PreExtractor+KG wiring (--graph),
    gap_resolver.py, web_search.py (ddgs).
  - EPMC fullTextXML REST endpoint is down — PMC OAI-PMH fallback works transparently.
  - SPECTER2 cache is DOI-keyed JSON at projects/default/spector2_cache.json.
  - ChromaDB dedup prevents duplicate-entry warnings on re-ingest.
  - Vision pipeline works with describe=True (Ollama gemma4:e4b).
  - 246 tests pass, zero failures.

PHASE 10 BUILD ORDER:
  1. src/agents/scheduler.py — cron/timer skeleton (~30 lines)
  2. src/agents/subagents.py — parallel search/extract via ThreadPoolExecutor (~50 lines)
  3. src/agents/orchestrator.py — main daemon loop: web discovery → gap resolve →
     EPMC search → ingest → extract → KG → handoff (~200 lines)
  4. src/agents/handoff.py — auto-generate HANDOFF.md from state (~80 lines)

ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE):
  - Do NOT remove per-paper source prefixes or chunk_index from PMCXMLParser
  - Do NOT use web search results as evidence — discovery only (source_type: "discovery")
  - Do NOT remove the PMC OAI fallback from full_text_xml()
  - Do NOT change the chunk format {"text": "...", "metadata": {...}}
  - Do NOT add Accept:application/json to session default
  - Do NOT delete scripts/headless_download.py or data/external/
  - All new ingestion paths MUST include PMCID in source + chunk_index in metadata
  - Do NOT use lstrip() for prefix removal — use removeprefix() or explicit check
  - Do NOT switch SPECTER2 cache key from DOI to S2 paper_id

REUSABLE PRIMITIVES (already built, call directly):
  - GapResolver.resolve_gaps(text, graph_storage=gs, ingest=True)
  - WebSearchClient().discover_topics(["term1", "term2"])
  - EuropePMCClient().full_text_xml(pmcid) — EPMC REST → PMC OAI fallback transparent
  - PMCXMLParser().parse(xml, pmcid=pmcid, doi=doi)
  - HybridRetriever.ingest(chunks) — deduped, won't create duplicates
  - PreExtractor.extract_paper(paper_id, chunks, graph_storage=gs)
  - IngestProgress.is_completed(pmcid) / checkpoint(pmcid)
  - Spector2Cache().get(doi) / put(doi, s2_id, emb)

QUICK START:
  python phase9_verify.py --fresh --skip-ingest
  python -m pytest tests/ -q --tb=short
  python phase9_europe_pmc_test.py --count 5 --coverage
```
