# Phase 8 Completion Summary — May 11, 2026

## Tests
**212 passing, 0 failures** (201 existing + 11 new Phase 8 integration tests)

Run anytime:
```bash
python -m pytest phase5_benchmark.py test_correctness.py phase6_security_fuzzer.py phase6_multi_run.py phase5_api_comparison.py tests/ -v
```

## What Was Built

### Step 2 — Infrastructure Fixes (all complete)
| Component | File | Tests |
|-----------|------|-------|
| BM25 persistence to disk | `src/retrieval/bm25_index.py` (rewritten) | 2 new |
| Claim Ledger O(1) dict index | `src/synthesis/claim_ledger.py` (updated) | 2 new |
| Native `include_figures` in HybridRetriever | `src/retrieval/hybrid_retriever.py` (updated) | 1 new |
| Vision `describe_later` queue | `src/vision/vision_ingest.py` (updated) | existing |
| Monkey-patch REMOVED from figure_embedder | `src/vision/figure_embedder.py` (cleaned) | existing |

### Paper Acquisition Infrastructure
| Component | File |
|-----------|------|
| Unpaywall API client | `src/retrieval/unpaywall.py` |
| PMC OA Service client | `src/retrieval/pmc_oa.py` |
| 4-layer PDF resolution chain | `src/retrieval/pdf_downloader.py` |
| Corpus acquisition script | `scripts/acquire_corpus.py` |

### Phase 8 Architecture
| Component | File | Tests |
|-----------|------|-------|
| Neo4j adapter (BaseGraphStorage) | `src/graph/neo4j_storage.py` | ready when Neo4j available |
| Graph storage factory | `src/graph/__init__.py` | 1 new |
| Hierarchical clustering (Level 1 + 2) | `src/agents/hierarchical_clusterer.py` | ready |
| L0 corpus claim index | `src/retrieval/claim_index.py` | 1 new |
| SQLite cache store (all levels) | `src/cache/cache_store.py` | 3 new |
| CACHE_VERSION=v4 | `src/cache/__init__.py` | 1 new |
| Phase 8 benchmark runner | `phase8_benchmark.py` | — |

## Tangible Results (open these files)

### Downloaded Papers
- **16 papers** in `data/external/` (see manifest: `data/external/manifest.json`)
- **184 papers** identified but unfetchable (see: `data/external/missing.json`)
  — these need manual download via Zotero or institutional EZProxy
- **260 papers** searched and cataloged in `projects/default/corpus_acquisition.json`

### Benchmarks
- Naive RAG baseline: `projects/default/phase8_naive_rag.json`
  - **10 claims, 1.000 anchoring, ~40s** (gemma4:e4b, dense-claim format)

### To run the full survey pipeline benchmark:
```bash
python phase8_benchmark.py --skip-ingest
```
(Requires Ollama running with gemma4:e4b and qwen3.6:35b. Takes 5-10 min.)

### To view cached results:
```bash
python phase8_benchmark.py --cached
python scripts/acquire_corpus.py --cached
```

## Key Architectural Decisions

1. **BM25 is now persisted** — `BM25Index(persist_dir=...)` auto-saves corpus to pickle. No more ChromaDB rebuild on every restart.
2. **`include_figures` is native** — no more monkey-patch, no more import-order dependency. Just `retriever.query(..., include_figures=True)`.
3. **Claim ledger is O(1)** — `_by_id` dict eliminates linear scans.
4. **Neo4j is ready** — set `GRAPH_BACKEND=neo4j` in `.env` when a Neo4j instance is available. All consumers use the factory in `src/graph/__init__.py`.
5. **Caching uses SQLite** — `CacheStore` replaces file-per-entry JSON. `CACHE_VERSION=v4` invalidates all stale entries.
6. **Vision at scale** — `describe=False` embeds captions immediately, queues full descriptions for background processing via `describe_queued_figures()`.

## What Still Needs Doing

### 1. Corpus expansion (S2 API key pending)
The S2 API key was approved but not yet active (429 errors). Once active:
```bash
python scripts/acquire_corpus.py --provider both  # includes S2 OA PDFs
```

### 2. Neo4j setup
Docker not running. Once Docker is available:
```bash
docker run -d --name neo4j-rag -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password123 neo4j:5
# Then in .env: GRAPH_BACKEND=neo4j
```

Or use Neo4j Aura free tier (cloud) with `NEO4J_URI` env var.

### 3. Full-scale benchmarks
Current corpus: 22 papers (6 lab + 16 external). Run the full pipeline:
```bash
python phase8_benchmark.py  # ingests + runs survey pipeline
```

### 4. Manual paper addition
184 papers in `data/external/missing.json` need manual download via:
- Zotero Connector browser extension
- Institutional EZProxy
- Email corresponding authors for PDFs

## Modified Files
- `src/retrieval/bm25_index.py` — persistence support
- `src/retrieval/hybrid_retriever.py` — native `include_figures`
- `src/retrieval/semantic_scholar.py` — rate limit fix
- `src/synthesis/claim_ledger.py` — O(1) lookup
- `src/vision/figure_embedder.py` — monkey-patch removed
- `src/vision/vision_ingest.py` — `describe_later` queue
- `src/cache/__init__.py` — CACHE_VERSION=v4
- `src/graph/__init__.py` — factory (NEW FILE for the package)
- `app.py` — uses factory + BM25 persistence
- `.env` — API keys + Neo4j config + GRAPH_BACKEND
- `phase7_baseline_comparison.py` — import-order cleanup
- `phase7_section_writing.py` — import-order cleanup
- `tests/synthesis/test_sectioned_survey.py` — import-order cleanup
- `tests/vision/test_figure_embedder.py` — removed query_with_figures import

## Model Configuration (unchanged)
| Tier | Model | Size | Purpose |
|------|-------|------|---------|
| Fast | `gemma4:e4b` | 9.6 GB | Per-theme, extraction, gap analysis, figure description |
| Reasoning | `qwen3.6:35b` | 23 GB | Cross-theme, critique, arbitration |
