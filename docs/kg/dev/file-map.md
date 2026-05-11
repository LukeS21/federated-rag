---
phase: all
status: reference
created: 2026-05-10
tags:
  - dev
  - architecture
  - files
links:
  - "[[system-overview]]"
  - "[[quick-start]]"
---
Key files overview organized by layer.

## Agents (5 files)

| File | Purpose | Phase |
|---|---|---|
| `agents/base.py` | Base agent class | 6 |
| `agents/survey.py` | Survey mode agent | 6 |
| `agents/sectioned_survey.py` | Sectioned survey agent | 7 |
| `agents/gap_analysis.py` | Gap analysis agent | 7 |
| `agents/literature_discovery.py` | Literature discovery agent | 9 |

## Vision (6 files)

| File | Purpose | Phase |
|---|---|---|
| `vision/ingest.py` | PDF figure extraction and ingestion | 7 |
| `vision/describe.py` | Figure description via multimodal LLM | 7 |
| `vision/classify.py` | Document figure classification | 7 |
| `vision/filter.py` | Figure relevance filtering | 7 |
| `vision/extract.py` | Figure extraction from PDFs | 7 |
| `vision/benchmark.py` | Vision pipeline benchmarking | 7 |

## Synthesis (3 files)

| File | Purpose | Phase |
|---|---|---|
| `synthesis/synthesize.py` | Claim generation and synthesis | 6 |
| `synthesis/merge.py` | Cross-theme merging | 7 |
| `synthesis/validate.py` | Claim validation | 6 |

## Graph (4 files)

| File | Purpose | Phase |
|---|---|---|
| `graph/base.py` | Base graph storage interface | 6 |
| `graph/networkx_store.py` | NetworkX JSON storage adapter | 6 |
| `graph/build.py` | Graph construction from claims | 6 |
| `graph/query.py` | Graph query and traversal | 6 |

## Retrieval (5 files)

| File | Purpose | Phase |
|---|---|---|
| `retrieval/embed.py` | Document embedding | 5 |
| `retrieval/chunk.py` | Document chunking | 5 |
| `retrieval/retrieve.py` | Vector search retrieval | 5 |
| `retrieval/rerank.py` | Result reranking | 5 |
| `retrieval/store.py` | ChromaDB vector store adapter | 5 |

## Ingestion (2 files)

| File | Purpose | Phase |
|---|---|---|
| `ingestion/pdf_loader.py` | PDF text extraction | 5 |
| `ingestion/zotero.py` | Zotero integration | 5 |

## LLM (1 file)

| File | Purpose | Phase |
|---|---|---|
| `llm/client.py` | Unified LLM client wrapper | 5 |

## Security (3 files)

| File | Purpose | Phase |
|---|---|---|
| `security/audit.py` | Security audit logging | 5 |
| `security/scrub.py` | Boundary scrubbing | 5 |
| `security/gliner.py` | GLiNER privacy scanning | 7 |

## Anchoring (1 file)

| File | Purpose | Phase |
|---|---|---|
| `anchoring/anchor.py` | Evidence anchoring and scoring | 6 |

## Cache (3 files)

| File | Purpose | Phase |
|---|---|---|
| `cache/llm_cache.py` | LLM response caching | 6 |
| `cache/claim_cache.py` | Claim ledger persistence | 7 |
| `cache/session_cache.py` | Session-level caching | 6 |

## Tests (12+ test files)

| File | Purpose | Phase |
|---|---|---|
| `tests/test_correctness.py` | Correctness benchmark tests | 6 |
| `tests/test_security.py` | Security audit tests | 5 |
| `tests/test_vision.py` | Vision pipeline tests | 7 |
| `tests/test_synthesis.py` | Synthesis quality tests | 6 |
| `tests/test_anchoring.py` | Evidence anchoring tests | 6 |
| `tests/test_retrieval.py` | Retrieval quality tests | 5 |
| `tests/test_graph.py` | Graph construction tests | 6 |
| `tests/test_baseline.py` | Baseline comparison tests | 7 |
| `tests/test_sectioned.py` | Sectioned survey tests | 7 |
| `tests/test_literature.py` | Literature discovery tests | 9 |
| `tests/test_concurrency.py` | Concurrency tests | 7 |
| `tests/test_ragas.py` | RAGAS evaluation tests | 6 |
