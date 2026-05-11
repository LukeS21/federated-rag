---
phase: 7
status: complete
tags: [phase-7, vision, figures, gemma4]
created: 2026-05-10
links: [figure-extraction, figure-filtering, vision-descriptor-api, cross-modal-retrieval, vision-ingest, model-comparison, gemma4-e4b]
---

# Phase 7a — Vision Pipeline

## Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Figure extraction | Complete | PDF metadata extraction via PyMuPDF |
| Smart filtering | Complete | Classification-first, 80.9% keep rate |
| Vision model integration | Complete | gemma4:e4b via Ollama REST API |
| Figure-to-text embedding | Complete | Vision descriptions embedded in vector store |
| Cross-modal retrieval | Complete | Figures retrievable alongside text chunks |
| Vision ingest hook | Complete | Auto-describe on PDF ingestion |
| UI integration | Complete | Figure thumbnails + descriptions in survey view |

## Model Selection

**Winner: gemma4:e4b**

| Candidate | Result | Reason |
|-----------|--------|--------|
| gemma4:e4b | **Selected** | Already loaded as text model, best biomedical accuracy |
| llava:7b | Rejected | Generic descriptions, rotation overhead |
| qwen3-vl:4b | Rejected | Mid accuracy, rotation overhead |

gemma4:e4b correctly identifies IL-6, CD4, CD8, cytokines, WT/knockout groups from figure labels — critical for biomedical RAG. See [[gemma4-e4b]] and [[model-comparison]] for details.

## Filtering Results

- **47 figures extracted** across 6 biomedical PDFs
- **38 kept** (80.9%) — charts, microscopy, diagrams, schematics
- **9 discarded** — all logos, icons, publisher thumbnails
- **Zero data loss**: no scientific figure was incorrectly discarded
- Classification-first approach (65% weight on type classification) proved superior to size heuristics

## Key Decisions

1. **Classification-first filtering** (65% weight): pixel-count alone is unreliable for biomedical figures; semantic type classification catches small-but-critical figures like cytokine signaling diagrams
2. **No model rotation**: gemma4:e4b already loaded for text tasks → reuse for vision, eliminating ~15-30s rotation overhead per query. See [[model-rotation]] for the abandoned rotation approach.
3. **num_predict workaround**: Ollama multimodal API returns empty responses when `num_predict` is passed. Workaround: send only `temperature`, truncate output post-generation.

## Test Coverage

- **56 vision tests passing** — full regression suite covering extraction, filtering, description, embedding, retrieval, and UI rendering
- Edge cases: empty PDFs, corrupt images, single-figure papers, multi-page figures

## Known Issues

- **Figure descriptions not regenerated on re-ingest** (minor): re-ingesting a PDF re-extracts figures but reuses cached descriptions. Manual cache invalidation required. Tracked for Phase 8 fix.
