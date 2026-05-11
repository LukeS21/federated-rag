---
phase: [3, 4]
status: reference
tags: [architecture, extraction, ner, scispacy]
created: 2026-05-10
links: [system-overview, knowledge-graph, category-discovery]
---

# Extraction Pipeline

Two-pass entity extraction pipeline for biomedical text.

## Pass 1 — Category Discovery

LLM reads retrieved chunks and discovers:
- `discovered_categories` — thematic groupings of entities
- `key_variables` — experimental variables identified
- `experimental_methods` — methods and assays used

Returns structured JSON for downstream processing.

## Human Checkpoint

`interrupt` after category discovery. User can refine, add, remove, or reorder categories before extraction proceeds. Ensures clinically/domain-relevant groupings.

## Pass 2a — SciSpaCy NER

- Model: `en_core_sci_sm`
- 155+ biomedical entity types: genes, chemicals, diseases, cell lines, organisms, anatomical structures
- Rule-based + statistical hybrid NER
- Fast CPU-bound operation, no LLM latency

## Pass 2b — LLM Structuring

- Normalizes ambiguous entities (e.g., "IL-4" vs "Interleukin-4" → canonical form)
- Structures entities into discovered categories
- Attaches evidence phrases from source text
- Resolves co-references and abbreviations

## Evidence Grounding

Every extracted entity is verified against source chunks:
- `source_paper` — paper identifier attached
- `chunk_index` — specific chunk location attached
- Enables full citation traceability

## Pre-Extraction at Ingest (Phase 4)

Optimization for Survey Mode:
- Entities extracted once during PDF ingest
- Stored as JSON in `projects/default/extractions/`
- Loaded from disk at query time
- Eliminates redundant NER + LLM extraction across queries
