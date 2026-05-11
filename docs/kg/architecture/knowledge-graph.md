---
phase: [3, 4]
status: reference
tags: [architecture, knowledge-graph, networkx, neo4j]
created: 2026-05-10
links: [system-overview, extraction-pipeline, phase-8-initiation]
---

# Knowledge Graph Architecture

Biomedical knowledge graph for entity and relationship storage.

## Abstract Interface

`BaseGraphStorage` defines:

| Method | Description |
|--------|-------------|
| `add_node` | Insert entity node with metadata |
| `add_edge` | Connect nodes with typed relationship |
| `get_neighbors` | 1-hop neighborhood lookup |
| `get_subgraph` | N-hop subgraph extraction |
| `query_relationships` | Edge-type filtered queries |
| `save` | Persist to disk |
| `load` | Restore from disk |

## Current Implementation: NetworkXJSONStorage

- File-based, stored as `project_graph.json`
- ~500 nodes / ~2000 edges at 6 papers
- Suitable for current corpus size

## Future Implementation: Neo4jStorage (Phase 8)

Required for scaling to 100K+ edges. Adds Cypher query support, property indexing, and graph-native traversal performance.

## Node Types

| Type | Example |
|------|---------|
| `material` | LPS, IL-4, dexamethasone |
| `cell_type` | CD4+ T cell, macrophage |
| `cytokine` | TNF-α, IFN-γ |
| `model_system` | BALB/c mouse, C57BL/6 |
| `method` | Flow cytometry, ELISA, RNA-seq |
| `finding` | "IL-4 upregulates GATA3" |
| `paper` | Source document metadata |

## Edge Types

| Type | Semantics |
|------|-----------|
| `measured_via` | Entity quantified by method |
| `observed_in` | Finding observed in model system |
| `expressed_on` | Molecule expressed on cell type |
| `reported_in` | Finding reported in paper |
| `upregulated_by` | Entity upregulated by stimulus |

## Edge Metadata

Every edge carries:
- `extracted_at` — ISO timestamp
- `source_paper` — Paper identifier
- `evidence_phrase` — Verbatim text from source

## Graph Construction Pipeline

```
CategoryDiscovery → Extraction → KGBuilder (co-occurrence edges)
```

## Graph Insights

`compute_graph_insights()` extracts:
- **Central entities** — highest degree nodes
- **Bridge entities** — highest betweenness centrality
- **2-hop subgraph** — formatted as text for the Drafter agent
