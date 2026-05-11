---
phase: 8
status: deferred
created: 2026-05-10
tags:
  - gaps
  - neo4j
  - graph
links:
  - "[[knowledge-graph]]"
  - "[[phase-8-initiation]]"
---
Neo4jStorage adapter deferred to Phase 8. Current NetworkX JSON handles ~500 nodes / ~2000 edges at 6 papers. Neo4j needed for 100K+ edges at publication scale. Interface (`BaseGraphStorage`) already abstracted — one config value swaps all consumers.
