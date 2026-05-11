---
phase: all
status: reference
tags: [architecture, pipeline, overview]
created: 2026-05-10
links: [agent-state, deep-mode-graph, survey-mode-graph, sectioned-survey-graph, hybrid-retriever, llm-provider, knowledge-graph]
---

# System Overview

High-level architecture of the Federated RAG biomedical research system.

## Pipeline

```mermaid
graph TD
    Input[User Query] --> Router[InputRouter]
    Router --> Retriever[HybridRetriever]
    Retriever --> Summarize[Summarize]
    Summarize --> CatDisc[CategoryDiscovery]
    CatDisc --> HumanCheck1[Human Checkpoint]
    HumanCheck1 --> SciNER[SciSpaCy NER]
    SciNER --> Extraction[LLM Extraction]
    Extraction --> KGBuilder[KG Builder]
    KGBuilder --> Drafter[Drafter Agent]
    Drafter --> Critic[Critic Agent]
    Critic --> Arbiter[Arbiter Agent]
    Arbiter --> Anchor1[AnchoringCheck1]
    Anchor1 -->|score >= 0.85| Scrub[Scrub]
    Anchor1 -->|score < 0.85| Arbiter2[ArbiterPass2]
    Arbiter2 --> Anchor2[AnchoringCheck2]
    Anchor2 -->|score >= 0.85| Scrub
    Anchor2 -->|score < 0.85| HumanGate[HumanGate]
    HumanGate --> Scrub
    Scrub --> Output[Final Output]
```

## Execution Modes

| Mode | Latency | Architecture | Use Case |
|------|---------|-------------|----------|
| Quick | ~10s | single-agent | Factual lookup |
| Deep | ~60s | full debate+anchoring | Rigorous synthesis |
| Survey | ~8.7min | thematic clustering + multi-theme debate | Broad literature review |

## Sectioned Mode (Phase 7b)

~50s for 4 IMRaD sections. Multi-turn writing with per-section review and claim ledger dedup.

## Core Design Principles

- **Determinism over probability** — reproducible pipelines, cached embeddings, seeded operations
- **Evidence grounding** — every claim traced to source chunks with citation metadata
- **Schema-less extraction** — LLM structures entities without rigid ontology constraints
- **Heterogeneous multi-agent debate** — different model families resist peer-pressure convergence
- **Air-gap security** — Docker network isolation between public and secure corpora

## Detailed Architecture Notes

- [[agent-state]] — Full AgentState TypedDict
- [[deep-mode-graph]] — 17-node deep reasoning graph
- [[survey-mode-graph]] — 8-node survey pipeline
- [[sectioned-survey-graph]] — 8-node IMRaD pipeline
- [[hybrid-retriever]] — ChromaDB + BM25 fusion
- [[llm-provider]] — Unified LLM interface
- [[knowledge-graph]] — KG storage and querying
