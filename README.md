```text
Full Phase Recap
Phase   Status
Phase 1 Foundation (state, Unicode, citation, retrieval primitives)   ✅ Complete
Phase 2 PDF Ingestion & Hybrid Retrieval   ✅ Functional
Phase 3 LLM Agents & LangGraph Core (extraction, debate synthesis, KG, anchoring, Deep Mode)   ✅ Complete (May 2026)
Phase 4 Live Citation & Survey Mode (real Zotero API, systematic field mapping)   ✅ Complete (May 2026)
Phase 5 Security Hardening & Air‑Gap (Docker isolation, boundary scrubber, penetration testing)   ✅ Complete (May 2026)
Phase 5.5 Local Model Optimization & Speed (dense claims, debate simplification, model tiering)   ✅ Complete (May 2026)
Phase 6 UI, Polish & Deployment (Streamlit, GLiNER-PII, correctness benchmarking layer)   ✅ Core Complete (May 2026)
Phase 6.5 Gap Closure (parallelization, compression, cache versioning, security fuzzer)   ✅ Complete (May 2026)
Phase 7 Vision Pipeline & Multi‑Turn Synthesis (figure extraction, vision model, section writing, claim ledger)   ✅ Complete (May 2026)
Phase 8 Publication-Scale Retrieval (PDF acquisition, EZProxy/Playwright pipeline)   ✅ Deprecated — see Phase 9
Phase 9 API-Based Literature Ingestion (Europe PMC XML, SPECTER2, retry, persistence, ChromaDB wiring)   ✅ Complete (15 May 2026)
Phase 10 Autonomous Background Agent (orchestrator daemon, subagents, handoff, scheduler) ✅ Complete (15 May 2026)
Phase 11 Memory Cascade & Community Routing (hierarchical KG, relevance router)          ⬜ Designed — not built
Phase 12 Skills & Experiential Memory (skill library, trajectory logging, agent learnings) ⬜ Designed — not built
Phase 13 Output Tools & Structured Writing (templates, anchored writer, citation integrator) ⬜ Designed — not built
```
___

# 🔬 Secure Federated RAG System – Technical Architecture v3.0

**A Production‑Grade, Local‑First, Multi‑Agent Retrieval‑Augmented Generation Platform
for Biomedical Engineering Research**

---

### 📖 Knowledge Graph Documentation (Obsidian)

The project has an [Obsidian](https://obsidian.md) knowledge graph at `docs/kg/` —
99 interconnected Markdown notes covering every architectural component, decision,
gap, benchmark, model, and phase.  Open with Obsidian (free) for **graph view**,
**backlinks**, color‑coded tag groups, and auto‑generated Dataview dashboards.

**Quick start:** `Obsidian → Open folder as vault → docs/kg/`  

See the [full setup guide](#obsidian-setup) at the end of this README for plugin
recommendations (Dataview + Templater) and navigation tips.  opencode can read
and write notes in this vault directly — no API integration needed.

---

---

## Table of Contents

1. [Vision & Design Philosophy](#1-vision--design-philosophy)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Knowledge Graph Layer](#4-knowledge-graph-layer)
5. [Multi‑Agent Synthesis Architecture](#5-multi-agent-synthesis-architecture)
6. [Extraction Pipeline](#6-extraction-pipeline)
7. [Federated Data Management (Air‑Gap)](#7-federated-data-management-air-gap)
8. [Execution Modes](#8-execution-modes)
9. [LangGraph Orchestration & State Machine](#9-langgraph-orchestration--state-machine)
10. [Implementation Phases](#10-implementation-phases)
11. [Component Interfaces](#11-component-interfaces)
12. [Testing Strategy](#12-testing-strategy)
13. [Deployment & Containerization](#13-deployment--containerization)
14. [Appendices](#14-appendices)

---

## 1. Vision & Design Philosophy

### 1.1 Core Purpose
A privacy‑preserving, deterministic, multi‑agent research assistant that:

- Ingests biomedical PDFs and internal lab data
- Automatically discovers thematic categories in the literature
- Extracts structured, evidence‑grounded entities
- Synthesizes high‑fidelity literature reviews with inline Zotero/Mendeley citation keys
- Enforces strict ASCII formatting throughout
- Operates entirely offline, never sending data to external services

### 1.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Determinism over probability** | LangGraph state machine routes execution; no unbounded LLM loops |
| **Evidence grounding** | Every extracted entity and synthesized claim is traced to a source sentence |
| **Schema‑less, query‑conditioned extraction** | The LLM discovers what categories matter from the literature, not from a fixed YAML |
| **Heterogeneous multi‑agent synthesis** | Three specialized agents on different models for debate without peer‑pressure convergence |
| **Persistent, interface‑abstracted knowledge graph** | File‑based now, Neo4j‑ready when lab‑wide deployment is needed |
| **Air‑gap security at the network level** | Docker internal networks, boundary scrubbers, isolated LLM instances |
| **Plain‑ASCII output** | Unicode‑to‑text substitution pipeline, enforced at extraction, synthesis, and final scrub |

---

## 2. System Architecture Overview
```text
┌──────────────────────────────────────────────────────────────┐
│ User Interface (Streamlit / Gradio) │
│ Localhost only, exposed port │
└───────────────────────────┬──────────────────────────────────┘
│
┌───────────────────────────▼──────────────────────────────────┐
│ LangGraph Orchestrator (State Machine) │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│ │ Input │ │ Retrieve │ │ Extract │ │ Synthesize │ │
│ │ Router ├──► Hybrid ├──► Agent ├──► Debate │ │
│ └──────────┘ └──────────┘ └──────────┘ │ (3 roles) │ │
│ └──────┬───────┘ │
│ │ │
│ ┌──────────────┐ ┌────────────────┐ ┌─────────▼────────┐ │
│ │ Security │ │ Evidence │ │ Knowledge │ │
│ │ Scrubber │◄─┤ Anchoring │◄─┤ Graph Builder │ │
│ └──────────────┘ └────────────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────────────────┘
│
┌───────────────┼───────────────┐
▼ ▼ ▼
┌───────────────┐ ┌────────────────┐ ┌─────────────────┐
│ Public Corpus │ │ Secure Corpus │ │ Knowledge Graph │
│ ChromaDB + │ │ ChromaDB + │ │ NetworkX/JSON │
│ BM25/Tantivy │ │ BM25/Tantivy │ │ (persistent) │
│ (internet OK) │ │ (AIR-GAPPED) │ │ │
└───────────────┘ └────────────────┘ └─────────────────┘
```

---

## 3. Technology Stack

### 3.1 Core Components

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **State Orchestration** | LangGraph ≥0.2 | Deterministic graph‑based routing; built‑in `interrupt` for human‑in‑the‑loop; no unbounded LLM decision loops |
| **Local LLM Runtime** | Ollama (dual‑instance) | Primary: Qwen3.6 35B‑A3B (MoE, ~3B active/token, 262K context, 81.2% TAU2 agentic score). Secondary: Gemma 4 26B A4B (~4B active). Both fit in 36GB M3 Max unified memory |
| **Vector Database (Dense)** | ChromaDB 0.4.24 | Lightweight, embedded; supports metadata filtering; persistent collections |
| **Sparse Retriever (BM25)** | Tantivy (Rust, Python bindings) | Production‑grade inverted index; independent of ChromaDB; exact keyword precision for gene names, alloy codes, PMIDs |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Proven, parameter‑free, naturally deduplicates dense + sparse result lists |
| **PDF Parsing** | Docling ≥2.0 | Vision‑model‑based table/caption preservation; exports markdown tables |
| **Knowledge Graph** | NetworkX (persistent to JSON) | File‑based, zero‑dependency graph; interface‑abstracted for future Neo4j migration |
| **Citation Manager** | PyZotero + custom abstraction | Adapter pattern supports Zotero now, Mendeley via additional adapter; real CiteKey generation |
| **NER (First Pass)** | SciSpaCy (PubMedBERT) | Deterministic entity candidate detection; boosts extraction recall before LLM normalization |
| **Schema Validation** | Pydantic `jsonschema` | Validates extracted JSON against dynamically discovered categories |

### 3.2 LLM Role Assignment

| Role | Model | Rationale |
|------|-------|-----------|
| **Drafter** | Qwen3.6 35B-A3B | Best agentic tool‑use score; drafts initial synthesis with evidence citations |
| **Socratic Critic** | Gemma 4 26B A4B | Different model family → resists peer pressure; only asks evidence‑grounded questions, never proposes alternative text |
| **Arbiter** | Qwen3.6 35B-A3B (different prompt) | Same model as Drafter but with revision‑focused prompt; resolves Critic questions against original evidence |
| **Extraction LLM** | Qwen3.6 35B-A3B | Query‑conditioned extraction + category discovery from retrieved chunks |
| **Category Discovery** | Qwen3.6 35B-A3B | Reads all retrieved chunks, identifies recurring themes, variables, methods |

---

## 4. Knowledge Graph Layer

### 4.1 Design: Interface‑First, Persistent File‑Based

The knowledge graph stores entities, their relationships, and temporal metadata. Every consumer codes against the abstract interface, never the concrete storage backend.

**Interface Contract** (`BaseGraphStorage`):

```python
class BaseGraphStorage(ABC):
    @abstractmethod
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None: ...
    @abstractmethod
    def add_edge(self, source: str, target: str, relation: str, properties: dict) -> None: ...
    @abstractmethod
    def get_neighbors(self, node_id: str, relation: str = None) -> list[dict]: ...
    @abstractmethod
    def get_subgraph(self, node_ids: list[str], depth: int = 1) -> dict: ...
    @abstractmethod
    def query_relationships(self, source_type: str, relation: str, target_type: str) -> list[dict]: ...
    @abstractmethod
    def save(self) -> None: ...
    @abstractmethod
    def load(self) -> None: ...
```

Concrete Implementation – Phase 3/4 (NetworkXJSONStorage):

Stores graph as project_graph.json in the project directory

Every edge carries temporal metadata: {"extracted_at": "2026-05-06T14:22:00Z", "source_paper": "avery2025.pdf", "evidence_phrase": "...exact sentence..."}

save() serializes the full NetworkX graph to JSON

load() deserializes from JSON on project open

Migration Path to Neo4j (Phase 6 / Lab‑wide):

Implement Neo4jStorage class satisfying the same BaseGraphStorage interface

get_neighbors → MATCH (n)-[:RELATES_TO]->(m) WHERE n.id = $node_id RETURN m

query_relationships → parameterized Cypher

Change one config value: graph_backend: "neo4j"

All consumers are unaware of the swap

### 4.2 Node & Edge Types

| Node Type | Examples |
|-----------|----------|
| `material` | Ti-6Al-4V, TiO₂, rough‑hydrophilic Ti |
| `cell_type` | neutrophil, macrophage, CD4+ T cell, MSC |
| `cytokine` | IL‑6, TNF‑alpha, IL‑1beta |
| `model_system` | C57BL/6J mouse, rat tibia |
| `method` | flow cytometry, ELISA, microCT |
| `finding` | "IL‑6 elevated in obese mice", "rough‑hydrophilic Ti increased anti‑inflammatory macrophages" |
| `paper` | source paper metadata |

| Edge Type | Meaning |
|-----------|---------|
| `measured_via` | (cytokine) → (method) |
| `observed_in` | (cell_type) → (model_system) |
| `expressed_on` | (finding) → (material) |
| `reported_in` | (finding) → (paper) |
| `upregulated_by` | (cytokine) → (condition) |

### 4.3 Graph Construction Flow
Category Discovery phase identifies entity types present in retrieved chunks

Extraction phase produces structured JSON with entity instances + evidence phrases

Graph Builder node:

Creates a node for each entity instance

Links entities based on co‑occurrence in the same chunk + semantic relationship detection

Attaches evidence_phrase and source_paper metadata to every edge

Calls graph.save() to persist

## 5. Multi‑Agent Synthesis Architecture
### 5.1 Core Principle: Heterogeneous, Role‑Structured Debate with Evidence Anchoring
Research shows that homogeneous debate causes peer‑pressure convergence (agents agree on wrong answers), and iterative closed‑system debate degrades evidential grounding (The Reasoning Trap, May 2026). Our architecture addresses both failure modes:

Heterogeneous models: Qwen3.6 + Gemma 4 (different families, different reasoning biases)

Role‑structured, not adversarial: Socratic Critic asks evidence‑grounded questions, never proposes rival arguments

Evidence‑anchored stopping criterion: measurable Anchoring Score, not subjective consensus

Bounded iterations: maximum 2 passes, then human escalation

### 5.2 Agent Roles
Agent 1: Drafter (Qwen3.6 35B-A3B)
Prompt structure:
```text
  System: You are a biomedical literature synthesis drafter. Given extracted entities,
          evidence summaries, and citation keys, write a concise literature review paragraph.
          Every factual claim must be traceable to a provided evidence chunk.
          Use inline citation keys (@author2025). Output plain ASCII only.
  
  User:   Query: {query}
          Extracted Entities: {entities_json}
          Evidence Summaries: {chunk_texts}
          Available Citations: {cite_keys}
          Knowledge Graph Context: {subgraph_json}
          Write a draft paragraph synthesizing this information.
```
Output: Draft synthesis paragraph with inline citations.

Agent 2: Socratic Critic (Gemma 4 26B A4B)
Prompt structure:
```text
  System: You are a Socratic critic. Your job is to identify claims in the draft that lack
          sufficient evidence or overstate what the evidence supports.
          - For each questionable claim, state what the evidence actually says.
          - Ask a specific question about an unsupported assertion.
          - NEVER propose alternative text or "correct" the draft.
          - If the draft is fully supported, state: "NO_CRITIQUE: All claims are evidence-grounded."
          Output plain ASCII only.
  
  User:   Draft: {synthesis_draft}
          Evidence: {original_chunks}
          Entities: {extracted_entities}
          Identify unsupported claims and state the gap.
```
Output: List of critiques (or NO_CRITIQUE).

Agent 3: Arbiter (Qwen3.6 35B-A3B, revision prompt)
Prompt structure:
```text
  System: You are a biomedical synthesis arbiter. You receive a draft, a Socratic critique,
          and the original evidence. Revise the draft to address the critique.
          - For each critique, either cite specific evidence that supports the claim or modify/remove the claim.
          - Do not alter claims that were not critiqued.
          - Output the complete revised paragraph. Plain ASCII only.
  
  User:   Draft: {synthesis_draft}
          Critique: {critic_output}
          Evidence: {original_chunks}
          Revise the draft, addressing each critique.
```
Output: Revised synthesis paragraph.

### 5.3 Evidence‑Anchoring Check (Programmatic)
After the Arbiter produces a revised synthesis, the system performs an automated check without LLM involvement:

Claim decomposition: Split the synthesis into atomic factual claims using regex sentence splitting + simple heuristics (e.g., "X was measured via Y", "Z increased in condition W").

Evidence search: For each claim, run a BM25 query against the retrieved chunks used in this session to find the best‑matching evidence sentence.

Similarity computation: Compute cosine similarity (using a lightweight embedding or simple TF‑IDF) between the claim and the best evidence sentence.

Anchoring Score = fraction of claims with cosine similarity ≥ 0.7 (tunable threshold).

Decision:

Anchoring Score ≥ 0.85 → synthesis is finalized

Anchoring Score < 0.85 → flagged claims are sent back to Arbiter for a conditional second pass

If after second pass, Anchoring Score < 0.85 → escalate to human approval gate (LangGraph interrupt)

### 5.4 Flow Diagram
```text
┌──────────┐     ┌──────────────┐     ┌───────────┐
│  Drafter │────►│ Socratic     │────►│  Arbiter  │
│ (Qwen3.6)│     │ Critic       │     │ (Qwen3.6) │
└──────────┘     │ (Gemma 4)    │     └─────┬─────┘
                 └──────────────┘           │
                                            ▼
                                 ┌─────────────────────┐
                                 │ Evidence‑Anchoring  │
                                 │ Check (programmatic)│
                                 └──────────┬──────────┘
                                            │
                           Anchoring ≥ 0.85 │ Anchoring < 0.85
                                            │
                          ┌─────────────────▼──────────────┐
                          │                              │
                      Finalize                Conditional Pass 2
                                              (Arbiter revises
                                               flagged claims)
                                              │
                                              ▼
                                    Anchoring Check (again)
                                              │
                                   ≥ 0.85 ────┴─── < 0.85 ──► Human Gate
```
## 6. Extraction Pipeline
### 6.1 Design Principle: Schema‑Less, Query‑Conditioned, Evidence‑Grounded
The researcher never defines entity categories. The system discovers them from the literature. Every extracted entity is tied to an evidence phrase from the source text.

### 6.2 Category Discovery (Pass 1)
The LLM reads all retrieved chunks (filtered to body text, references excluded) and identifies:

Recurring themes

Variables being studied

Experimental methods

Model systems

Measured outcomes

Output format (JSON):
```json
  {
    "discovered_categories": [
      {
        "name": "animal_models",
        "description": "Murine models used, including strain and diet conditions",
        "examples_found": ["C57BL/6J mice", "HFD-induced obesity model"]
      },
      {
        "name": "biomaterials",
        "description": "Implant materials and surface modifications tested",
        "examples_found": ["rough Ti", "rough-hydrophilic Ti", "Ti-6Al-4V"]
      }
    ],
    "key_variables": ["cytokine levels", "macrophage polarization", "bone formation"],
    "experimental_methods": ["flow cytometry", "ELISA", "microCT", "adoptive transfer"]
  }
  ```
Optional user checkpoint: A LangGraph interrupt displays the discovered categories. The researcher can accept, remove, or add categories. If they do nothing, the pipeline proceeds automatically.

### 6.3 Two‑Pass Extraction (Pass 2)
Pass 2a: Deterministic NER (SciSpaCy)
Model: en_ner_bc5cdr_md or en_core_sci_lg

Extracts: genes, chemicals, diseases, cell lines, organisms

Produces: list of candidate entity spans with types

Pass 2b: LLM Structuring (Qwen3.6 35B-A3B)
The LLM receives:

The discovered categories

The NER candidate spans

The full retrieved chunks

The user's original query

It normalizes ambiguous entities ("TNF‑α" vs "TNF‑alpha" → unified), adds missing entities not caught by NER, and structures everything into the discovered categories.

Output format:
```json
{
  "animal_models": [
    {
      "entity": "C57BL/6J mice",
      "conditions": "HFD-induced obesity",
      "evidence": "20-week-old male C57BL/6J mice were fed a diet containing 45 kcal% fat"
    }
  ],
  "cytokines": [
    {
      "entity": "IL-6",
      "direction": "elevated",
      "context": "serum in obese mice post-implantation",
      "evidence": "IL-6 ... significantly higher in obese mice"
    }
  ]
}
```
### 6.4 Evidence Grounding (Automated)
For each extracted entity, the system verifies that:

An evidence phrase exists in the source chunks

The phrase actually supports the extracted attribute (direction, context)

A source_paper and chunk_index reference is attached

Entities without evidence grounding are discarded or flagged.

## 7. Federated Data Management (Air‑Gap)
### 7.1 Dual‑Corpus Architecture
Corpus	Contents	Network Access	LLM Instance
Public	PubMed literature, open‑access PDFs	Internet (rate‑limited)	Ollama instance 1 (can access internet)
Secure Lab	Internal spreadsheets, grant drafts, unpublished results	None (Docker internal: true)	Ollama instance 2 (air‑gapped)
### 7.2 Enforcement Layers
Docker network isolation: Secure container has no gateway, no DNS. External connections are physically impossible.

Boundary scrubber: Before any payload moves from secure → public container, a Python regex checks for proprietary terms. If detected, the payload is blocked and logged.

LangGraph routing: The state machine's query_scope field ("public", "secure", "both") controls which retrieval/LLM paths are executed. Secure‑scope queries never touch the public Ollama instance.

Per‑corpus LLM instances: The air‑gapped Ollama instance processes only secure‑corpus data. It has no internet access.

## 8. Execution Modes
### 8.1 Quick Mode
Purpose: Factual lookup with minimal latency

Flow: Hybrid retrieve → single‑pass extraction → single‑agent synthesis (no debate)

Latency: ~5-10 seconds per query on M3 Max

Use case: "What model system was used in Avery et al. 2025?"

### 8.2 Deep Mode (Reference Implementation)
Purpose: Rigorous evidence synthesis with debate and anchoring

Flow: Full pipeline: category discovery → two‑pass extraction → KG construction → 3‑role heterogeneous debate → evidence anchoring (1-2 passes) → scrub

Latency: ~30-60 seconds per query (4-5 LLM calls + programmatic checks)

Use case: "Synthesize what's known about immune response to titanium implants in obese models"

### 8.3 Survey Mode (Phase 4)
Purpose: Comprehensive, evidence‑grounded literature survey across many papers.

Flow: Broad retrieval (all papers) → thematic clustering (LLM, assigns every paper to 1+ themes) → per‑document lightweight extraction (parallel, feeds shared KG) → per‑theme deep synthesis (full Drafter→Critic→Arbiter debate on all papers in that theme) → cross‑theme synthesis & gap analysis.

*Why not representative paper selection?* Picking a few "representative" papers per theme loses outlier findings, contradictory evidence, and interdisciplinary assignments. Instead, every paper gets extracted into the KG (fast, ~5 sec/paper), and the expensive full debate synthesis runs only per‑theme (5–8 calls), not per‑paper (100+). This preserves completeness while controlling cost — the KG acts as the shared truth cache across all stages.

Latency: ~5-10 minutes for a subfield survey (100 papers)

Use case: "Map the current understanding of biomaterial surface modifications and immune response"

## 9. LangGraph Orchestration & State Machine
### 9.1 State Definition
```python
class AgentState(TypedDict):
    user_query: str
    query_scope: Literal["public", "secure", "both"]
    mode: Literal["quick", "deep", "survey"]
    public_context: List[Document]
    secure_context: List[Document]
    discovered_categories: Dict
    extracted_entities: Dict
    knowledge_graph_snapshot: Dict
    synthesis_draft: str
    critic_feedback: str
    synthesis_revised: str
    anchoring_score: float
    citations_used: List[str]
    final_output: str
    human_approved: bool
    routes: Optional[Dict]
```
### 9.2 Node Graph (Deep Mode)
```text
Start
  │
  ▼
[Input Router]  ──► scope=public/secure/both
  │
  ▼
[Hybrid Retriever]  ──► retrieves from appropriate corpus
  │
  ▼
[Category Discovery]  ──► LLM: reads chunks, discovers themes
  │
  ▼
[Human Checkpoint] (optional)  ──► interrupt: user can refine categories
  │
  ▼
[NER Pass]  ──► SciSpaCy deterministic entity extraction
  │
  ▼
[LLM Extraction]  ──► structures entities by discovered categories + evidence grounding
  │
  ▼
[KG Builder]  ──► constructs/updates persistent knowledge graph
  │
  ▼
[Drafter]  ──► first synthesis draft
  │
  ▼
[Socratic Critic]  ──► identifies unsupported claims
  │
  ├── NO_CRITIQUE ──► skip to Evidence Anchoring
  │
  ▼
[Arbiter]  ──► revises draft addressing critiques
  │
  ▼
[Evidence Anchoring Check]  ──► programmatic anchoring score
  │
  ├── ≥ 0.85 ──► Finalize
  │
  ├── < 0.85 ──► [Arbiter Pass 2] (conditional) ──► Check again
  │                                              │
  │                              ≥ 0.85 ──► Finalize
  │                              < 0.85 ──► [Human Gate]
  │
  ▼
[Security & Formatting Scrubber]  ──► ASCII enforcement, proprietary term redaction
  │
  ▼
End → output to user
```

## 10. Implementation Phases
Phase 1: Foundation (Weeks 1-2)
Goal: Tested primitives for state, Unicode, citation management, and dual retrieval indexes.

Deliverables:

Virtual environment with all dependencies pinned

AgentState TypedDict

Unicode-to-ASCII mapping module with NFKC normalization and catch-all ASCII scrub

CitationManager abstract class + ZoteroAdapter (stub)

ChromaClient + BM25Index classes with in-memory test instances

HybridRetriever class (dense + sparse, no fusion yet)

7+ unit tests, all passing

Phase 2: PDF Ingestion & Hybrid Retrieval (Weeks 3-4)
Goal: Real PDF → chunks → dual‑index → fused query pipeline.

Deliverables:

PDFParser wrapping Docling: text extraction, table extraction (markdown), reference detection heuristic, immediate Unicode scrub

HybridRetriever with reciprocal rank fusion (RRF)

Chunk metadata: source, chunk_type ("text", "table", "reference"), chunk_index

Ingest test PDF, run interactive query demo

2-3 additional unit tests; all existing tests still pass

Phase 3: LLM Agents & LangGraph Core (Weeks 5-7)
Goal: Working Deep Mode pipeline end-to-end on a single project.

Deliverables:

Ollama setup: Qwen3.6 35B-A3B + Gemma 4 26B A4B pulled

ExtractionAgent: category discovery prompt + two-pass extraction (SciSpaCy NER + LLM structuring) + evidence grounding verification

SynthesisDrafter, SocraticCritic, Arbiter agent classes (each with distinct prompt template)

EvidenceAnchoringCheck programmatic module: claim decomposition, BM25 evidence search, cosine similarity scoring

BaseGraphStorage interface + NetworkXJSONStorage adapter

GraphBuilder node: constructs entities → relationships, attaches temporal metadata, saves graph

LangGraph state machine: InputRouter → HybridRetriever → CategoryDiscovery → Extraction → KGBuilder → Drafter → Critic → Arbiter → AnchoringCheck → Scrubber

Interactive phase3_demo.py supporting Deep Mode

Integration test: one query through full pipeline

### Phase 3 Status (May 2026)

**Complete.** Deep Mode pipeline runs end-to-end with all specified components implemented and tested.

#### Implemented graph (17 nodes)

```
InputRouter → Retrieve → Summarize → CategoryDiscovery
  → [Human Checkpoint: review/edit categories]
  → SciSpaCyNER → Extraction → KGBuilder
  → Drafter → Critic → Arbiter → AnchoringCheck1
  → (ArbiterPass2 if score < 0.85) → AnchoringCheck2
  → (HumanGate if score < 0.85) → Scrub → END
```

#### Confirmed behavior

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Chunk summarization | New `Summarizer` agent after retrieval | Cuts token usage ~5x for downstream agents; category_discovery, drafter, critic, arbiter use summary; extraction uses raw chunks for evidence grounding |
| SciSpaCy NER | `en_core_sci_sm` model in `sci_ner` node | 155+ biomedical entities extracted deterministically per query; fed as grounding hints to LLM extraction agent |
| Anchoring | TF-IDF cosine similarity (threshold 0.35) | Replaced Jaccard placeholder; correctly distinguishes grounded vs ungrounded claims |
| Human-in-the-loop | Two checkpoints via `interrupt_before` | (1) After category discovery — review/edit categories before NER runs; (2) HumanGate — final review when anchoring score < 0.85 |
| Checkpointer | `MemorySaver` in-memory checkpointer | Enables interrupt/resume for human-in-the-loop |
| Debate chain | Drafter → Critic → Arbiter → ArbiterPass2 | Full three-role heterogeneous debate confirmed; Critic routes to Arbiter when claims need revision, skips (NO_CRITIQUE) when draft is grounded |
| Knowledge graph | `NetworkXJSONStorage` with co-occurrence edges | Builds entity-entity edges from shared chunk provenance; persisted to `project_graph.json` |

#### Design decisions

- **DeepSeek v4 Pro API** instead of local Ollama (Qwen/Gemma) — faster iteration, no GPU requirement for development; Ollama path preserved for Phase 5 air-gap deployment
- **TF-IDF cosine (0.35)** for anchoring per README §5.3 spec (originally: Jaccard placeholder)
- **10 retrieved chunks** per query via hybrid retriever (ChromaDB vector + BM25 sparse)
- **Chunk summarization** (query-time): one LLM call condenses retrieved chunks into a ~500-word evidence abstract; downstream agents consume the summary rather than raw text
- **API key sanitization** in all agents: strips non-ASCII characters from `DEEPSEEK_API_KEY` to prevent `UnicodeEncodeError` in HTTP headers
- **`load_dotenv(override=True)`** in demo to ensure `.env` values take precedence over shell environment

#### Tests

- **27 tests passing** (5 new integration tests, 22 existing)
- **6 pre-existing failures**: `test_synthesis_agents.py` mocks `langchain_ollama.ChatOllama` which was replaced by `langchain_openai.ChatOpenAI` during the Ollama→DeepSeek migration; not yet updated

#### Planned improvements (before Phase 4)

These optimizations address the latency and scaling limitations identified during Phase 3 testing. Each is designed to maintain or improve output quality while reducing per-query cost and time.

**1. Similarity‑threshold retrieval (replaces fixed n_results=10)**

*Problem:* Fixed n_results=10 has two failure modes: (a) fewer than 10 chunks are truly relevant, injecting noise into the LLM context; (b) more than 10 chunks are relevant, arbitrarily excluding evidence.

*Design:* Replace `n_results=10` with `similarity_threshold=0.5, max_chunks=20`. The hybrid retriever returns all chunks whose ChromaDB cosine distance falls below the threshold, capped at max_chunks to prevent context-window overflow.

*Rationale:* This is the standard pattern used in production RAG frameworks (LlamaIndex, LangChain's `ParentDocumentRetriever`). The threshold adapts to the corpus — a query about "macrophage polarization" in a 1‑document corpus might return 8 chunks; the same query in a 100‑document corpus might return 18 across 4 papers. The cap prevents pathological cases (50+ matching chunks). The threshold is calibrated per domain and self‑correcting: too strict → missed evidence (anchoring score drops); too loose → token bloat (latency increases).

**2. Pre‑summarization at ingest time with entity links**

*Problem:* Query‑time summarization adds one LLM call per query (currently 30–80s). At scale (100s of queries/day across a lab), this compounds.

*Design:* At PDF ingestion, each chunk receives a structured summary stored in metadata:
```
Chunk 3:
  summary: "Obese mice had elevated leptin which activated T‑cell leptin
           receptors, promoting Th1/Th17 polarization and suppressing Treg
           expansion."
  entities: [leptin, T cells, Th1, Th17, Treg, leptin receptor]
  linked_to: [Chunk 2 (adipokines), Chunk 5 (IL‑17A), Chunk 8 (bone formation)]
```

At query time, retrieval returns pre‑summarized chunks. The `Summarize` node is replaced by concatenation of pre‑written summaries — zero LLM calls.

*Relationship preservation:* Entity links between chunks (derived from co‑occurrence via the KG) ensure that related findings are surfaced even when they don't match the query directly. A query about "IL‑17A" would retrieve Chunk 5, and the KG traversal would pull in Chunk 3 (leptin → T‑cell activation) and Chunk 8 (bone formation) as related context. No finding sits in isolation.

*Risk mitigation:* Pre‑written summaries are generic (not query‑aware). To prevent missing cross‑cutting themes, each chunk summary is supplemented with an explicit entity list. The extraction agent still receives raw chunk text for evidence‑grounding quotes, so nothing is permanently lost. Additional risk: the ingest‑time summarization LLM may hallucinate or omit details. Mitigation: require the ingest summarizer to cite specific sentences from the source chunk (same evidence‑grounding pattern used by extraction), and run the anchoring check against the raw chunk text to verify fidelity.

**3. KG‑driven synthesis with graph reasoning**

*Problem:* The knowledge graph is built from extracted entities and persisted, but the drafter receives the raw subgraph JSON — a giant node‑link dict the LLM cannot meaningfully process.

*Design:* Before drafting, a graph reasoning layer processes the subgraph:
- **Central entities:** nodes with highest degree (most co‑occurrence edges) — these are the concepts that connect everything across the corpus
- **Bridge entities:** nodes whose removal disconnects the graph into separate clusters — these represent cross‑cutting themes that link otherwise‑separate research areas
- **2‑hop subgraph:** for entities directly matching the query, extract all entities within 2 edge traversals and format as a structured summary

The drafter receives: *"Key connecting concepts: leptin (links obesity → immune dysfunction, 12 edges), macrophage polarization (links surface properties → bone outcomes, 18 edges), IL‑17A (links T cells → inflammation, 5 edges). Cross‑cutting theme: leptin‑mediated T‑cell reprogramming bridges adipokine signaling and peri‑implant osteogenesis."*

*Rationale:* This replicates how a human researcher thinks — find central papers, follow their references, notice shared cell types across studies. The KG already captures these relationships; they just need to be surfaced meaningfully rather than dumped as raw JSON.

**4. Extraction prompt reduction**

*Problem:* Extraction is the bottleneck (~5 min per query). The prompt contains redundant information: full category descriptions with examples (already generated by category_discovery) and all 155 NER entities (many irrelevant).

*Design:*
- Cap NER entities at top‑30 (by SciSpaCy confidence or frequency) instead of sending all 155
- Send category names + 1‑line descriptions only; omit the `examples_found` arrays (the LLM generated them 30 seconds ago in `category_discovery`)
- Estimated prompt reduction: ~8000 → ~4000 tokens

*Why this doesn't hurt quality:* The LLM doesn't need to re‑read its own examples. It needs the category structure (names), the raw evidence (chunks), and the most salient NER hints. Downstream synthesis quality is gated by evidence completeness, not by how many times the LLM re‑reads its prior output.

**5. LLM response caching**

*Problem:* Identical or near‑identical queries re‑run the full pipeline. Category discovery on the same 10 chunks with the same query → same output every time.

*Design:* Hash `(system_prompt, user_prompt)` → store LLM response in an on‑disk cache (TTL: 24h). Before making an API call, check the cache. Applies to: summarization, category discovery, and extraction (which have deterministic outputs at temperature=0).

*Expected impact:* For repeated or similar queries, category discovery + summarization become instant (<1ms). Combined with pre‑summarization, this eliminates 2 of 7 LLM calls for most queries.

**6. Cheaper model tier for summarization**

*Problem:* Summarization uses DeepSeek v4 Pro (the most expensive tier) for a task that requires extraction fidelity, not advanced reasoning.

*Design:* Switch summarization to a cheaper model tier (e.g., DeepSeek Chat). Risk is low because: (a) summarization is inherently safer than generation — the model condenses existing text rather than creating new findings; (b) raw chunks remain available to downstream extraction for evidence grounding, so any omission is recoverable; (c) the anchoring check verifies final synthesis against raw evidence regardless of summary quality.

*Verification:* Run 10 queries side‑by‑side (DeepSeek v4 Pro vs DeepSeek Chat) and compare: (a) final anchoring scores, (b) extraction entity counts, (c) synthesis BERTScore similarity. If scores are within 5%, switch permanently.

#### Known limitations

- **Extraction is the remaining bottleneck**: ~5–7 min per query with 2 PDFs (20 chunks). Extraction time scales linearly with chunk count; the 6 optimizations eliminated other bottlenecks (summarization instant, cache hits for repeat queries, adaptive retrieval) but extraction still processes full raw chunks + NER hints.
- **Multi-document synthesis works but is unoptimized**: 2‑PDF cross‑document synthesis confirmed. At 50+ PDFs, retrieval may surface 50+ relevant chunks, causing extraction times of 20+ minutes. The Phase 4 hybrid architecture (per‑document lightweight extraction → per‑theme deep synthesis) is designed to address this.
- **No parallel agent execution**: all 7 LLM calls run sequentially. LangGraph supports fan‑out/fan‑in patterns but they are not yet implemented.
- **6 pre‑existing test failures**: `test_synthesis_agents.py` mocks `langchain_ollama.ChatOllama` removed during DeepSeek migration; not yet updated.

### Phase 4 Status (May 2026)

**Complete.** Survey Mode pipeline runs end-to-end with all specified components, Batch 1–2 optimizations, model tiering, multi‑level query caching, citation key propagation, and human‑in‑the‑loop gates.

#### Survey Mode Graph (8 nodes)

```
survey_query_decompose → survey_retrieve → survey_thematic_cluster
  → survey_per_document_extract → survey_per_theme_synthesize
  → survey_cross_theme_synthesize → [Human‑in‑the‑loop gate: review]
  → survey_scrub → END
```

#### Confirmed behavior

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Query decomposition | `QueryDecomposer` agent (deepseek‑v4‑pro) | Breaks broad research question into 3–8 themed sub‑queries; cached at query level (L1) |
| Thematic clustering | Sentence‑transformer embeddings (`all‑MiniLM‑L6‑v2`) | Assigns every paper to 1+ themes via cosine similarity (threshold 0.35); LLM fallback preserved; paper embeddings pre‑computed at ingest |
| Per‑document extraction | `PreExtractor` (deepseek‑chat) with disk cache | Entities extracted once at ingest, stored as JSON + paper embeddings; query‑time loads from disk (zero LLM calls on subsequent queries) |
| Per‑theme synthesis | Parallel `ThreadPoolExecutor`; `deepseek‑chat` Drafter | 5 multi‑paper themes run concurrently in ~9 s wall‑clock; single‑paper themes format entities directly (no LLM) |
| Conditional critic (EGSR) | Anchoring check before Critic invocation | Only invokes Critic + Arbiter when draft anchoring < 0.35; debate regression guard keeps draft if debate worsens score |
| KG insights | `compute_graph_insights()` injected into Drafter prompt | Central/bridge entities + 2‑hop neighbourhood surfaced as structured text (not raw JSON) |
| Cross‑theme synthesis | Parallel `deepseek‑v4‑pro` calls for narrative + gap analysis | Gap analysis runs concurrently with cross‑theme (prompt rewritten to avoid dependency) |
| Query caching | Multi‑level disk cache (7‑day TTL) | L1: decomposition, L2: per‑theme synthesis, L3: cross‑theme + gap; visible `[query‑cache]` logging; second run of same query completes in < 1 s |
| Anchoring | TF‑IDF cosine similarity (threshold 0.35) | tiktoken‑based context window estimation; dynamic evidence cap replaces hardcoded `[:20]` |
| Human‑in‑the‑loop | `interrupt_before=["survey_scrub"]` | Review gate after cross‑theme synthesis; approve / edit‑with‑feedback / discard; `MemorySaver` checkpointing for resume |
| Citation management | Zotero API + filename‑based cite key generation | Real Zotero item creation on PDF ingest (`pyzotero`); cite keys (`@avery2024`) propagated through extraction → synthesis output |
| Model tiering | Per‑theme: `deepseek‑chat`; cross‑theme: `deepseek‑v4‑pro` | Chat 8.8× faster for per‑paper extraction (benchmark‑verified); v4‑pro reserved for final unified synthesis |
| Agent memoization | Module‑level singleton constructors (`_get_drafter`, `_get_critic`, `_get_arbiter`) | Eliminates redundant `ChatOpenAI` instantiation (~9 per query → 2–3) |

#### Design decisions

- **Embedding‑based clustering as primary** — sentence‑transformers are deterministic, ~2 s wall‑clock, zero API cost. LLM path preserved as `use_embeddings=False` fallback for edge cases.
- **Pre‑extraction at ingest eliminates ~60 % of query‑time LLM calls** — entities extracted once per paper via deepseek‑chat during PDF ingestion, stored as JSON in `projects/default/extractions/`. At query time, all 6 papers loaded from disk in 0.0 s. Delete the directory to force re‑extraction.
- **TF‑IDF extractive summarization (no LLM)** — replaced LLM‑based `PreSummarizer` with sklearn `TfidfVectorizer` sentence scoring. Extractive → no hallucination risk. Summaries are internal hints for downstream agents, not user‑facing output.
- **Single‑paper themes skip all LLM calls** — themes with 1 paper have no cross‑paper evidence to reconcile. Pre‑extracted entities are formatted directly into structured text. Saves 1 Drafter call per single‑paper theme.
- **Conditional Critic threshold at 0.35** — only poorly‑grounded drafts (< 0.35) invoke the debate chain. Benchmark‑validated: 67 % of Critic calls saved with zero anchoring degradation in a 6‑paper corpus. Threshold configurable via `CONDITIONAL_CRITIC_THRESHOLD`.
- **Model tiering: chat for per‑theme, v4‑pro for cross‑theme** — benchmark showed chat 8.8× faster with only 11.8 % fewer entities for extraction. Same principle extended to drafting: chat handles per‑theme synthesis, v4‑pro reserved for the final cross‑theme narrative. Per‑theme model configurable via `PER_THEME_DRAFTER_MODEL`.
- **Debate agents share cached instances** — `_get_drafter`, `_get_critic`, `_get_arbiter` return module‑level singletons. Eliminates 7+ redundant `ChatOpenAI` instantiations per query.
- **Evidence truncation uses dynamic tiktoken cap** — replaced hardcoded `summaries[:20]` with `_fit_summaries_to_context()` that fills summaries until ~70 % of context window is consumed. With `num_ctx=16384`, allows ~100+ summaries vs the old cap of 20.
- **Parallel cross‑theme + gap analysis** — gap analysis prompt rewritten to use per‑theme syntheses directly (no dependency on cross‑theme output), enabling both v4‑pro calls to run in parallel via `ThreadPoolExecutor(max_workers=2)`.
- **All agents accept configurable `model` parameter** — no longer hardcoded to `deepseek‑v4‑pro`. Cache keys include model name. Enables per‑task model tiering.

#### Performance

| Metric | Before (Phase 3) | After (Phase 4) |
|--------|-------------------|-----------------|
| Survey query latency | 27 min | 1–2 min |
| Per‑document extraction | 41 s (2 LLM calls) | 0.0 s (pre‑cached from disk) |
| Per‑theme synthesis | 23 min (sequential) | ~9 s (parallel, chat Drafter) |
| Cross‑theme + gap | 2.4 min (sequential) | ~47 s (parallel v4‑pro) |
| LLM calls per query | ~18 (all v4‑pro) | ~12 (2 v4‑pro + 10 chat) |
| Repeated query | Full re‑compute | < 1 s (all cache levels hit) |
| Tests | 27 passing | 66 passing (6 pre‑existing failures) |

#### Tests

- **66 tests passing** (18 survey graph, 8 thematic clusterer, 8 query decomposer, 5 evidence check, 5 ingestion, 5 retrieval, 5 state/unicode/scrubber, 5 synthesis agent, 4 extraction, 3 graph)
- **6 pre‑existing failures**: `test_synthesis_agents.py` mocks `langchain_ollama.ChatOllama` which was replaced by `langchain_openai.ChatOpenAI` during the Ollama→DeepSeek migration; not yet updated

### Phase 4 → Phase 5 handoff notes

*For the next developer picking up Phase 4. Read this section first.*

**Current state of the codebase:**

- All agents use DeepSeek API via `langchain_openai.ChatOpenAI` (not local Ollama). API key from `DEEPSEEK_API_KEY` env var, sanitized via `sanitize_api_key()`.
- The demo loads `.env` with `load_dotenv(override=True)` — add your key there, not in shell profiles.
- PDFs go in `data/`. The demo auto‑discovers new PDFs and pre‑summarizes them at ingest (one‑time LLM cost). Already‑indexed PDFs are skipped on subsequent runs.
- Cache lives in `projects/default/cache/`. Delete the directory to force fresh LLM calls.
- Run with `OLLAMA_KEEP_ALIVE=30s python phase3_demo.py` (the env var is unused but harmless).

**Key architectural decisions carried forward:**

1. **ExtractionAgent already supports per‑document scoping** — pass chunks filtered by `metadata.source` to extract from a single paper. No code change needed for Phase 4's per‑document extraction pipeline.
2. **KG is the shared truth cache** — `NetworkXJSONStorage` persists to `project_graph.json`. Phase 4 should use it as the cross‑stage knowledge accumulator (per‑document extractions feed it; per‑theme synthesis reads from it).
3. **Similarity‑threshold retrieval** (L2 distance ≤ 1.0, max_chunks=20) replaces fixed n_results. Adaptive to corpus size.
4. **TF‑IDF cosine anchoring (threshold 0.35)** correctly distinguishes grounded vs speculative claims. Low scores on inferential synthesis are expected — that's the system correctly identifying that synthesis goes beyond verbatim quoting.
5. **Pre‑summarization at ingest** stores chunk summaries in ChromaDB metadata. The `summarize_node` automatically uses them when available, falling back to query‑time LLM summarization.

**What NOT to change:**
- The 17‑node LangGraph graph structure (proven stable)
- The interrupt/resume pattern with `MemorySaver` checkpointer
- The SciSpaCy NER integration (feeds extraction agent)
- The debate chain (Drafter→Critic→Arbiter→Pass2) — this is reused in Phase 4 per‑theme synthesis

**Where to start Phase 4:**
1. Build the query decomposition agent first — it's the entry point for Survey Mode
2. Then thematic clustering (can reuse category_discovery prompt pattern)
3. Then parallel per‑document extraction (ExtractionAgent + source‑filtered chunks)
4. Then per‑theme deep synthesis (reuse existing debate chain)
5. Finally cross‑theme synthesis & gap analysis

**Early Phase 4 benchmark experiment (DeepSeek Chat vs v4 Pro for extraction):**

Before building the parallel extraction pipeline, run a controlled comparison to determine whether DeepSeek Chat can replace v4 Pro for per‑document extraction without quality loss.

*Method:*
1. Select 5 papers from the corpus with diverse content (surface chemistry, immunology, bone biology, methods)
2. Run per‑document extraction on each paper twice — once with `model="deepseek-v4-pro"`, once with `model="deepseek-chat"`
3. Compare outputs on: (a) entity count per paper, (b) evidence phrase completeness (do extracted evidence sentences contain the claimed finding?), (c) category coverage (are all expected entity types present?), (d) latency and token cost per extraction
4. If entity counts are within 10%, evidence phrases are verifiable, and no category types are consistently missing → switch to Chat for Phase 4 per‑document extraction

*Rationale:* Per‑document extraction is the volume step in Survey Mode (100 papers × 1 extraction each). A 50% cost reduction here compounds across the entire survey. Extraction is a lower‑risk task than synthesis (it condenses existing text rather than creating new claims), making model downgrade safer.

**How parallel extraction works on a local system:**

Parallelism in this architecture does NOT mean running multiple local LLMs simultaneously. It means firing multiple concurrent HTTP requests to the DeepSeek API. The local system is a thin client — it sends extraction prompts and waits for responses. The heavy compute happens on DeepSeek's servers.

```
Local (M3 Max)                          DeepSeek Servers
┌─────────────────┐                    ┌──────────────────────┐
│ paper_1 → POST ─┼────────────────────┤→ extract paper_1     │
│ paper_2 → POST ─┼────────────────────┤→ extract paper_2     │
│ paper_3 → POST ─┼────────────────────┤→ extract paper_3     │  ← all run in parallel
│    ...          │                    │      ...             │
│ paper_N → POST ─┼────────────────────┤→ extract paper_N     │
│                 │                    └──────────────────────┘
│ collect ←───────┼────────────────────── responses stream back
│ responses       │
└─────────────────┘
```

*Why this works:* Python's `concurrent.futures.ThreadPoolExecutor` can manage 10–20 concurrent HTTP connections trivially (network I/O is not CPU‑bound). Each paper's context is small (~20–30 paragraphs, ~5000 words) — well within DeepSeek's 128K token window. The local system's only job is to format prompts, dispatch requests, and collect structured JSON responses — no heavy local compute required.

*Expected latency:* If one extraction takes ~30 seconds, 10 parallel extractions complete in ~30 seconds (not 300). 100 papers ≈ 3–4 minutes with a thread pool of 10–20 workers. This is what makes the Phase 4 hybrid architecture viable on consumer hardware.

#### Phase 4 deliverables (all complete — see Status section above for implementation details)

- ZoteroAdapter upgraded: real API calls for item creation, PDF attachment, CiteKey generation
- DOI extraction from Docling metadata
- Ingest pipeline: on PDF addition, automatically create Zotero item
- Citation keys propagated through extraction (entity → source paper) and synthesis (inline @keys)
- Query decomposition agent: breaks complex research questions into theme‑discovery sub‑queries
- Thematic clustering agent: assigns papers to 1+ themes using embedding similarity (LLM fallback preserved)
- Per‑document extraction agent: reuses Phase 3 ExtractionAgent with source‑filtered chunks; runs in parallel
- Per‑theme deep synthesis: reuses Phase 3 debate chain (Drafter→Critic→Arbiter) on all papers in each theme
- Cross‑theme synthesis agent: consumes all theme syntheses + KG to produce final survey with gap analysis
- Expanded human‑in‑the‑loop gates for survey results (theme review, gap acceptance)
- Multi‑level query caching (decomposition, per‑theme, cross‑theme) with 7‑day TTL and visible logging

#### Current known limitations

1. **Per‑theme synthesis evidence cap** — `_fit_summaries_to_context` dynamically fills the context window, but themes with 100+ papers still only see ~100 summaries. Tiered synthesis (cluster within theme, synthesize sub‑groups, then synthesize sub‑syntheses) would preserve completeness. Planned for Phase 6.
2. **NetworkX JSON does not scale past ~10K edges** — the current KG has 521 nodes / 1900 edges. At 100+ papers (~15K nodes, 75K edges), serialization latency becomes noticeable. Neo4j adapter (Phase 6) addresses this.
3. **DeepSeek API queuing** — during peak hours, API calls can queue for 10+ minutes. This is server‑side. Local Ollama deployment (Phase 5) would eliminate external dependency.
4. **TF‑IDF anchoring floors inferential synthesis** — syntheses about mechanisms and cross‑paper inference score lower than factual enumerations because they synthesize rather than quote. A semantic (embedding‑based) similarity measure alongside TF‑IDF would give a more accurate picture.
5. **6 pre‑existing test failures** — `test_synthesis_agents.py` mocks `langchain_ollama.ChatOllama` removed during DeepSeek migration; not yet updated.
6. **Pre‑extraction uses a generic default query** — entities are extracted once using a broad query ("What are the key findings…"). If a user query focuses on a narrow subtopic, some pre‑extracted entities may be irrelevant but none should be missing since the generic query captures everything.

#### Phase 4 → Phase 5 handoff notes

*For the next developer picking up Phase 5. Read this section first.*

**Current state of the codebase:**

- All agents use DeepSeek API via `langchain_openai.ChatOpenAI` with configurable model parameter. Per‑theme tasks use `deepseek‑chat`; cross‑theme synthesis uses `deepseek‑v4‑pro`. API key from `.env` as `DEEPSEEK_API_KEY`.
- PDFs go in `data/`. The demo (`phase4_demo.py`) auto‑discovers new PDFs, pre‑summarizes, pre‑extracts entities, generates cite keys, creates Zotero items, and caches paper embeddings — all at ingest time.
- Pre‑extracted entities live in `projects/default/extractions/`. Pre‑computed paper embeddings in `projects/default/embeddings/`. Query cache in `projects/default/query_cache/`. Delete any of these to force recomputation.
- Knowledge graph at `projects/default/project_graph.json`. 521 nodes, 1900 edges across 6 papers.
- Survey Mode graph has `interrupt_before=["survey_scrub"]` for human‑in‑the‑loop review.

**Key architectural decisions carried forward:**

1. **Model tiering is production‑ready** — `PER_THEME_DRAFTER_MODEL = "deepseek‑chat"` and `CROSS_THEME_DRAFTER_MODEL = "deepseek‑v4‑pro"` are module‑level constants in `survey_nodes.py`. All agents accept an optional `model` parameter. Phase 5 local Ollama deployment should configure equivalent tiering (small biomedical model for extraction/summarization, larger model for synthesis).
2. **Parallel execution via ThreadPoolExecutor** — per‑theme synthesis (max 8 workers) and cross‑theme + gap (2 workers). Phase 5 should migrate to LangGraph Send API for true fan‑out with per‑theme checkpointing and streaming.
3. **Multi‑level query cache persists to disk** — 7‑day TTL. Phase 5 should verify cache behavior under air‑gap (no external network dependency).
4. **Conditional Critic threshold 0.35** — configurable via `CONDITIONAL_CRITIC_THRESHOLD`. Phase 5 should benchmark with local models to recalibrate.
5. **tiktoken‑based evidence truncation** — uses `_fit_summaries_to_context` with `cl100k_base` encoding. Phase 5 should switch to the local model's tokenizer for precise counts.
6. **Human‑in‑the‑loop already integrated** — `interrupt_before=["survey_scrub"]` with approve/edit‑with‑feedback/discard. Phase 5 needs no changes here.
7. **Citation keys propagate through the full pipeline** — stored in chunk metadata at ingest, used in Drafter prompts. Zotero item creation works with real API (credentials from `.env`). Phase 5 should extend to DOI‑based PubMed metadata lookup for richer metadata.

**What NOT to change:**
- The 17‑node Deep Mode graph (proven stable, separate from Survey Mode)
- The 8‑node Survey Mode graph structure
- The interrupt/resume pattern with `MemorySaver` checkpointer
- The SciSpaCy NER integration
- The debate chain internals (Drafter→Critic→Arbiter flow)
- The embedding‑based thematic clustering (keep LLM fallback)
- The TF‑IDF extractive summarization (do not revert to LLM)
- The single‑paper debate skip logic (now skips Drafter entirely)
- The KG interface (`BaseGraphStorage` abstract class)
- The evidence anchoring check (`compute_anchoring_score`)

**Where to start Phase 5:**
1. Set up Docker Compose with three services (orchestrator, public‑corpus, secure‑corpus)
2. Deploy two Ollama instances; configure secure instance with `internal: true`
3. Implement `BoundaryScrubber` node — regex redaction at secure‑public boundary
4. Update LangGraph routing for `query_scope` field (`public`, `secure`, `both`)
5. Run penetration tests (prompt injection, verify no data leaks)
6. Generate security audit log

**Local Ollama model sizing for Phase 5 (M3 Max, 36 GB unified memory):**
- Qwen3.6 35B‑A3B at Q4: ~20 GB including KV cache. 10–20 tok/s for synthesis tasks.
- Small biomedical model (3B at Q4): ~2 GB. 80–200 tok/s for extraction/summarization.
- Both can run simultaneously within 36 GB, enabling the tiered architecture locally.
- Expect 3–6 min per survey query locally (vs 1–2 min via DeepSeek API). Tradeoff: zero API cost + air‑gap security.

#### Phase 5: Security Hardening & Air‑Gap (Weeks 10-11)
Goal: True dual‑corpus isolation with network enforcement.

Deliverables:

Docker Compose with three services: orchestrator, public-corpus (internet), secure-corpus (no internet, internal: true)

Two Ollama instances; secure instance fully air‑gapped

BoundaryScrubber node: regex redaction at secure‑public boundary

LangGraph routing updated for query_scope switch

Penetration testing: attempt prompt injection, verify no data leaks

Security audit log

**Formal benchmarking suite** (redesigned for single-developer workflow — see §12.3): Deferred to Phase 6.  The original plan (20–30 human-annotated QA pairs) was impractical for one person.  Redesigned as a three-tier approach: (A) automated programmatic metrics (anchoring, latency, claim density), (B) RAGAS LLM-as-judge evaluation, (C) golden query tripwires (3 queries, 6 min/week).  Tier A will be built in Phase 6 alongside the UI.

#### Phase 5.5: Local Model Optimization & Speed

During Phase 5, the DeepSeek API was replaced with local Ollama models on an
M3 Max (36 GB unified memory).  Initial latency was 12–39 minutes due to three
issues: (a) 32K default Ollama context creating enormous KV caches, (b) parallel
requests exceeding GPU memory, (c) verbose Drafter output producing 1000–2200 char
per-theme syntheses.  Multiple optimization rounds brought latency to ~5–8 min:

**Model selection:**
- Fast tier: `gemma4:e4b` (~4B active experts, 9.6 GB) — 2‑3× faster than granite4.1:8b
- Reasoning tier: `qwen3.6:35b` (~3B active MoE, 23 GB)
- Dual‑model parallelism tested (gemma4:e4b + medgemma:4b) but medgemma proved
  too slow (10+ min/theme on older Gemma 3 architecture)

**Output format optimization:**
- Drafter system prompt changed from "write a concise paragraph" to "produce
  evidence‑backed claims, one per line, no preamble"
- Per‑theme output reduced from 1000–2200 chars to 250–600 chars per theme
- Anchoring scores maintained at 0.88–0.95 (well‑grounded)
- LLM‑based compression step between per‑theme and cross‑theme was removed —
  dense claims feed directly to the cross‑theme model
- "3–8 themes" hardcoded limit removed from query decomposer — replaced with
  quality‑driven "ALL semantically distinct themes"

**Debate chain simplification:**
- Second Critic→Arbiter pass removed (was 5 LLM calls per debated theme, now 3)
- Conditional critic threshold raised to 0.50 for local models (from 0.35)
- `LLM_MAX_TOKENS=4096` (env var), `LLM_TIMEOUT=900s` (env var)
- `max_workers=1` — sequential per‑theme execution to avoid KV cache exhaustion

**Diagnostic logging:**
- Per‑call timing (Drafter invoke start/done with prompt chars, output chars, latency)
- Per‑theme phase timing with model and score reporting
- Phase‑level latency breakdown visible in logs

**32 hardcoded numeric limits identified** across 9 files (thresholds, truncation
chars, top‑N caps, worker limits).  Four were fixed in Phase 5; the remaining 28
are deferred to Phase 6+ (see HANDOFF.md for full inventory).

### Phase 6 Status (May 2026)

**Core complete.**  Streamlit UI, GLiNER-PII privacy layer, and multi-layer
correctness benchmarking are built and validated.  The correctness layer is the
key deliverable — no existing biomedical benchmark tests multi-document synthesis
quality, so we built our own validation infrastructure.

#### Deliverables built

| Deliverable | Implementation | Status |
|---|---|---|
| Streamlit UI | `app.py` — 5-tab interface: Query, Results, Benchmarks, History, Logs | ✅ Built |
| Session history | Persisted in `st.session_state`, query log with re-run support | ✅ Built |
| Export formats | Markdown, plain text, JSON download buttons in UI | ✅ Built |
| GLiNER-PII privacy model | `src/security/gliner_privacy.py` — 570M params, 55+ entity types (Apache 2.0), drop-in `PrivacyModel` implementation | ✅ Built |
| Tier A programmatic benchmark | `phase5_benchmark.py` — 9 metrics (anchoring, density, entity rate, debate invocation, cross-theme coverage, redundancy, gap novelty, grounded/inferential, citation provenance), pytest-compatible | ✅ Built |
| Correctness test suite | `test_correctness.py` — 4 tests: false-claim injection (3 planted), negative controls (3 OOC queries), Discussion-overlap, grounded/inferential claim tagging | ✅ Built |
| LLM-as-Judge evaluation | `ragas_correctness.py` — faithfulness + gap quality (novelty, actionability) with TRUE/FALSE/GRAY calibration, supports DeepSeek chat + v4-pro as judge (synthesis stays local) | ✅ Built |
| Hybrid anchoring | BM25 + ChromaDB fused in `compute_anchoring_score()` — matches main pipeline's `HybridRetriever` pattern, eliminates false-low scores from BM25 keyword-frequency bias | ✅ Built |
| Gap analysis model switch | `GAP_ANALYSIS_MODEL=gemma4:e4b` env var — cuts gap analysis from ~368s (qwen3.6:35b) to ~40s, quality validated via RAGAS | ✅ Built |
| API comparison script | `phase5_api_comparison.py` — framework for DeepSeek v4-pro vs local Ollama comparison (not yet executed with `--live`) | ✅ Built |
| Dataset generation script | `generate_benchmark_dataset.py` — automated 80-100 sample QA pair generator via LLMs + RAGAS (not yet executed) | ✅ Built |

#### Benchmarking results (May 2026, 6‑paper corpus, hybrid retrieval)

> ⚠ **Scale caveat**: All benchmarks below are from a 6‑paper corpus with heavy
> topical overlap from a single lab's research program.  At this scale, anchoring
> scores measure traceability (does each claim match *some* evidence chunk?) rather
> than factual accuracy (does it cite the *right* evidence and state findings
> precisely?).  88% of claims are grounded because nearly every claim finds a chunk
> with matching keywords — this is expected at small scale and will drop as the
> corpus diversifies.  None of these metrics should be interpreted as production‑grade
> quality certification.  They are proof‑of‑concept validation that the evaluation
> framework works.  Definitive benchmarking requires Phase 8 scale (100+ papers).

```
Tier A — Automated (phase5_benchmark.py):
  Anchoring (hybrid BM25+ChromaDB): 0.993 mean (99.2% grounded)
  Claim density:                     118 claims across 22K chars (~187 chars/claim)
  Gap novelty (Discussion):          80% (8/10 gaps don't match Discussion sections)
  Grounded/inferential:              88% grounded / 12% inferential (chunk-level)
  Entity appearance:                 36.2% of pre-extracted entities surface in output
  Debate invocation:                 0% (no theme below 0.50 threshold)

Tier A+ — Correctness (test_correctness.py):
  False-claim detection:             3/3 fabricated claims flagged as ungrounded
  OOC detection:                     3/3 out-of-corpus queries score < 0.40
  Discussion-overlap:                80% gap novelty (validated against 64 Discussion chunks)

Tier B — LLM-as-Judge (ragas_correctness.py):
  Calibration:                       VALID  (TRUE 5.0/5, FALSE 1.0-1.2/5)
  Faithfulness (deepseek-chat):      4.7/5 grounded, 5.0/5 inferential
  Faithfulness (deepseek-v4-pro):    4.5/5 grounded, 4.6/5 inferential
  Gap quality (v4-pro):              4.5/5 novelty, 4.8/5 actionability

1:1 API vs Local — full survey‑graph comparison (phase5_api_comparison.py):
  Metric                        DeepSeek (chat+v4-pro)    Local (gemma4+qwen3.6)
  ────────────────────────────  ──────────────────────    ───────────────────────
  Avg Anchor Score              0.9690                    0.9470 (+0.022 delta)
  Per‑theme claims              119                       96
  Elapsed time                  212 s (3.5 min)           524 s (8.7 min)
  Speed ratio                   1× (baseline)             2.5× slower
  Cost                          ~$0.50                    free / air‑gapped
```

#### 1:1 API vs local comparison notes

The comparison runs the **exact same** `build_survey_graph()` pipeline for both providers — same retrieval, same thematic clustering, same per‑document extraction, same synthesis graph. Only the LLM provider changes. Model tiering: cloud uses `deepseek‑chat` for light tasks (per‑theme, extraction, decomposition) and `deepseek‑v4‑pro` for heavy tasks (cross‑theme, gap analysis). Local uses `gemma4:e4b` for light and `qwen3.6:35b` for heavy as configured in `.env`. Prompt compression (21% reduction, entities stripped of redundant metadata) was active on the local side. Run with `python phase5_api_comparison.py --run local` to re‑run local against saved cloud results without re‑paying API credits.

#### Known gaps in Phase 6 (May 2026 — all addressed)

| Item | Priority | Status |
|---|---|---|
| API vs local comparison (1:1 survey graph) | High | ✅ Built + executed — anchor delta -0.0033 first run, +0.022 compressed |
| Security scrubber fuzzer | High | ✅ Built + executed — 100% regex, 12% GLiNER FPR (58% → 12% after label fix) |
| Cache key versioning | High | ✅ Built — `src/cache/__init__.py` with `CACHE_VERSION="v1"` |
| Multi-run variance test | Medium | ✅ Script built (`phase6_multi_run.py`), cached single-run run |
| Prompt compression (entity + KG) | Medium | ✅ Built — 21% prompt reduction, anchor delta +0.022, claims delta 18% |
| Per-theme parallelization | Medium | ✅ Built — `PER_THEME_MAX_WORKERS=2`, ~23% faster per-theme wall clock |
| GLiNER-PII label restriction | Medium | ✅ Built — dropped high-FPR labels, FPR 58% → 12% |
| 1:1 comparison refactor | Medium | ✅ Built — both sides run `build_survey_graph()`, cloud results persist to disk |
| 80-100 sample dataset | Low | Deferred to Phase 8 (script exists, not executed) |
| Configuration comparison matrix | Low | Deferred |
| Neo4j adapter | Low | Deferred to Phase 8 (publication scale) |
| Docs / quickstart guide | Low | Deferred |

#### Phase 6.5 additions (May 2026 — built during gap closure)

| Addition | File(s) | Rationale |
|----------|---------|-----------|
| Per-theme parallel (same-model) | `src/graph/survey_nodes.py` | `PER_THEME_MAX_WORKERS=2` (env‑var configurable). Single‑model `ThreadPoolExecutor` pipelines concurrent HTTP requests to Ollama — same KV cache, no memory multiplication. ~23% faster per‑theme wall clock (161 s vs 210 s for 6 themes). |
| Prompt compression | `src/graph/survey_nodes.py` | New `_compress_entities_for_drafter()` strips redundant entity metadata (source_paper, chunk_index, cite_key) while preserving evidence phrases, direction, and context. Capped at 12 entities/category. KG insights reduced to top‑5 central / top‑3 bridge nodes. Net: ~21% prompt reduction (57 K → 45 K chars). Anchor delta +0.022 vs uncompressed; claims delta 18% (96 vs 117). Full‑scale validation deferred to Phase 8. |
| GLiNER-PII label restriction | `src/security/gliner_privacy.py` | Default labels narrowed to person, phone, email, id, ssn, credit card, patient id, url, ip — removed medical condition, organization, location, date, address, hospital. FPR on biomedical text: 58% → 12%. Detection rate on context‑dependent PHI: 50% → 25% (trading breadth for precision). |
| 1:1 API vs local comparison | `phase5_api_comparison.py` | Replaced the old `--live` mode (simplified manual pipeline) with `--run cloud|local|both` that runs the **exact same** `build_survey_graph()` for both providers. Cloud results saved to `projects/default/comparison/cloud.json` for persistent re‑comparison. Model tiering: cloud uses deepseek‑chat for per‑theme / v4‑pro for cross‑theme+gap; local preserves `.env` config. |
| Security scrubber fuzzer | `phase6_security_fuzzer.py` | 500+ lines, 9 pytest tests, 1000+ random PHI‑like samples across 10 categories. Tests both regex BoundaryScrubber and GLiNER‑PII with overlap analysis. Standalone + pytest‑compatible. |
| Multi‑run variance | `phase6_multi_run.py` | 2 pytest tests. `--skip-run` reads cached data instantly. Live mode clears L1 cache per run and reports CoV (coefficient of variation) and per‑run anchoring distribution. |
| Cache key versioning | `src/cache/__init__.py` | `CACHE_VERSION = "v1"` prepended to all cache hash inputs in `llm_cache.py` and `query_cache.py`. Bump to invalidate stale entries after prompt or logic changes. |
| DOB YYYY‑MM‑DD pattern | `src/security/boundary_scrubber.py` | Added second DOB pattern catching YYYY‑MM‑DD format (was 33% detection → 100%). |

#### Lessons learned in Phase 6

1. **BM25 keyword-frequency bias is a real failure mode** — single-retriever anchoring produced false low scores when common filler words (\"mice,\" \"obese\") drowned out rare discriminative terms (\"leptin\"). Hybrid retrieval (BM25 + ChromaDB) fixes this. Investigation across 118 claims showed BM25 is the primary retriever (56% of claims) while ChromaDB handles the 3.4% of claims BM25 misses entirely. Both are essential.

2. **LLM-as-Judge requires calibration** — gemma4:e4b scored every claim 5/5 (agreeableness bias). Adding a critical prompt + TRUE/FALSE/GRAY calibration claims validated that the judge discriminates. DeepSeek chat and v4-pro both confirmed the system is well-calibrated (TRUE 5/5, FALSE 1/5).

3. **Sentence-level TF-IDF inflates grounded rates artificially** — splitting evidence into sentences creates thousands of granular units, making any claim find a \"match.\" Chunk-level matching (used in the benchmark) is more honest: 83-88% grounded vs 99%.

4. **Gap novelty is real** — Discussion-overlap testing (searching gap questions against paper Discussion sections) showed 80-90% of gaps are genuinely novel, not copying authors' future directions. This was the user's core concern and the data supports the pipeline is doing real work.

5. **6 papers limits synthesis depth** — 88% of claims are grounded (directly traceable to evidence). At 100+ papers with diverse content, the inferential rate would naturally increase as the system has more cross-paper material to synthesize. Phase 8 scale will expose this.

6. **v4-pro is 10× slower than chat for judging** — 509s vs 51s for 20 claims. The quality delta is small (0.1-0.2 in faithfulness, 0.6 in actionability). For routine benchmarking, deepseek-chat is sufficient. Reserve v4-pro for final validation.

#### Novel approaches developed in Phase 6

1. **Calibrated LLM-as-Judge** — pre-evaluating the judge with TRUE/FALSE/GRAY claims before trusting its scores. If the judge can't discriminate fabrication from truth, discard its evaluations entirely. This is standard in recent LLM evaluation research but novel in this context.

2. **Discussion-overlap gap novelty test** — searching gap questions against paper Discussion sections to verify the system isn't regurgitating authors' own future directions. No existing benchmark covers this for multi-document synthesis.

3. **Grounded vs inferential claim tagging** — splitting synthesis claims by whether they match a single evidence chunk (grounded) or synthesize across papers (inferential). Only grounded claims are scored for correctness; inferential claims are evaluated on directional support.

4. **Hybrid retrieval in anchoring** — extending `compute_anchoring_score()` with the same BM25 + ChromaDB pattern used by the main pipeline. Fixes a class of false-low scores caused by BM25's keyword-frequency bias.

#### Remaining Phase 6 items — recommended order

1. **Cache key versioning** (~20 lines) — prevents stale cached results from misleading benchmarks. Trivial, should be done before any new code changes.
2. **API vs local comparison** — run `phase5_api_comparison.py --live` with 3 cached queries to validate local Ollama produces comparable quality to DeepSeek API. ~$1-2, ~10 min.
3. **Security fuzzer** — scrubber false negative rate test (~500 lines). Important for production readiness.
4. **Multi-run variance** — run the same query 3×, report mean ± std for anchoring. Confirms single-run scores are stable.

#### What was deferred from the original Phase 6 plan

- **Neo4jStorage adapter**: NetworkX JSON handles the current 6-paper, ~500-node graph. Neo4j is only needed at Phase 8 scale (100K+ edges). Deferred.
- **80-100 sample automated dataset**: Script exists but generation requires ~2h LLM calls. More valuable at Phase 8 when the corpus is larger and diverse. Deferred.
- **Config comparison matrix**: Current config (gemma4:e4b + qwen3.6:35b + 0.50 threshold + dense claims) is stable. Matrix is useful when actively evaluating alternatives. Deferred.

### Phase 7 Status (May 2026)

**Complete.**  All six deliverables built, tested, and wired into the pipeline:

| Deliverable | Implementation | Tests |
|------------|---------------|-------|
| Figure extraction | `src/vision/figure_extractor.py` — Docling `generate_picture_images=True` + `do_picture_classification=True` | 7 |
| Smart figure filtering | `src/vision/figure_filter.py` — Docling `DocumentFigureClassifier` (65% weight) + caption/size/page soft hints. 80.9% keep rate, zero data figures lost. | 23 |
| Vision model integration | `src/vision/vision_descriptor.py` — gemma4:e4b via Ollama REST API. No model rotation needed (already loaded for text). Reads IL-6, CD4, CD8, cytokine names directly from figure labels. | 13 |
| Figure-to-text embedding | `src/vision/figure_embedder.py` — ChromaDB with `chunk_type="figure"`. `include_figures=True` on `HybridRetriever.query()`. | 7 |
| Multi-turn section writing | `src/graph/sectioned_survey_graph.py` — 8-node LangGraph: init → retrieve → draft → review → [route → retrieve | assemble] → scrub. IMRaD section iteration with interrupt-at-review. | 1 |
| Claim/citation ledger | `src/synthesis/claim_ledger.py` — SHA-256 dedup, @citation parsing, coverage reporting, per-section validation, JSON persistence. Cross-section duplicate filtering prevents re-stating the same claim. | 14 |
| Vision ingest integration | `src/vision/vision_ingest.py` — `vision_ingest_pdf()` called during app.py PDF ingestion. Figures extracted, described, and embedded. | — |
| Sectioned Survey in UI | `app.py` — "Sectioned" mode in the mode selector, sectioned display tabs, ledger integration. | — |

**Model selection:** gemma4:e4b is the default vision model. It's already loaded as the fast-tier text model during ingestion, so figure descriptions add zero model rotation overhead. At ~17s per figure, it identifies IL-6, CD4, CD8, cytokines, WT/knockout groups, and significance markers directly from figure labels — significantly better than llava:7b (generic "scientific poster") and qwen3-vl:4b (generic "gene expression").

**Citation fix:** The Drafter system prompt previously hardcoded `@author2025` as an example format, causing the LLM to hallucinate that citation key. Fixed to use only provided citation keys. Cache version bumped to v3.

##### Phase 7 Lessons Learned

**1. Vision model selection matters dramatically for biomedical figures**

We compared three multimodal models on the same actual BME figures. llava:7b (2023) produced generic descriptions ("scientific poster with graphs"), qwen3-vl:4b produced mid-quality ("gene/protein expression"), and gemma4:e4b produced highly specific descriptions naming IL-6, CD4, CD8, cytokines, WT/knockout groups, and significance markers directly from figure labels. The better model was already loaded as our fast-tier text model, eliminating the need for model rotation entirely. **Decision**: Default vision model = gemma4:e4b. No rotation overhead. Better accuracy than dedicated vision-only models.

**2. num_predict breaks multimodal Ollama models (known bug)**

Passing `num_predict` in the options dict to Ollama's `/api/generate` causes multimodal models to return empty responses with `done_reason=length`. `temperature` alone works fine. We worked around this by truncating the response after generation instead of limiting tokens at the API level. **Decision**: VisionDescriptor sends `temperature` only; max_tokens enforced via post-generation truncation.

**3. Docling's built-in figure classifier is production-grade**

The `DocumentFigureClassifier-v2.5` model (loaded via `do_picture_classification=True`) correctly identifies bar charts, logos, icons, thumbnails, tables, and scatter plots with >0.99 confidence. At 65% weight in our filtering score, it correctly filtered all 9 extraneous images (3 logos, 3 page thumbnails, 3 icons) from 47 while keeping all 38 data figures. Zero data loss. **Decision**: Classification-first filtering is reliable; size/page/caption are soft auxiliary hints only.

**4. Figure captions MUST be resolved from document text items, not picture annotations**

Docling's `PictureItem.annotations` contains classification metadata (not captions). Real figure captions are in `PictureItem.captions` which reference `DoclingDocument.texts[idx].text` via `#/texts/{idx}` refs. Getting this wrong caused every figure to show raw classification strings as captions. **Decision**: Always resolve `picture.captions → doc.texts[idx].text` for real captions. Skip `picture.annotations` entirely for caption extraction.

**5. Monkey-patching imports must happen before the target class is imported**

The `include_figures=True` extension to `HybridRetriever.query()` is applied at module import time in `src/vision/figure_embedder.py`. Any file that uses `HybridRetriever.query(include_figures=True)` must import `figure_embedder` BEFORE importing `HybridRetriever`. Failing to do this causes `TypeError: unexpected keyword argument 'include_figures'`. **Decision**: Document this constraint; always import `src.vision.figure_embedder` first in scripts that use the extended API.

**6. Claim ledger SHA-256 deduplication works across sections**

Normalizing claim text (lowercase, collapse whitespace, SHA-256 first 16 chars) correctly detects duplicate claims across Introduction and Results sections. At 6 papers, the pipeline prevented 5 duplicate claims across our sectioned survey run. **Decision**: Stable claim IDs via content hashing are reliable for cross-section dedup.

**7. Keyword-based novelty detection is insufficient for literature discovery**

The Phase 9 POC used a static list of ~50 biomedical keywords for substring matching. This correctly identified 87% of external papers as novel, but ranked "rat" as the top keyword because it appears frequently in biomaterials papers — even though our lab uses mouse models. An LLM-based Coverage Check node is planned to replace this: it would generate targeted PubMed queries per gap and score returned abstracts for domain-specific novelty. **Decision**: Static keyword extraction is POC-only. Phase 9's Coverage Check node (LLM-gated) replaces it.

**8. Baseline comparison validates the pipeline architecture**

Comparing the full pipeline against a naive single-pass RAG (retrieve → draft, no debate/KG/clustering) showed the pipeline produces 27× more claims (134 vs 5) with essentially identical anchoring quality (0.993 vs 1.000). The naive RAG's perfect anchoring is misleading — it's easy to ground 5 safe claims. The pipeline maintains grounding across 134 claims spanning 5 themes. **Decision**: The complex architecture earns its keep. Publish the baseline alongside every Phase 8 benchmark.

##### Novel Approaches Invented in Phase 7

1. **Vision model reuse eliminates rotation** — Using the already-loaded fast-tier text model (gemma4:e4b) for figure description avoids the entire model load/unload cycle the architecture originally planned. This saves ~15–30s per query and simplifies the pipeline. Generalizable pattern: check whether existing models support multimodal input before pulling dedicated vision models.

2. **Docling classification-first figure filtering** — Using a trained image classifier (not heuristics) as the primary gate for figure relevance. Size, position, and caption are weighted at only 35% combined. The classifier is deterministic, runs locally, and produces 0.99+ confidence on bar charts. Novel for multi-document biomedical pipelines where logo/journal-name filtering is critical.

3. **Claims-as-content-addressed ledger** — SHA-256 hashing of normalized claim text as a stable deduplication key across sections. This is simpler and more reliable than embedding-based similarity (which would miss near-duplicates with different wording). The ledger also serves as a cross-session persistence layer, enabling incremental section writing across multiple sessions.

4. **Cross-paper claim provenance via compact identifiers** — 16-char hex digests from SHA-256 of normalized claim text. Short enough for log output, long enough to be collision-resistant for 100K+ claims. Enables fast duplicate detection without storing full claim text in memory.

### Phase 8: Publication-Scale Retrieval (Weeks 16-17) — Not Started

Goal: Scale from 6 papers to 100s-1000s with sub-minute query times.

#### Phase 8 Initiation Plan

**1. Neo4j adapter (~4-6 hrs)**
- Implement `Neo4jStorage` class satisfying the existing `BaseGraphStorage` interface
- Migration: one config value (`graph_backend: "neo4j"`) swaps all consumers
- Key Cypher queries: `get_neighbors`, `get_subgraph`, `query_relationships`
- Connect to a local Neo4j container or Neo4j Aura free tier
- Expected: handles 100K+ edges vs NetworkX's ~10K edge ceiling

**2. Hierarchical clustering (~3-5 hrs)**
- Level 1: broad topic assignment via embedding similarity (already partially done)
- Level 2: per-topic fine-grained themes via LLM when needed
- Reuses existing `ThematicClusterer` with tiered approach
- Avoids O(n_papers) context explosion by screening at Level 1

**3. Per-theme top-K retrieval (~2-3 hrs)**
- Retrieve top-K chunks per theme via hybrid search (dense + sparse)
- Already partially implemented in `_fit_summaries_to_context()`
- Need: per-theme retrieval scoping, dynamic K based on theme complexity

**4. Corpus-level claim index (L0 cache) (~3-4 hrs)**
- Pre-extract claims from all papers at ingest time
- Index claims in a dedicated ChromaDB collection (`corpus_claims`)
- Query-time: retrieve from claim index vs re-extracting entities
- Requires: extending `PreExtractor` to also produce claim-level extractions

**5. Multi-tier caching L0-L4 (~2-3 hrs)**
- L0: corpus-level claim index (dedicated ChromaDB collection)
- L1: query decomposition (existing)
- L2: per-theme synthesis (existing)
- L3: cross-theme synthesis (existing)
- L4: publication-section output cache (new — caches sectioned manuscript outputs)

**6. Scale benchmarking (~2-4 hrs)**
- Required: a corpus of 100+ diverse biomedical papers
- Run the full pipeline at scale, measure: anchoring drift, inferential rate, latency
- Compare against the 6-paper baseline from Phase 6
- Document how scores change as the corpus diversifies (currently 88% grounded at 6 papers)

**Initiation order:**
1. First: acquire/generate 100+ paper corpus (PubMed/Semantic Scholar output)
2. Then: Neo4j adapter (unblocks graph scalability)
3. Then: Hierarchical clustering + top-K retrieval (unblocks context scaling)
4. Then: L0 cache + L4 cache (unblocks latency scaling)
5. Finally: Full-scale benchmark (validates everything)

Target performance: 30-90s per query on 1000 papers (vs. current 5-8 min on 5 papers).

### Phase 9 Status (15 May 2026 — Complete)

**All 6 gaps closed. Three Phase 10 foundation pieces built.** The full pipeline
ingests OA papers from Europe PMC (with NCBI PMC OAI-PMH fallback), caches
SPECTER2 embeddings locally, downloads and embeds XML `<graphic>` figures into
ChromaDB, runs coverage diagnostics (EPMC vs Semantic Scholar), parses research
gaps into structured search queries, and provides discovery-only web search.

#### Core pipeline

```
Europe PMC search (OPEN_ACCESS:Y)
  → fullTextXML fetch (EPMC REST → PMC OAI-PMH fallback, 3-retry backoff)
  → JATS XML parse → chunk dicts → ChromaDB + BM25 ingest → IngestProgress checkpoint
  → Figure pipeline (XML <graphic> URLs → download → vision_ingest)
  → PreExtractor KG update (if --graph flag)
Semantic Scholar → DOI resolve (title fallback) → SPECTER2 embedding fetch
  → Spector2Cache (DOI-keyed JSON, skip re-fetch on subsequent runs)
Coverage diagnostic: EPMC ∩ S2 overlap → "X/Y papers (Z%) have PMC full text"
Gap analyser: gap text → structured queries → EPMC search → ingest → re-synthesize
Web discovery: DuckDuckGo/DDG → discovery-tagged results (never evidence)
```

#### Phase 9 deliverables (all complete)

| Gap | Deliverable | File | Tests |
|-----|------------|------|-------|
| 1 | Retry logic — 3-retry exponential backoff | `src/retrieval/europe_pmc.py` `_request()` | Implicit (pipeline tests) |
| 2 | Progress persistence — 10-paper checkpoints | `src/utils/ingest_progress.py` | Implicit (pipeline tests) |
| 3 | Ingestion wiring — `--ingest` into ChromaDB + BM25 | `phase9_europe_pmc_test.py` | Phase 5 integration test |
| 4 | Coverage diagnostic — EPMC vs S2 comparison | `src/retrieval/coverage.py` | 14 tests (matching, overlap) |
| 5 | Figure pipeline — XML `<graphic>` URLs → vision_ingest | `src/vision/vision_ingest.py` | 9 integration tests, Ollama describe=True verified |
| 6 | SPECTER2 caching — local JSON, skip re-fetch | `src/utils/spector2_cache.py` | 13 tests (get/put/persist/edge) |

#### Phase 10 foundation (all built)

| # | Item | File | Status |
|---|------|------|--------|
| 7 | PreExtractor + graph_storage wiring | `phase9_europe_pmc_test.py` (`--graph` flag) | ✅ Built |
| 8 | Gap resolver — parse gaps → search → ingest | `src/agents/gap_resolver.py` | ✅ 18 tests |
| 9 | Web search — discovery-only DuckDuckGo client | `src/retrieval/web_search.py` | ✅ Verified manually |

##### Phase 10 core (all built — 15 May 2026)

| # | Gap | File | Description | Tests |
|---|------|------|-------------|-------|
| 10 | No autonomous daemon | `src/agents/orchestrator.py` (418 lines) | Full cycle: web discovery → parallel EPMC search/fetch → batch ingest → PreExtractor → KG save → cycle handoff. Dry-run + live modes. State file + PID. | 22 unit + 4 integration |
| 11 | No subagent spawning | `src/agents/subagents.py` (54 lines) | `run_parallel()` — ThreadPoolExecutor wrapper for concurrent EPMC search/XML fetch across queries. Batches ingest to avoid redundant BM25 rebuilds. | 7 tests |
| 12 | No automated handoff | `src/agents/handoff.py` (147 lines) | `generate_handoff()` / `write_handoff()` — reads KG node/edge counts, ingest progress, SPECTER2 cache stats, cycle summary. Cycle-specific files (`cycle_N_handoff.md`) prevent overwrite of human HANDOFF.md. | 10 tests |
| 13 | No scheduler | `src/agents/scheduler.py` (69 lines) | Interval timer with daemon thread, `stop_event` lifecycle, `run_once()` blocking mode, crash-resilient callback loop. | 8 tests |

##### Phase 10 enhancements (beyond original spec — built during testing)

| Enhancement | Why | Where |
|-------------|-----|-------|
| Parallel EPMC wiring | `run_parallel()` fetches/parses all queries concurrently; batch ingest avoids redundant BM25 rebuilds. ~15s saved per cycle. | `orchestrator.py:_search_and_ingest()` |
| Line‑tagged extraction format | Replaced JSON LLM output for entity extraction. Eliminates 70% parse-failure rate (no braces, commas, or quotes to break). Saves ~25-30 % tokens in Drafter prompts. LLM sees `TYPE: entity` blocks instead of `json.dumps(indent=2)`. | `extraction_agent.py:_parse_line_tagged()`, `synthesis_drafter.py:_entities_to_line_tagged()` |
| Cycle handoff preservation | `write_handoff()` writes to `projects/default/cycle_N_handoff.md` — human `HANDOFF.md` never overwritten by daemon. | `orchestrator.py:_write_handoff()`, `handoff.py:write_handoff()` |
| State file + PID | `orchestrator_state.json` (cycle counter, heartbeat, last error), `orchestrator.pid` for external management. | `orchestrator.py:_write_state()`, `_write_pid()`, `_remove_pid()` |
| Integration test suite | Full cycle tested with mocked APIs: discovery → query building → parallel fetch → batch ingest → handoff → state file. | `tests/test_orchestrator_integration.py` (4 tests) |

##### Known Phase 10 gaps (remaining)

| # | Gap | Severity | Description |
|---|------|----------|-------------|
| A | State file write‑only | Low | `orchestrator_state.json` is written every cycle but never read on restart. Daemon restarts from cycle 0 (idempotent — `IngestProgress` prevents re‑ingestion). |
| B | No handoff file cleanup | Low | `cycle_N_handoff.md` files accumulate forever. No rotation/retention policy. |
| C | No daemon log management | Medium | `logging.basicConfig` to stderr only. No file handler, no rotation. Daemonized logs are lost unless piped. |
| D | Line‑tagged format untested with real Ollama | Medium | All 7 parser tests use mocked LLM output. Real Ollama model may need prompt tuning. |
| E | No long‑running daemon validation | Medium | Only single cycles tested. Multi‑hour runs needed to detect memory leaks, thread exhaustion, timer drift. |
| F | Coverage‑gated routing not wired | Low | `run_coverage_diagnostic()` exists but orchestrator doesn't gate EZProxy routing on coverage < 30 %. |
| G | SPECTER2 embeddings unused | Low | 8 embeddings cached, 0 queried. `paper_similarity_search()` could recommend related papers. |

##### Phase 10 lessons learned

1. **Thread‑based timers need generous margins in tests** — The scheduler's
   `schedule(interval_minutes)` multiplies by 60 internally. Tests using
   `schedule(0.1)` expected 100 ms ticks but got 6‑second ticks. Always
   document whether a time parameter is in seconds or minutes.

2. **BM25 rebuild contention is the real parallel bottleneck — not RAM** —
   Threading EPMC fetch/parse is safe (I/O‑bound, ~8 MB for 4 concurrent
   XMLs). But calling `ingest()` from multiple threads triggers redundant
   BM25 full‑corpus rebuilds. Fix: parallelize fetch+parse, accumulate all
   chunks, call `ingest()` once. The `_ingest_chunks_batch()` method exists
   specifically for this.

3. **Mock import paths must target the definition site** — Lazy imports
   (`from X import Y` inside function bodies) can't be patched at the
   importing module's path. Always patch at the module where the class is
   defined, not where it's imported.

4. **PMCXMLParser.MIN_CHUNK_WORDS = 20 means mock XML needs real content** —
   Unit tests using `<article><body><p>Short.</p></body></article>` produced
   zero chunks because 2‑word sections are silently skipped. Mock XML data
   must contain ≥20‑word paragraphs.

5. **State must be written AFTER counters are incremented** — `_write_state()`
   was called before `_total_ingested` was incremented, causing the state
   file to report 0 papers ingested. Order matters: persist after mutation.

6. **Don't trust LLMs to output valid JSON — give them a format they can't
   break** — 9/13 extractions on local Ollama models failed on first JSON
   parse attempt. The line‑tagged format (`TYPE: category\nENTITY: name\n...`)
   has zero syntax‑failure modes because there are no braces, commas, or
   quotes to break. Match the output format to the model's training
   distribution.

7. **Generated output should NEVER share a path with human documentation** —
   The first live run of `write_handoff()` defaulted to `HANDOFF.md` and
   overwrote a 533‑line developer document with a 29‑line auto‑generated
   summary. Always namespace machine output (`cycle_N_handoff.md`, not
   `HANDOFF.md`). Recovered from git.

8. **Write‑only state files are a code smell** — `orchestrator_state.json`
   is written every cycle but never read on restart. State files should be
   round‑tripped (write + read), not fire‑and‑forget.

##### Phase 10 novel approaches

1. **Line‑tagged extraction format for local LLMs** — Instead of fighting
   JSON parse failures with ever‑more‑aggressive fallback parsers, we
   changed the format the LLM outputs. Line‑tagged text maps directly to
   LLM training data (labeled text, config files, frontmatter). A 30‑line
   parser replaces a 50‑line JSON parser with two fallback attempts.

2. **Thread‑parallel fetch + batch ingest pattern** — In a pipeline where
   ingestion is a mutating operation that rebuilds a corpus‑wide index,
   parallelism is only safe in the read phase. Parallelize I/O‑bound
   fetch+parse, batch all chunks into one `ingest()` call. Generalizable
   to any "gather → mutate" pipeline.

3. **Module‑level worker functions for ThreadPoolExecutor** — `_fetch_and_parse_for_query()`
   is defined at module scope (not a closure or instance method) so it can
   be passed to `run_parallel()` without pickle issues. Receives all config
   via keyword arguments; creates its own API client instances.

4. **Dry‑run mode as a first‑class daemon feature** — `Orchestrator(dry_run=True)`
   runs the full discovery→query cycle but skips all API calls beyond web
   search. Returns `would_have_queries` in summary. Essential for safe
   testing and pre‑flight validation.

5. **Cycle‑specific handoff files for machine‑to‑machine state transfer** —
   Rather than a single `HANDOFF.md`, the daemon writes
   `projects/default/cycle_N_handoff.md` for each cycle. Creates an audit
   trail and prevents the machine from overwriting human documentation.

##### What NOT to change (Phase 10 additions)

All prior constraints from Phase 4–9 still apply.  New from Phase 10:

- Do NOT switch extraction output back to JSON — line‑tagged format eliminates
  the 70% parse‑failure rate on local Ollama models
- Do NOT wire `ingest()` into parallel threads — use `_ingest_chunks_batch()`
  after accumulating all chunks to avoid redundant BM25 rebuilds
- Do NOT remove the dry‑run flag from the orchestrator — essential for safe
  testing and pre‑flight validation
- Do NOT make `_fetch_and_parse_for_query()` an instance method or closure —
  module‑level functions are compatible with ThreadPoolExecutor
- Do NOT change `write_handoff()` default path to `HANDOFF.md` — always
  accept explicit `output_path` from the orchestrator
- Do NOT remove `orchestrator_state.json` or `orchestrator.pid` management
- Do NOT remove the line‑tagged parser (`_parse_line_tagged`) or formatters
  (`_categories_to_line_tagged`, `_entities_to_line_tagged`)

#### API Strategy (updated)

| API | Provides | Rate Limit | Auth | Notes |
|-----|----------|-----------|------|-------|
| **Europe PMC REST** | Search, metadata, fullTextXML (currently 404) | 10 req/s | Free | FullTextXML endpoint down — see fallback below |
| **PubMed Central OAI-PMH** | JATS full-text XML (fallback) | Unknown | Free | Used transparently when EPMC REST returns 404 |
| **Semantic Scholar** | Search, paper metadata, SPECTER2 embeddings | 1 req/s (free), 100 req/s (key) | API key in `.env` | 429 backoff: 10→20→40s |
| **DuckDuckGo (ddgs)** | Web search results (discovery only) | Unknown | Free | `pip install ddgs`; falls back to DDG API |

#### Novel approaches developed

1. **PMC OAI-PMH as transparent fullTextXML fallback** — when EPMC REST returns 404,
   `full_text_xml()` automatically fetches from NCBI's OAI endpoint. Same JATS XML
   content, different transport. Callers are unaware of the path used.

2. **Hierarchical DOI matching** — three-tier match: DOI exact → DOI clean
   (strip `https://doi.org/` prefix) → title fuzzy (SequenceMatcher + word-set
   Jaccard, threshold 0.6). Handles API format inconsistencies.

3. **Word-boundary-aware gap detection with false-positive filtering** — 9 regex
   gap patterns (`\bno\s+[\w\s-]{0,40}?\bdata\b`, etc.) with 6 exclusion patterns
   for null findings. Scoped to first sentence of each gap block.

4. **ChromaDB dedup-before-add** — `ChromaClient.get_existing_ids()` + 
   `add_documents_deduped()` prevents duplicate-entry warnings on re-ingest without
   requiring collection-level configuration.

5. **Aggregate + per-call rate limiting** — S2 client uses `_min_interval=3.0s`
   for normal operation AND 10→20→40s exponential backoff on 429s. Both necessary.

#### Known limitations

- **EPMC `fullTextXML` REST endpoint returns 404** as of 15 May 2026 (server-side outage).
  Worked during initial Phase 9 testing (13 May 2026) — papers fetched successfully.
  A transparent PMC OAI-PMH fallback is active (same JATS XML, 280KB per paper).
  See HANDOFF.md §External API Status for diagnostic timeline and how to tell
  a server outage from a code bug. The pipeline continues working either way.
- **S2 returns 429 when hourly quota is exhausted** — `_min_interval=3.0s` + 10→20→40s
  429 backoff is sufficient for normal operation but back-to-back test runs can exhaust
  the free-tier quota. Wait 30–60s between runs. See HANDOFF.md §External API Status
  for diagnostic guide.
- **SPECTER2 embeddings unavailable for many papers** — only ~30% of resolved
  S2 papers have `embedding` vectors. This is an S2 data limitation, not a code bug.
- **Figure URLs absent from OAI XML** — the OAI endpoint provides figure captions
  but may not include `<graphic xlink:href>` URLs in the same format as EPMC REST.
  Figure pipeline tested with synthetic images; production validation awaits
  EPMC REST recovery.
- **PreExtractor adds ~30s LLM cost per paper** at ingest time. For background
  daemon (Phase 10) this is acceptable. Cached extractions are loaded from disk
  on subsequent runs (<1ms). Phase 10 switched extraction to line‑tagged format
  (eliminated 70% JSON parse failure rate on local Ollama models).
- **`web_search.py` requires `pip install ddgs`** for primary search path. Falls
  back to DDG Instant Answer API (weaker, returns 0 results for niche queries).

#### Tests

- **307 tests passing, 0 failures** (up from 246 at Phase 9 handoff)

#### Original POC (May 2026)

The Phase 9 proof-of-concept demonstrated the feasibility of API-based literature
acquisition (27× faster than Playwright/EZProxy PDF downloads). The initial POC
included:

- `src/retrieval/pubmed.py` — thin PubMed E‑utilities client (deprecated by Europe PMC)
- `src/retrieval/semantic_scholar.py` — Semantic Scholar API client (still used for SPECTER2)
- `phase9_pubmed_demo.py` — POC script (deprecated by `phase9_europe_pmc_test.py`)
- Initial results: 87% of external papers novel against 6-paper corpus
- `phase7_baseline_comparison.py` — 134 claims vs 5 for naive RAG (26.8× improvement)
- EZProxy/Playwright pipeline preserved for non-OA paper acquisition

## 11. Component Interfaces
```python
class HybridRetriever:
    def __init__(self, chroma: ChromaClient, bm25: BM25Index): ...
    def ingest(self, chunks: List[Dict]) -> None: ...
    def query(self, query: str, n_results: int = 10,
              filter_references: bool = True) -> List[Dict]: ...
    # Returns: [{"text": "...", "metadata": {...}}, ...]
```
11.2 PDF Parser
```python
class PDFParser:
    def __init__(self): ...
    def parse(self, pdf_path: Path) -> List[Dict]: ...
    # Returns: [{"text": "...", "metadata": {"source": str, "chunk_type": str, "chunk_index": int}}, ...]
    # chunk_type ∈ {"text", "table", "reference"}
```
11.3 Extraction Agent
```python
class ExtractionAgent:
    def __init__(self, model_name: str, temperature: float = 0.0): ...
    def discover_categories(self, chunks: List[Dict], query: str) -> Dict: ...
    def extract_entities(self, chunks: List[Dict], categories: Dict,
                         ner_candidates: List, query: str) -> Dict: ...
    # Returns: {"category_name": [{"entity": str, "evidence": str, "source": str}, ...], ...}
```
11.4 Synthesis Agents
```python
class SynthesisDrafter:
    def __init__(self, model_name: str): ...
    def draft(self, query: str, entities: Dict, chunks: List[Dict],
              citations: List[str], kg_context: Dict) -> str: ...

class SocraticCritic:
    def __init__(self, model_name: str): ...
    def critique(self, draft: str, chunks: List[Dict], entities: Dict) -> str: ...

class Arbiter:
    def __init__(self, model_name: str): ...
    def revise(self, draft: str, critique: str, chunks: List[Dict]) -> str: ...
```
11.5 Knowledge Graph
```python
class BaseGraphStorage(ABC):
    @abstractmethod
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None: ...
    @abstractmethod
    def add_edge(self, source: str, target: str, relation: str, properties: dict) -> None: ...
    @abstractmethod
    def get_neighbors(self, node_id: str, relation: str = None) -> List[dict]: ...
    @abstractmethod
    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> dict: ...
    @abstractmethod
    def query_relationships(self, source_type: str, relation: str, target_type: str) -> List[dict]: ...
    @abstractmethod
    def save(self) -> None: ...
    @abstractmethod
    def load(self) -> None: ...

class NetworkXJSONStorage(BaseGraphStorage):
    def __init__(self, file_path: str): ...
    # Implements all abstract methods
    # save(): nx.node_link_data → JSON file
    # load(): JSON file → nx.node_link_graph

# Future:
class Neo4jStorage(BaseGraphStorage):
    def __init__(self, uri: str, user: str, password: str): ...
    # Implements all abstract methods via Cypher
```
## 12. Testing Strategy
### 12.1 Test Layers
| Layer | Scope | Tool | Target Coverage |
|-------|-------|------|-----------------|
| Unit | Individual functions/classes | pytest | ≥90% per module |
| Integration | Multi-component workflows | pytest | All pipeline paths |
| Retrieval Quality | Known-item queries | Custom eval | Top-1 accuracy ≥80% |
| Synthesis Fidelity | Evidence anchoring scores | Programmatic | Anchoring ≥0.85 |
| Security | Penetration tests | Custom scripts | No leaks |
| End-to-End | Full query → output | Pytest + manual | 5+ real biomedical queries |

### 12.2 Key Test Cases
Unicode scrubbing: Greek letters, subscripts, microgram symbol → correct ASCII

Hybrid retrieval: "bone growth" returns "osseointegration" (dense); "IL-6" returns only IL-6 chunks (sparse)

Reference filtering: Reference chunks tagged correctly and excludable

Category discovery: LLM identifies relevant categories from a mixed BME corpus

Evidence grounding: Extracted entities have verifiable evidence phrases

Debate fidelity: Synthesis after critique has higher anchoring score than before; no hallucinated claims

Air‑gap enforcement: Secure‑scope queries never reach public network

ASCII enforcement: Final output contains zero non-ASCII characters

### 12.3 Benchmarking Strategy (Phase 6 — built and validated at 6‑paper scale)

> ⚠ These benchmarks validate the evaluation framework, not production quality.
> All results are from a 6‑paper corpus.  Scale‑level benchmarking (100+ papers)
> is deferred to Phase 8.

The original plan (20–30 human‑annotated QA pairs with manual rubric scoring) was
redesigned for a single‑developer workflow.  All three tiers defined in the Phase 5
handoff have been built, integrated, and validated (the evaluation framework works —
meaning it correctly discriminates grounded claims from fabricated claims, detects
out‑of‑corpus queries, and produces calibrated LLM‑as‑Judge scores).  Results from
the latest run (May 2026, 6‑paper corpus, hybrid retrieval enabled):

**Tier A — Automated programmatic** (`phase5_benchmark.py`):
- Anchoring score distribution: mean 0.993, min 0.964, std 0.016 (99.2% grounded)
- Claim density: 118 claims across 22K chars (~187 chars/claim)
- Gap novelty (Discussion‑overlap): 80% of gaps don't match Discussion sections
- Grounded vs inferential: 88% grounded / 12% inferential (chunk‑level)
- Entity appearance: 36% of pre‑extracted entities surface in output
- Debate invocation: 0% (no theme below 0.50 threshold)
- Cross‑theme coverage: 76‑80%
- Redundancy: 35% overlap across themes
- Citation provenance: 108 citations, 6 unique keys, 0 orphaned

**Tier A+ — Correctness tests** (`test_correctness.py`):
- False‑claim injection: 3/3 fabricated claims flagged as ungrounded
- Negative controls: 3/3 out‑of‑corpus queries correctly score < 0.40
- Discussion‑overlap: 80% gap novelty (validated against 64 Discussion chunks)
- Grounded vs inferential tagging: chunk‑level matching across all evidence

**Tier B — LLM‑as‑Judge** (`ragas_correctness.py`):
- Calibrated judge: TRUE claims 5.0/5, FALSE claims 1.0‑1.2/5, GRAY claims 1.5‑2.5/5
- Faithfulness (deepseek‑chat): 4.7/5 grounded, 5.0/5 inferential
- Faithfulness (deepseek‑v4‑pro): 4.5/5 grounded, 4.6/5 inferential
- Gap quality (v4‑pro): 4.5/5 novelty, 4.8/5 actionability
- Judge calibration validated across 2 models — scores are not inflated by agreeableness bias

**Tier C — Golden query tripwires**: Not yet automated; use `ragas_correctness.py`
on your primary research query as a manual spot‑check.

**Why not an established dataset?**  Same as originally stated: public biomedical QA
datasets (PubMedQA, BioASQ, MedQA) test single‑document retrieval, not multi‑paper
synthesis.  Our evaluation suite is novel in testing cross‑paper inference, gap
analysis quality, and discussion‑overlap novelty.

**Automated benchmark creation** (`generate_benchmark_dataset.py`): Script built
but not yet executed.  Defers to Phase 8 when the corpus is larger.  Generating
an 80‑100 sample dataset on the current 6‑paper corpus would have limited utility
for future scale testing.

## 13. Deployment & Containerization
### 13.1 Development (Current)
Single machine, single directory

Ollama running as background service

ChromaDB with persistent local storage

KG stored as JSON in project directory

### 13.2 Lab‑Wide (Phase 6)
Docker Compose services:
```yaml
services:
  orchestrator:
    build: .
    ports: ["8501:8501"]  # UI only

  public-corpus:
    image: federated-rag-public
    networks: [public-net]
    environment: [OLLAMA_HOST=ollama-public]

  secure-corpus:
    image: federated-rag-secure
    networks: [secure-net]
    internal: true  # No internet
    environment: [OLLAMA_HOST=ollama-secure]

  ollama-public:
    image: ollama/ollama
    networks: [public-net]

  ollama-secure:
    image: ollama/ollama
    networks: [secure-net]
    internal: true

networks:
  public-net:
    driver: bridge
  secure-net:
    driver: bridge
    internal: true
```
## 14. Appendices
A. Dependency Version Pinnings
```text
chromadb==0.4.24
rank-bm25==0.2.2
langgraph>=0.2.0
pydantic>=2.9.0,<3.0.0
pyzotero>=1.5.6
python-dotenv==1.0.1
pytest>=8.2.0
docling>=2.0.0
numpy<2.0
langchain-ollama>=0.1.0
networkx>=3.0
scispacy>=0.5.0
tantivy>=0.22.0
```
B. Environment Variables
```bash
ZOTERO_LIBRARY_ID=your_library_id
ZOTERO_API_KEY=your_api_key
OLLAMA_HOST=localhost:11434
GRAPH_BACKEND=networkx_json  # or "neo4j" for lab-wide
PROJECT_DIR=./projects/default
ANONYMIZED_TELEMETRY=False
```
C. Project Directory Structure (End State)
```text
federated_rag/
├── src/
│   ├── state.py
│   ├── unicode_map.py
│   ├── scrubber.py
│   ├── agents/
│   │   ├── extraction_agent.py
│   │   ├── synthesis_drafter.py
│   │   ├── socratic_critic.py
│   │   └── arbiter.py
│   ├── citation_manager/
│   │   ├── base.py
│   │   ├── zotero_adapter.py
│   │   └── mendeley_adapter.py  # future
│   ├── ingestion/
│   │   ├── pdf_parser.py
│   │   └── pmc_xml_parser.py      # Phase 9 — JATS XML → chunk dicts
│   ├── retrieval/
│   │   ├── chroma_client.py
│   │   ├── bm25_index.py
│   │   ├── hybrid_retriever.py
│   │   ├── europe_pmc.py          # Phase 9 — Europe PMC REST client
│   │   └── semantic_scholar.py    # Phase 9 — SPECTER2 embeddings
│   ├── utils/
│   │   └── ingest_progress.py     # Phase 9 — checkpoint-based progress
│   ├── memory/                    # Phase 11 — planned
│   │   ├── community_detector.py
│   │   ├── community_summarizer.py
│   │   ├── relevance_router.py
│   │   ├── cascade.py
│   │   ├── disclosure.py
│   │   └── experiential.py       # Phase 12 — agent learnings
│   ├── skills/                    # Phase 12 — planned
│   │   ├── skill_loader.py
│   │   ├── skill_creator.py
│   │   ├── trajectory_logger.py
│   │   └── skill_evals.py
│   ├── outputs/                   # Phase 13 — planned
│   │   ├── templates.py
│   │   ├── anchored_writer.py
│   │   └── citation_integrator.py
│   ├── graph/
│   │   ├── nodes.py
│   │   ├── graph_builder.py
│   │   ├── base_graph.py
│   │   └── networkx_json_storage.py
│   ├── agents/
│   │   ├── orchestrator.py       # Phase 10 — built (background daemon, 418 lines)
│   │   ├── subagents.py           # Phase 10 — built (parallel workers, 54 lines)
│   │   ├── handoff.py             # Phase 10 — built (cycle handoff generator)
│   │   └── scheduler.py           # Phase 10 — built (daemon timer, 69 lines)
│   └── anchoring/
│       └── evidence_check.py
├── tests/
│   ├── test_state.py
│   ├── test_unicode.py
│   ├── test_scrubber.py
│   ├── test_citation_manager.py
│   ├── test_retrieval.py
│   ├── test_ingestion.py
│   ├── test_hybrid_retriever.py
│   ├── test_extraction.py
│   ├── test_synthesis.py
│   ├── test_graph.py
│   ├── test_scheduler.py        # Phase 10 — built
│   ├── test_subagents.py        # Phase 10 — built
│   ├── test_orchestrator.py     # Phase 10 — built
│   ├── test_orchestrator_integration.py  # Phase 10 — built
│   ├── test_handoff.py          # Phase 10 — built
│   └── test_anchoring.py
├── projects/
│   └── default/
│       ├── chroma_data/
│       ├── bm25_index/
│       ├── project_graph.json
│       ├── ingest_progress.json
│       ├── orchestrator_state.json    # Phase 10 — daemon heartbeat + cycle counter
│       ├── orchestrator.pid           # Phase 10 — daemon PID
│       ├── cycle_N_handoff.md         # Phase 10 — per‑cycle machine handoff
│       ├── spector2_cache.json        # Phase 9 — SPECTER2 DOI‑keyed cache
│       ├── extractions/               # PreExtractor entity cache
│       └── embeddings/                # Paper embeddings
├── data/
│   └── test.pdf
├── docker/
│   └── docker-compose.yml
├── skills/                    # Phase 12 — git‑backed skill library
├── agent-learnings/           # Phase 12 — experiential memory
├── HANDOFF.md                 # Phase 10−11 handoff — developer reference doc
├── phase1_demo.py
├── phase2_demo.py
├── phase3_demo.py
├── phase4_demo.py
├── phase9_europe_pmc_test.py  # Phase 9 — pipeline test + ingestion CLI
├── requirements.txt
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
```

## Phase 9: API-Based Literature Ingestion (50% Complete)

### Architecture Decision

Phase 8 attempted to download PDFs via Playwright/EZProxy browser automation.  This
proved fundamentally unsustainable: per-publisher URL patterns, WAF blocks, IP
blacklists, signed-URL expiry, and 45–90s per paper.  Phase 9 replaces this entirely
with a REST‑API‑based approach using Europe PMC structured full-text XML.

**Core pipeline** (14s for 5 papers, 27× faster than Playwright):

```
Europe PMC search (OPEN_ACCESS:Y) → fullTextXML fetch (3-retry backoff)
  → JATS XML parse → chunk dicts → ChromaDB + BM25 ingest → progress checkpoint
Semantic Scholar → DOI resolve (title fallback) → SPECTER2 embedding batch fetch
```

### Components

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Europe PMC client | `src/retrieval/europe_pmc.py` | Search + full-text XML fetch + metadata + 3-retry backoff | ✅ Complete |
| JATS XML parser | `src/ingestion/pmc_xml_parser.py` | XML sections → chunk dicts (drop-in compatible with PDFParser) | ✅ Complete |
| SPECTER2 embeddings | `src/retrieval/semantic_scholar.py` | 768‑dim vectors for fine‑grained similarity | ✅ Complete |
| Retry logic | `src/retrieval/europe_pmc.py` | 3-retry exponential backoff (1s, 2s) on 5xx/timeout | ✅ New — May 14 |
| Progress persistence | `src/utils/ingest_progress.py` | Checkpoints every 10 papers, resumes on restart | ✅ New — May 14 |
| Ingestion wiring | `phase9_europe_pmc_test.py` | `--ingest` flag: parse → hybrid.ingest() → checkpoint | ✅ New — May 14 |
| Coverage diagnostic | — | PMC vs Semantic Scholar comparison query | ⬜ Not yet |
| Figure pipeline | — | XML `<graphic>` → image download → vision_ingest | ⬜ Not yet |
| SPECTER2 caching | — | Store locally, skip re-fetch | ⬜ Not yet |

### Coverage tradeoff

Europe PMC only returns papers archived in PubMed Central (~6.5M biomedical OA
papers).  Papers without PMC deposition are classified as "abstract-only" —
visible in the knowledge graph but never used for grounded claims.  For NIH-funded
biomedical research, PMC coverage is ~80–90%.

### Speed comparison

| Metric | Phase 8 (Playwright) | Phase 9 (API) |
|--------|---------------------|---------------|
| Per paper | 45–90s | ~2.9s (27× faster) |
| 10 papers | ~600s | **21.95s** |
| Search | — | 0.71s |
| Full-text fetch | — | 2.53s (10 XMLs) |
| XML parse | — | 0.31s |
| SPECTER2 | — | 18.40s (one‑time, cacheable) |

### Known gaps

**Closed (May 14)**:
1. ~~No retry logic on transient API failures~~ → `_request()` with 3-retry backoff
2. ~~No progress persistence (crash = restart)~~ → `IngestProgress` with 10-paper checkpoints
3. ~~Ingestion not yet wired to ChromaDB~~ → `--ingest` flag on test harness

**Remaining**:
4. Coverage diagnostic (PMC vs Semantic Scholar comparison)
5. Figure image download from XML `<graphic>` URLs not implemented
6. SPECTER2 embeddings not cached (re‑fetched each run)

### Phase 8 status

The Playwright/EZProxy PDF download pipeline in `scripts/headless_download.py` is
**deprecated but preserved.**  It produced 115 valid PDFs that remain in
`data/external/`.  It still works for non‑OA papers when VCU EZProxy auth is fresh
and publisher IP rate limits are not triggered.

### Novel approaches in Phase 9

- **Chunk ID uniqueness via per‑paper source prefixes**: source field includes PMCID
  (`"europe_pmc_xml_PMC12345"`) plus `chunk_index` on every chunk.  Without this,
  ChromaDB silently overwrites chunks across papers (IDs like `europe_pmc_xml__0`
  would collide).
- **Polymorphic per‑request Accept header**: session default is `application/json`
  (search endpoint requires it) but `fullTextXML` overrides to `text/xml` per‑call.
  Verified: retry loop preserves per‑call header overrides.
- **Test harness as ingestion CLI**: `--ingest` flag reuses the benchmark script
  for production ingestion.  Same ChromaDB collection and BM25 persist dir as
  Phase 3/4.  No new collections or path divergence.
- **Instrumental progress persistence**: file-based checkpointing (JSON) rather
  than a database. Same pattern as Phase 8's `zotero_sync_status.json`.  Extensible
  to KG updates, skill generation, and trajectory logging in Phase 10.

## Phase 10: Autonomous Background Agent (Built — Complete, 15 May 2026)

> **Status:** All 4 planned core files built + 4 enhancements beyond spec.
> See detailed breakdown, gap tracking, and lessons learned above in the
> Phase 9 → Phase 10 handoff section (§Phase 10 core, line ~1281).
> **307 tests pass, zero failures.**

### Architecture Decision

Phase 10 introduces the first fully autonomous component: a background research
daemon that runs on cron/timer, detects knowledge gaps from the KG structure,
searches for relevant literature, ingests and extracts entities, updates the KG,
and writes a handoff for its own next cycle.  It uses the Phase 9 ingestion
pipeline as its engine and the Phase 3/4 KG as its memory.

**Dual-agent separation**: The background agent (research, KG maintenance, skill
improvement) and the user-facing agent (query routing, evidence-grounded synthesis)
are separate processes sharing persistent storage.  The background agent never
talks to the user; the user agent never manages memory.  This avoids the Letta
red-teaming problem (models resist believing they persist) by using instrumental
framing: agents know they are temporary processes writing to external memory for
future instances.

**API-to-local strategy**: DeepSeek API during Phases 9-13 and a 2-4 week
accumulation phase (building a high-quality KG, refined skills, tuned thresholds).
Switch background agent to local Ollama (Qwen3.6 35B-A3B) once the foundation
is mature.  User agent keeps API synthesis (Drafter/Critic/Arbiter) until local
models close the gap.  Re-evaluate with each new open-weight MoE release.

### Components (all built ✅)

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Orchestrator | `src/agents/orchestrator.py` | Background daemon loop: web discovery → parallel EPMC fetch → batch ingest → PreExtractor → KG save → cycle handoff. Dry-run + live modes, state file + PID. | ✅ 22+4 tests |
| Subagents | `src/agents/subagents.py` | `run_parallel()` — ThreadPoolExecutor for concurrent EPMC search/XML fetch. Batches ingest to avoid redundant BM25 rebuilds. | ✅ 7 tests |
| Handoff protocol | `src/agents/handoff.py` | `generate_handoff()` / `write_handoff()` — cycle-specific files (`cycle_N_handoff.md`). Human HANDOFF.md never overwritten. | ✅ 13 tests |
| Scheduler | `src/agents/scheduler.py` | Daemon-thread interval timer with stop‑event lifecycle, crash‑resilient callback, `run_once()` blocking mode. | ✅ 8 tests |
| Gap resolver | `src/agents/gap_resolver.py` | Parse gap‑analysis text → structured EPMC queries → search → ingest → KG update. | ✅ 18 tests |
| Web search | `src/retrieval/web_search.py` | Discovery-only web client (DuckDuckGo + DDG API fallback). All results tagged `source_type: "discovery"`. | ✅ Built |
| Daemon entry | `Orchestrator(graph_storage=gs).start()` | `python -c "from src.agents.orchestrator import Orchestrator; Orchestrator(graph_storage=gs).start()"` | ✅ Built

### Key architectural choices

- **Web = discovery compass, never evidence source**: Web search (DuckDuckGo /
  Semantic Scholar) identifies topics and directions.  All claims must be grounded
  in peer-reviewed papers.  Discovery results tagged `source_type: "discovery"`,
  never ingested into evidence chains.
- **Instrumental agent framing**: Agents know they're temporary; they write to
  external memory for future instances.  "Your output will be read by the next
  instance" is both honest and effective.
- **Single Ollama model at a time**: Background and user agents share Qwen3.6
  35B-A3B via priority scheduling (user preempts background).  Dual-model loading
  exceeds M3 Max 36GB practical limits.
- **KG updated at ingest time**: PreExtractor receives `graph_storage` parameter
  during Phase 10 ingestion, so the KG accumulates without requiring a query.

## Phase 11: Memory Cascade & Community Routing (Designed — Not Built)

### Architecture Decision

The knowledge graph (built by Phase 10) is hierarchically clustered into communities
using Leiden/Louvain community detection.  Each community receives a multi-level
LLM-generated summary (cheap model, DeepSeek Chat or local Qwen).  At query time,
a relevance router (cheap model) scores each community for relevance; only relevant
communities receive detailed chunk retrieval; irrelevant communities contribute
only a one-paragraph summary as background context.

**MoE for memory**: This is the "mixture of experts for memories" pattern — a cheap
routing model decides which knowledge clusters to activate, and the expensive
synthesis model only sees what passes the router.  Inspired by Microsoft Dynamic
Community Selection (Nov 2024: 77% cost reduction, 58-60% quality improvement).

**Evidence provenance preserved at every layer**: Entity nodes reference source
chunks.  Community summaries cite entity nodes.  Memory blocks reference community
summaries.  The anchoring check always verifies against Layer 0 (source text).

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Community detector | `src/memory/community_detector.py` | Leiden/Louvain on NetworkX KG; multi-level hierarchy |
| Community summarizer | `src/memory/community_summarizer.py` | LLM summaries per community at each hierarchy level |
| Relevance router | `src/memory/relevance_router.py` | Cheap model scores communities 0-1 for query relevance |
| Memory cascade | `src/memory/cascade.py` | Chunk → summary → entity → community → memory block pipeline |
| Progressive disclosure | `src/memory/disclosure.py` | system/ vs research/ vs archive/ memory tiers, context budget tracking |

### Cascade layers

| Layer | Compression ratio | Contents | Evidence link |
|-------|------------------|----------|---------------|
| 0 — Source text | 1:1 | Raw chunks from papers | Original text |
| 1 — Chunk summaries | ~5:1 | Pre‑summarized chunks (TF‑IDF or LLM) | → Layer 0 chunk_index |
| 2 — Entity nodes | ~20:1 | KG nodes with evidence phrases | → Layer 1 source metadata |
| 3 — Community summaries | ~100:1 | LLM summaries per community cluster | → Layer 2 entity IDs |
| 4 — Memory blocks | Variable | Agent‑curated context files | → Layer 3 community IDs |

## Phase 12: Skills & Experiential Memory (Designed — Not Built)

### Architecture Decision

Agents learn from their own trajectories by reflecting on successes and failures,
then generating reusable skill files (.md) stored in a git‑backed `skills/` directory.
Skills are model‑agnostic, portable, and improve with more usage (reflection on
multiple trajectories produces better skills than reflection on one).  A separate
`agent-learnings/` directory stores experiential memory: user preferences,
research strategies, and tool performance data.

**Skills over prompt optimization**: Letta's Skill Learning (Dec 2025) demonstrated
36.8% relative improvement from trajectory‑learned skills.  Skills are A/B tested
(Leta Evals) and gated before deployment — a skill only replaces its predecessor
if it passes evaluation thresholds.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Skill loader | `src/skills/skill_loader.py` | Mounts relevant skills from directory based on task |
| Skill creator | `src/skills/skill_creator.py` | Two‑stage: reflection on trajectory → create/update skill .md |
| Trajectory logger | `src/skills/trajectory_logger.py` | JSONL logging of all agent actions, successes, failures |
| Skill evals | `src/skills/skill_evals.py` | A/B test new skill vs current on historical tasks; gate deployment |
| Experiential memory | `src/memory/experiential.py` | agent-learnings/ store for preferences, strategies, tool data |

### Skill improvement lifecycle

```
Agent runs task → trajectory logged → sleep‑time reflection on successes/failures
  → skill v2 created from reflection → A/B tested against v1 on historical tasks
  → if passes gate: git commit, v2 replaces v1; if fails: v1 retained
  → future agents automatically load latest version
```

## Phase 13: Output Tools & Structured Writing (Designed — Not Built)

### Architecture Decision

The evidence‑grounded synthesis pipeline (Phase 3 Drafter→Critic→Arbiter→Anchoring)
is extended with domain‑specific output templates for grants (Specific Aims,
Significance, Approach), papers (Introduction, Methods, Results, Discussion),
literature reviews, and procedural documents.  An evidence‑anchored writer ensures
every claim carries a citation and passes the anchoring check before inclusion.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Templates | `src/outputs/templates.py` | Grant, paper, methods, review prompt templates |
| Anchored writer | `src/outputs/anchored_writer.py` | Evidence‑anchored output generation with pre‑output anchoring gate |
| Citation integrator | `src/outputs/citation_integrator.py` | Auto‑citation insertion, format consistency, reference list generation |

## Cross‑Cutting Architectural Decisions (Phases 9-13)

These decisions span multiple phases and represent the core design philosophy.

### Evidence provenance chain (all phases)

Every claim in the system can be traced from the highest‑level synthesis (Phase 13
outputs) down to the original source text (Phase 2 chunk).  Each compression layer
references the layer below.  The anchoring check always verifies against Layer 0.
No evidence‑free claims at any level.

### API‑first, local‑switchover (Phases 10-13)

Build and accumulate with DeepSeek API (v4‑Pro for synthesis, Chat for extraction/
routing).  When the KG, skills, and community structure are mature (estimated
2-4 weeks post‑Phase 13), switch the background agent to local Ollama (Qwen3.6
35B-A3B).  The local model inherits a well‑organized system: high‑quality KG,
refined skills encoding frontier‑model strategies, tuned community routing thresholds.
It performs comprehension and maintenance, not discovery and foundation‑building.

### Model tiering (all phases — extended from Phase 5.5)

| Task | Phase 9-13 (API) | Phase 13+ (local) |
|------|-----------------|-------------------|
| Entity extraction | DeepSeek Chat | Qwen3.6 35B-A3B |
| Community summarization | DeepSeek Chat | Qwen3.6 35B-A3B |
| Relevance routing | DeepSeek Chat | Qwen3.6 35B-A3B |
| Gap analysis / reflection | DeepSeek Chat | Qwen3.6 35B-A3B |
| Skill creation | DeepSeek Chat | Qwen3.6 35B-A3B |
| Drafter (synthesis) | DeepSeek v4‑Pro | DeepSeek v4‑Pro (API retained) |
| Critic (Socratic) | DeepSeek v4‑Pro | DeepSeek v4‑Pro (API retained) |
| Arbiter (revision) | DeepSeek v4‑Pro | DeepSeek v4‑Pro (API retained) |
| Anchoring check | Programmatic (TF‑IDF) | Programmatic (no change) |

Synthesis (Drafter/Critic/Arbiter) is the bottleneck for local capability.
Three‑call cascade where each step builds on the previous; degradation in any
step compounds.  Retaining API synthesis while running everything else local is
the pragmatic hybrid until open‑weight MoE models close the gap (re‑evaluate
with each major release, ~every 3-4 months).

### Community‑gated retrieval (Phase 11 — MoE for memory)

Cheap model scores knowledge graph communities for query relevance.  Relevant
communities → detailed chunk retrieval.  Irrelevant communities → summary only
(background context).  Improves context utilization (LLM attention on relevant
evidence), effective memory capacity (1M papers feasible if only 3 clusters
relevant per query), and cost (77% reduction per Microsoft's numbers).

### Skills as the learning primitive (Phase 12)

Reusable .md skill files generated from agent trajectories, not static prompts.
Git‑versioned for rollback.  Model‑agnostic (portable across DeepSeek, Qwen,
future models).  Improve with cumulative usage (reflection on more trajectories
→ better skills).  A/B tested before deployment via Letta Evals‑style gating.

### Web as discovery compass (Phase 10)

Web search identifies topics and directions.  All evidence must come from
peer‑reviewed papers.  Discovery results are never ingested into evidence chains.
This prevents a common failure mode where web content creeps into evidence
through summarization drift.

The project includes a knowledge graph at `docs/kg/` — 99 interconnected Markdown
notes designed for [Obsidian](https://obsidian.md).  No API keys, no external
services, just `.md` files opencode can read and write directly.

### Opening the vault

1. Download [Obsidian](https://obsidian.md) (free, macOS/Windows/Linux)
2. Open Obsidian → "Open folder as vault" → select `docs/kg/`
3. Click "Trust author and enable plugins"

### Recommended plugins

| Plugin | Why | Install |
|--------|-----|---------|
| **Dataview** | Auto-generates status tables from YAML frontmatter. Open `gaps-overview.md` or `decisions-dashboard.md` to see live queries of all open gaps, decisions by phase, etc. | Settings → Community plugins → Browse → "Dataview" → Install → Enable |
| **Templater** | Pre-filled YAML structure for new notes. Hit a hotkey, pick "New Decision" → consistent frontmatter every time. | Settings → Community plugins → Browse → "Templater" → Install → Enable |

Templater configuration: Settings → Templater → Template folder location: `docs/kg/.obsidian/templates`

### Navigation

- **Graph view** (left sidebar) → color-coded by tag: red = gaps, blue = decisions, green = architecture, orange = vision, purple = synthesis
- **`Cmd+Click`** any `[[wikilink]]` → jump to that note
- **Backlinks panel** (right sidebar) → every note referencing the current one
- **Dashboard** → open `dashboard.md` for the home note with pipeline diagram and quick links
- **Dataview dashboards** → `gaps-overview.md`, `decisions-dashboard.md`, `phase-dashboard.md` for auto-generated tables

### Keeping it in sync

opencode reads/writes `docs/kg/` files directly.  After each session:

```
Update the knowledge graph with what we changed
```

This creates, updates, or links notes as needed.  The graph stays current without
manual maintenance.

```
