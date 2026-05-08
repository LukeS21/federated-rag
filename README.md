```text
Full Phase Recap
Phase   Status
Phase 1 Foundation (state, Unicode, citation, retrieval primitives)   ✅ Complete
Phase 2 PDF Ingestion & Hybrid Retrieval   ✅ Functional
Phase 3 LLM Agents & LangGraph Core (extraction, debate synthesis, KG, anchoring, Deep Mode)   ✅ Complete (May 2026)
Phase 4 Live Citation & Survey Mode (real Zotero API, systematic field mapping)   🔜 Next
Phase 5 Security Hardening & Air‑Gap (Docker isolation, boundary scrubber, penetration testing)   📋 Planned
Phase 6 UI, Polish & Deployment (Streamlit/Gradio, session history, Neo4j adapter, Docker Compose)   📋 Planned
```
___

# 🔬 Secure Federated RAG System – Technical Architecture v3.0

**A Production‑Grade, Local‑First, Multi‑Agent Retrieval‑Augmented Generation Platform
for Biomedical Engineering Research**

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

- **~5–10 minute latency per query**: 7 sequential LLM calls; extraction alone is ~5 minutes due to large prompt (raw chunks + 155 NER entities + full categories)
- **Single-document scale**: `n_results=10` is tuned for one PDF; multi-document retrieval not yet tested
- **No cross-query learning**: each query runs independently; no caching, no session memory
- **Knowledge graph underutilized**: built but not meaningfully traversed during synthesis
- **Survey Mode not implemented**: broad retrieval → theme clustering → per-theme synthesis → gap analysis is deferred to Phase 4

Phase 4: Live Citation & Survey Mode (Weeks 8-9)
Goal: Real Zotero integration + comprehensive literature surveying.

Revised architecture (May 2026): Two‑stage hybrid approach replacing the original representative‑paper model.

Survey Mode flow:
1. **Broad retrieval** — fetch all matching papers from PubMed + local corpus
2. **Thematic clustering** — LLM assigns every paper to 1+ themes; no paper is excluded
3. **Per‑document lightweight extraction** — structured entities (materials, methods, findings) extracted per paper in parallel (~5 sec/paper). All extractions feed the shared persistent KG.
4. **Per‑theme deep synthesis** — full Drafter→Critic→Arbiter debate on ALL papers in each theme (not just representatives). The KG enriches cross‑document context.
5. **Cross‑theme synthesis & gap analysis** — single debate synthesis across all theme outputs + KG, identifying contradictions, missing evidence, and research gaps.

*Why hybrid instead of pure per‑document synthesis?* Per‑document debate (100 papers × 30 sec = 50 min) is too slow. Per‑document extraction (100 papers × 5 sec = 8 min) is fast and lossless. The expensive debate synthesis runs only 5–8 times (per theme), preserving depth without sacrificing completeness.

Deliverables:

ZoteroAdapter upgraded: real API calls for item creation, PDF attachment, CiteKey generation

DOI extraction from Docling metadata or PubMed API lookup

Ingest pipeline: on PDF addition, automatically create Zotero item

Citation keys propagated through extraction (entity → source paper) and synthesis (inline @keys)

Query decomposition agent: breaks complex research questions into theme‑discovery sub‑queries

Thematic clustering agent: assigns papers to 1+ themes using lightweight LLM call

Per‑document extraction agent: reuses Phase 3 ExtractionAgent with source‑filtered chunks; runs in parallel

Per‑theme deep synthesis: reuses Phase 3 debate chain (Drafter→Critic→Arbiter) on all papers in each theme

Cross‑theme synthesis agent: consumes all theme syntheses + KG to produce final survey with gap analysis

Expanded human‑in‑the‑loop gates for survey results (theme review, gap acceptance)

Phase 5: Security Hardening & Air‑Gap (Weeks 10-11)
Goal: True dual‑corpus isolation with network enforcement.

Deliverables:

Docker Compose with three services: orchestrator, public-corpus (internet), secure-corpus (no internet, internal: true)

Two Ollama instances; secure instance fully air‑gapped

BoundaryScrubber node: regex redaction at secure‑public boundary

LangGraph routing updated for query_scope switch

Penetration testing: attempt prompt injection, verify no data leaks

Security audit log

Phase 6: UI, Polish & Deployment (Weeks 12-13)
Goal: Professional, lab‑ready package.

Deliverables:

Streamlit or Gradio UI (localhost only): project management, schema review, synthesis editing, export

Session history: past queries, extracted entities, synthesized drafts

Export formats: Markdown, plain text, JSON (for downstream tools)

Complete test suite: unit + integration + security (target: 90%+ coverage)

Comprehensive documentation: README, quickstart guide, architecture reference

Neo4jStorage adapter (optional upgrade path)

Final Docker Compose production configuration

## 11. Component Interfaces
11.1 Hybrid Retriever
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
│   │   └── pdf_parser.py
│   ├── retrieval/
│   │   ├── chroma_client.py
│   │   ├── bm25_index.py
│   │   └── hybrid_retriever.py
│   ├── graph/
│   │   ├── nodes.py
│   │   ├── graph_builder.py
│   │   ├── base_graph.py
│   │   └── networkx_json_storage.py
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
│   └── test_anchoring.py
├── projects/
│   └── default/
│       ├── chroma_data/
│       ├── bm25_index/
│       └── project_graph.json
├── data/
│   └── test.pdf
├── docker/
│   └── docker-compose.yml
├── phase1_demo.py
├── phase2_demo.py
├── phase3_demo.py
├── requirements.txt
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
```
