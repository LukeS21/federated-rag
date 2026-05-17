# 🔬 Federated RAG — AI Research Brain for Biomedical Labs

**A production-grade, local-first, multi-agent platform for autonomous biomedical research — persistent evidence-grounded knowledge, self-updating beliefs, and cross-cycle reasoning.**

```text
Phase   Status
───     ──────
 1      Foundation (state, unicode, citation, retrieval)                       ✅ Complete
 2      PDF Ingestion & Hybrid Retrieval                                       ✅ Complete
 3      LLM Agents & LangGraph Core (extraction, debate, KG, anchoring)        ✅ Complete
 4      Live Citation & Survey Mode                                            ✅ Complete
 5      Security Hardening & Air‑Gap                                           ✅ Complete
 5.5    Local Model Optimization & Speed                                       ✅ Complete
 6      UI, Polish & Deployment (Streamlit, GLiNER‑PII, benchmarking)          ✅ Complete
 6.5    Gap Closure (parallelization, compression, cache, security fuzzer)     ✅ Complete
 7      Vision Pipeline & Multi‑Turn Synthesis                                 ✅ Complete
 8      Publication‑Scale Retrieval                                            ⬜ Deprecated → Phase 9
 9      API‑Based Literature Ingestion (Europe PMC, SPECTER2)                  ✅ Complete
10      Autonomous Background Agent (orchestrator daemon)                      ✅ Complete
10.5    Extraction Hardening (batched extraction, memory mgmt, streaming)        ✅ Complete
11      Memory Cascade & Community Routing                                     ⬜ Designed, partial build
12      Skills & Experiential Memory                                           ⬜ Designed
13      Output Tools & Structured Writing                                      ⬜ Designed
```

---

## Table of Contents

1. [North Star Vision](#1-north-star-vision)
2. [Current State](#2-current-state)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Knowledge Graph Layer](#5-knowledge-graph-layer)
6. [Multi-Agent Synthesis](#6-multi-agent-synthesis)
7. [Extraction Pipeline](#7-extraction-pipeline)
8. [Federated Data Management](#8-federated-data-management)
9. [Execution Modes](#9-execution-modes)
10. [LangGraph Orchestration](#10-langgraph-orchestration)
11. [Background Daemon](#11-background-daemon)
12. [Component Interfaces](#12-component-interfaces)
13. [Testing Strategy](#13-testing-strategy)
14. [Deployment](#14-deployment)
15. [Current Performance](#15-current-performance)
16. [Phase Evolution](#16-phase-evolution)
17. [Architectural Constraints](#17-architectural-constraints)
18. [Planned Capabilities](#18-planned-capabilities)
19. [Obsidian Knowledge Graph](#19-obsidian-knowledge-graph)

---

## 1. North Star Vision

### 1.1 The Goal: An Automated PhD for the Lab

We are building an **AI research brain** — a system that doesn't just answer questions, but _maintains and evolves its understanding of a research domain over time_. Like a PhD student or PI, it should:

- **Know what the lab knows** — ingest local PDFs, data, and results into structured knowledge
- **Find what's missing** — identify gaps in understanding, contradictory claims, unexplored connections
- **Search the world to fill those gaps** — continuously discover and ingest relevant literature
- **Maintain evolving beliefs** — hypotheses that strengthen, weaken, or get revised as new evidence arrives
- **Think across cycles** — not one-shot answers, but persistent reasoning that compounds over days and weeks
- **Propose new directions** — generate novel, testable hypotheses grounded in evidence
- **Present findings proactively** — surface discoveries, contradictions, and emergent insights to the researcher
- **Support researcher tasks** — answer questions, write papers and grants, explore contradictory claims — all evidence-grounded

### 1.2 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Evidence grounding** | Every extracted entity and synthesized claim is traced to a source sentence; anchoring scores gate all output |
| **Persistent, evolving knowledge** | Knowledge graph grows every background cycle; claims tracked with confidence and version history |
| **Hierarchical compression** | Communities, summaries, and attention routing keep context relevant while preserving detail for recall |
| **Deterministic orchestration** | LangGraph state machine routes execution; no unbounded LLM loops |
| **Heterogeneous multi-agent debate** | Different model families resist peer-pressure convergence during synthesis |
| **Local-first, air-gap ready** | Defaults to local Ollama; dual-instance Docker architecture for sensitive/secure data |
| **Schema-less extraction** | The LLM discovers categories from the literature, not from a fixed YAML |

---

## 2. Current State

**As of 16 May 2026 — Phase 10.5 complete, Phase 11 partial build.**

| Metric | Value |
|--------|-------|
| Tests passing | **375** (zero failures) |
| Knowledge graph | ~3,810 nodes, ~262K edges |
| BM25 corpus | 27K+ indexed documents |
| Papers ingested | ~43+ OA papers (multiple daemon cycles) |
| Daemon | Orchestrator runs full cycle every 60 min (web discovery → EPMC → ingest → batched extraction → KG → community detection → handoff) |
| LLM provider (dev) | DeepSeek API (fast iteration) |
| LLM provider (target) | Local Ollama (gemma4:e4b + qwen3.6:35b) |
| UI | Streamlit (`streamlit run app.py`) |

### What's Running

- **Background daemon**: Autonomous cycle every 60 min — discovers new topics via web search, fetches OA papers from Europe PMC, ingests into ChromaDB + BM25, extracts entities with batched evidence‑grouped line‑tagged format, updates the knowledge graph, runs community detection, writes cycle handoff. Between‑batch Ollama process restarts (SIGKILL + `ollama serve`) prevent Metal‑backend GPU fragmentation. The macOS Ollama.app launchd watchdog is disarmed at cycle start so the daemon owns the process lifecycle.
- **Web UI**: Query interface with 4 modes (Survey, Deep, Quick, Sectioned), benchmark dashboard, session history, export
- **Vision pipeline**: Extracts, filters, describes, and embeds figures from ingested PDFs
- **Streaming extraction**: Real‑time token‑level visibility into LLM generation during entity extraction
- **Extraction quality defenses**: Block‑level repetition detection in `_parse_line_tagged`'s `_commit()`; word‑level and character‑level (hyphen‑split) consecutive‑repetition detection catches token spam like `Energy: Energy: Energy: …` and `e-coli-coli-coli…`
- **Yield protocol**: Daemon pauses extraction between papers when sentinel file `projects/default/daemon_yield` exists — unloads gemma4 to free GPU memory for a user query (qwen3.6:35b maxes out the 36 GB M3 Max alone; both models can never be loaded simultaneously)

### Open Problems

- **Ollama GPU memory at the Metal layer is fundamentally opaque** (Gap I). No software‑level API — in any framework (Ollama, llama‑cpp‑python, MLX) — can verify true Metal buffer state. Apple does not expose GPU memory telemetry through a public API. The `/api/ps` endpoint reports Ollama's administrative view, not hardware reality. The 0.8 s "confirmed unloaded" for a 9.6 GB model is physically impossible on Apple Silicon. Process‑death (SIGKILL) is the strongest guarantee available, but even a dead process does not prove Metal freed its pages — only that the PID is gone.
- **Mid‑batch KV‑cache corruption** under sustained generation load. Not fixable by process restarts — it occurs *within a single generation*. The model produces clean output for 100‑150 entities then degrades (prompt‑example regurgitation, `(Skipping…)` loops, hyphenated single‑token spam, 600 s timeouts). Smaller batch sizes (4 chunks) and the existing spam detectors are the only defenses. This appears to be an inherent limitation of llama.cpp's Metal backend on Apple Silicon under sustained token generation.
- **Batch failures discard good entities**. When `_parse_line_tagged` raises `RuntimeError` (repetition detection), the entire batch's partial entities — including 100‑177 correctly parsed entities before the error — are lost. A partial‑save wrapper is designed but not built (see Phase 10.5 remaining gaps in HANDOFF).
- **Phase 11 partial build not yet wired**. `community_detection.py` runs in orchestrator cycles but `community_summarizer.py`, `relevance_router.py`, and `progressive_disclosure.py` are designed and tested but not integrated into the daemon or retrieval pipeline.

### What's Next

Phase 11 community routing is designed and partial‑build committed. The immediate architectural priority is wiring community access into Survey Mode retrieval and integrating progressive KG disclosure. See [Planned Capabilities](#18-planned-capabilities).

---

## 3. System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                  User Interface (Streamlit)                  │
│                  Localhost only, port 8501                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│              LangGraph Orchestrator (State Machine)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Input   │  │ Retrieve │  │ Extract  │  │  Synthesize  │ │
│  │  Router  ├──►  Hybrid  ├──►  Agent   ├──►   Debate     │ │
│  └──────────┘  └──────────┘  └──────────┘  │ (3 roles)    │ │
│                                             └──────┬───────┘ │
│                                                    │         │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────▼────────┐ │
│  │  Security    │  │   Evidence     │  │   Knowledge      │ │
│  │  Scrubber    │◄─┤   Anchoring    │◄─┤   Graph Builder  │ │
│  └──────────────┘  └────────────────┘  └──────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │         Background Daemon (Phase 10 Orchestrator)        ││
│  │  Web → EPMC → Ingest → Extract → KG → Handoff (60 min)  ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────────┐
│  Public Corpus   │    │    Secure Corpus     │
│  ChromaDB + BM25 │    │   ChromaDB + BM25    │
│  (internet OK)   │    │   (AIR‑GAPPED)       │
└─────────────────┘    └─────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────┐
│        Knowledge Graph (NetworkX)        │
│  nodes: entities   edges: co‑occurrence  │
│  persisted to project_graph.json         │
└─────────────────────────────────────────┘
```

**Flow**: User asks a question → hybrid retrieval finds relevant chunks → category discovery identifies themes → SciSpaCy NER + LLM extraction pulls entities → knowledge graph built/updated → multi-agent debate (Drafter→Critic→Arbiter) writes synthesis → evidence anchoring check scores every claim → below threshold triggers second pass or human review → security scrub → output.

**Background loop** (runs independently): Web discovery → EPMC search → parallel XML fetch → batch ingest → PreExtractor → KG update → community detection → cycle handoff → state persist.

---

## 4. Technology Stack

### 4.1 Core Components

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **State Orchestration** | LangGraph ≥0.2 | Deterministic graph routing; built-in `interrupt` for human-in-the-loop; conditional edges |
| **LLM Provider (dev)** | DeepSeek API (deepseek-chat + deepseek-v4-pro) | Fast iteration during development; large context windows |
| **LLM Provider (target)** | Ollama (dual-instance) — gemma4:e4b (~4B active experts, 9.6 GB) + qwen3.6:35b (~3B active MoE, 23 GB) | Zero API cost; air-gap capable; fits in 36 GB M3 Max unified memory |
| **Vector Database** | ChromaDB 0.4.24 | Lightweight, embedded; metadata filtering; persistent collections |
| **Sparse Retriever** | rank-bm25 (BM25Okapi) | Pure Python; exact keyword precision for gene names, alloy codes, PMIDs |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Parameter-free; naturally deduplicates dense + sparse result lists |
| **PDF Parsing** | Docling ≥2.0 | Vision-model-based table/caption preservation; exports markdown tables |
| **Knowledge Graph** | NetworkX (persistent to JSON) | File-based, zero-dependency graph; abstract interface for future Neo4j migration |
| **Citation Manager** | PyZotero + custom abstraction | Adapter pattern via `BaseCitationManager`; real CiteKey generation |
| **NER (First Pass)** | SciSpaCy (en_core_sci_sm) | Deterministic entity candidate detection; ~155 entities per query |
| **Privacy** | GLiNER-PII (570M params, on-device) + regex boundary scrubber | Multi-layer PII detection; configurable scrub patterns |
| **Vision** | Docling picture extraction + gemma4:e4b multimodal description | Zero model rotation overhead (same model as fast-tier text LLM) |

### 4.2 LLM Role Assignment

| Role | Model (Dev) | Model (Target) | Rationale |
|------|------------|----------------|-----------|
| **Drafter** | deepseek-v4-pro | qwen3.6:35b | Best agentic score; drafts initial synthesis with evidence citations |
| **Socratic Critic** | deepseek-v4-pro | qwen3.6:35b (different prompt) | Evidence-grounded questions only; never proposes alternative text |
| **Arbiter** | deepseek-v4-pro | qwen3.6:35b (revision prompt) | Same model as Drafter but revision-focused; resolves Critic questions |
| **Extraction / Summarization** | deepseek-chat | gemma4:e4b | Fast, cheap; condenses existing text rather than generating new claims |
| **Category Discovery** | deepseek-chat | gemma4:e4b | Reads retrieved chunks, identifies recurring themes and variables |
| **Per-theme Synthesis** | deepseek-chat | gemma4:e4b | Lower risk — synthesizes within a single theme; anchoring catches errors |
| **Cross-theme Synthesis** | deepseek-v4-pro | qwen3.6:35b | High reasoning demand — connects themes across papers |
| **Gap Analysis** | deepseek-chat | gemma4:e4b | Configured via `GAP_ANALYSIS_MODEL` env var; validated through RAGAS |

---

## 5. Knowledge Graph Layer

### 5.1 Design: Interface-First, Evolving

The knowledge graph is the system's **external brain** — a structured network of biomedical entities and their relationships. Every consumer codes against the abstract interface, never the concrete storage backend.

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

**Current implementation**: `NetworkXJSONStorage` — file-based, zero-dependency, persisted to `project_graph.json`. Every edge carries temporal metadata: `extracted_at`, `source_paper`, `evidence_phrase`.

**Migration path**: A `Neo4jStorage` adapter exists (ready for when the graph exceeds ~10K edges). Change one config value to swap backends; all consumers are unaware.

### 5.2 Node & Edge Types

| Node Type | Examples |
|-----------|----------|
| `material` | Ti-6Al-4V, TiO₂, rough-hydrophilic Ti |
| `cell_type` | neutrophil, macrophage, CD4+ T cell, MSC |
| `cytokine` | IL-6, TNF-alpha, IL-1beta |
| `model_system` | C57BL/6J mouse, rat tibia |
| `method` | flow cytometry, ELISA, microCT |
| `finding` | "IL-6 elevated in obese mice" |
| `paper` | source paper metadata |

| Edge Type | Meaning |
|-----------|---------|
| `measured_via` | (cytokine) → (method) |
| `observed_in` | (cell_type) → (model_system) |
| `expressed_on` | (finding) → (material) |
| `reported_in` | (finding) → (paper) |
| `upregulated_by` | (cytokine) → (condition) |
| `co_occurs_with` | any two entities appearing in the same chunk |

### 5.3 Graph Construction & Growth

The graph grows through two channels:

1. **Query-time**: During Deep/Survey mode, `GraphBuilder` creates nodes for each extracted entity and edges for co-occurring pairs within the same chunk. Attaches `evidence_phrase` and `source_paper` to every edge.

2. **Background daemon**: The orchestrator runs `PreExtractor` on newly ingested papers, extracts entities, and feeds them into `GraphBuilder` via `graph_storage`. Community detection (Louvain algorithm) runs after each cycle, grouping entities into research clusters (e.g., "titanium surface modification" vs. "macrophage signaling").

**Future: Probabilistic edges** — as the belief store is implemented, edges will carry `confidence` scores that are adjusted over time based on supporting or contradicting evidence discovered in background cycles.

---

## 6. Multi-Agent Synthesis

### 6.1 Core Principle: Heterogeneous, Role-Structured Debate with Evidence Anchoring

Research shows that homogeneous debate causes peer-pressure convergence (agents agree on wrong answers), and iterative closed-system debate degrades evidential grounding. Our architecture addresses both failure modes:

- **Heterogeneous models**: Different model families (Qwen, Gemma) with different reasoning biases
- **Role-structured, not adversarial**: Socratic Critic asks evidence-grounded questions, never proposes rival arguments
- **Evidence-anchored stopping criterion**: Measurable Anchoring Score, not subjective consensus
- **Bounded iterations**: Maximum 2 passes, then human escalation

### 6.2 Agent Roles

**Agent 1: Drafter**

System prompt: _"You are a biomedical literature synthesis drafter. Given extracted entities, evidence summaries, and citation keys, write a concise literature review paragraph. Every factual claim must be traceable to a provided evidence chunk. Use inline citation keys (@author2025). Output plain ASCII only."_

Output: Draft synthesis paragraph with inline citations.

**Agent 2: Socratic Critic**

System prompt: _"You are a Socratic critic. Your job is to identify claims in the draft that lack sufficient evidence or overstate what the evidence supports. For each questionable claim, state what the evidence actually says. Ask a specific question about an unsupported assertion. NEVER propose alternative text or 'correct' the draft. If the draft is fully supported, state: 'NO_CRITIQUE: All claims are evidence-grounded.' Output plain ASCII only."_

Output: List of critiques (or `NO_CRITIQUE`).

**Agent 3: Arbiter**

System prompt: _"You are a biomedical synthesis arbiter. You receive a draft, a Socratic critique, and the original evidence. Revise the draft to address the critique. For each critique, either cite specific evidence that supports the claim or modify/remove the claim. Do not alter claims that were not critiqued. Output plain ASCII only."_

Output: Revised synthesis paragraph.

### 6.3 Evidence Anchoring Check (Programmatic)

After the Arbiter produces a revised synthesis, the system performs an automated check without LLM involvement:

1. **Claim decomposition**: Split synthesis into atomic factual claims via sentence splitting + heuristics
2. **Evidence search**: For each claim, run hybrid retrieval (BM25 + ChromaDB with RRF) against source chunks
3. **Similarity computation**: TF-IDF cosine similarity between claim and best evidence sentence
4. **Anchoring Score**: fraction of claims with cosine similarity ≥ threshold (default 0.35)

**Decision flow**:
- Anchoring Score ≥ 0.85 → synthesis finalized
- Anchoring Score < 0.85 → flagged claims sent back to Arbiter for conditional second pass
- If after second pass Anchoring Score < 0.85 → escalate to human approval gate (LangGraph interrupt)

### 6.4 Flow Diagram

```text
┌──────────┐     ┌──────────────┐     ┌───────────┐
│  Drafter │────►│ Socratic     │────►│  Arbiter  │
└──────────┘     │ Critic       │     └─────┬─────┘
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

---

## 7. Extraction Pipeline

### 7.1 Design Principle: Schema-Less, Query-Conditioned, Evidence-Grounded

The researcher never defines entity categories. The system discovers them from the literature. Every extracted entity is tied to an evidence phrase from the source text.

### 7.2 Category Discovery (Pass 1)

The LLM reads all retrieved chunks and identifies recurring themes, variables, experimental methods, model systems, and measured outcomes. Output format (JSON):

```json
{
  "discovered_categories": [
    {"name": "animal_models", "description": "Murine models including strain and diet conditions", "examples_found": ["C57BL/6J mice", "HFD-induced obesity model"]},
    {"name": "biomaterials", "description": "Implant materials and surface modifications tested", "examples_found": ["rough Ti", "rough-hydrophilic Ti"]}
  ],
  "key_variables": ["cytokine levels", "macrophage polarization", "bone formation"],
  "experimental_methods": ["flow cytometry", "ELISA", "microCT"]
}
```

A LangGraph interrupt at a human checkpoint allows the researcher to accept, remove, or add categories.

### 7.3 Batched Entity Extraction (Pass 2b)

**Problem (Phase 10.5):** Full-paper extraction (37+ chunks in one LLM call) produced 20K‑50K token prompts that caused gemma4:e4b to take 15 min+ per call or hang indefinitely. Two hung papers wasted 90+ minutes per daemon cycle.

**Solution — batched extraction:** Chunks are split into groups of 8. Each batch gets its own LLM call (~6K tokens, 30‑90s on gemma4:e4b). Results merged and deduplicated across batches (`_merge_entity_batches()` — normalizes names, keeps longest evidence). Three call sites: daemon ingest (`PreExtractor`), Survey Mode, Deep Mode.

### 7.4 Evidence-Grouped Output Format (Pass 2b)

**Problem:** The original line-tagged format stated evidence, source, and type per entity, causing the same evidence text to be repeated N times for N entities from the same sentence — wasting 60‑70% of output tokens.

**Solution — grouped format:** Evidence, source, and type are stated once per group. Entities are listed compactly with pipe-delimited per-entity attributes:

```
EVIDENCE: Polymeric nanocarriers like polyethyleneimine, dendrimers, and graphene-based materials offer efficient, non-viral alternatives...
SOURCE: Chunk 1 | europe_pmc_xml_PMC11918598
TYPE: material
ENTITY: polyethyleneimine | DIRECTION: unchanged
ENTITY: dendrimers | DIRECTION: unchanged
ENTITY: graphene-based materials | DIRECTION: elevated
```

The parser (`_parse_line_tagged`) is fully backward-compatible with the old per-entity format. Groups are separated by blank lines.

### 7.5 DIRECTION Field

DIRECTION is constrained to measurable changes vs baseline. Valid values: `elevated`, `decreased`, `increased`, `reduced`, `unchanged`, `upregulated`, `downregulated`, `up`, `down`. For entities where direction makes no sense (materials, methods, equipment, anatomical structures, concepts), DIRECTION is **omitted entirely** — not filled with placeholder values like "source", "characteristic", "application", "general", "N/A", or "target".

### 7.6 Streaming & Repetition Detection

- **Streaming output**: Extraction uses `streaming=True` on the LLM instance with `TokenStreamHandler` — tokens appear in realtime during generation. Degradation (garbage, loops, stalls) is visible immediately.
- **Repetition detection**: The parser compares each committed entity to the previous one. Identical blocks (same entity, evidence, source, type, direction) trigger `RuntimeError` → batch aborted. Catches the most common degradation pattern (LLM stuck repeating output).
- **Per‑batch timeout**: Each batch is wrapped in `ThreadPoolExecutor` with a 600s Python‑level timeout.
- **No retries**: Extraction LLM created with `max_retries=0` — hung batch fails once, pipeline continues.

### 7.7 Evidence Grounding

For each extracted entity, the system verifies:
- An evidence phrase exists in the source chunks
- The phrase actually supports the extracted attribute
- A `source_paper` and `chunk_index` reference is attached

Entities without evidence grounding are discarded or flagged.

---

## 8. Federated Data Management (Air-Gap)

### 8.1 Dual-Corpus Architecture

| Corpus | Contents | Network Access | LLM Instance |
|--------|----------|---------------|--------------|
| Public | PubMed literature, open-access PDFs | Internet (rate-limited) | Ollama instance 1 (internet-accessible) |
| Secure Lab | Internal spreadsheets, grant drafts, unpublished results | None (`internal: true`) | Ollama instance 2 (air-gapped) |

### 8.2 Enforcement Layers

1. **Docker network isolation**: Secure container has no gateway, no DNS. External connections physically impossible.
2. **Boundary scrubber**: Regex redaction at secure→public boundary. Proprietary terms blocked and logged.
3. **LangGraph routing**: `query_scope` field ("public", "secure", "both") controls which retrieval/LLM paths execute.
4. **Per-corpus LLM instances**: Air-gapped Ollama processes only secure-corpus data. No internet access.
5. **GLiNER-PII**: On-device 570M-param model detects names, phone numbers, emails, IDs in output.

---

## 9. Execution Modes

### 9.1 Quick Mode
**Purpose**: Factual lookup with minimal latency.  
**Flow**: Hybrid retrieve → single-pass extraction → single-agent synthesis (no debate).  
**Latency**: ~5-10 seconds.

### 9.2 Deep Mode
**Purpose**: Rigorous evidence synthesis with debate and anchoring.  
**Flow**: Full pipeline — category discovery → two-pass extraction → KG construction → 3-role heterogeneous debate → evidence anchoring (1-2 passes) → scrub.  
**Latency**: ~30-60 seconds (4-5 LLM calls + programmatic checks).

### 9.3 Survey Mode
**Purpose**: Comprehensive literature survey across many papers.  
**Flow**: Query decomposition → broad retrieval → thematic clustering → per-document parallel extraction → per-theme deep synthesis → cross-theme synthesis + gap analysis.  
**Latency**: ~5-10 minutes for subfield survey (100 papers).

### 9.4 Sectioned Mode
**Purpose**: IMRaD-style manuscript section writing.  
**Flow**: Init → retrieve → draft section → review → [route back for more content or proceed] → assemble → scrub.  
**Latency**: ~50 seconds.

---

## 10. LangGraph Orchestration

### 10.1 State Definition

```python
class AgentState(TypedDict):
    user_query: str
    query_scope: Literal["public", "secure", "both"]
    mode: Literal["quick", "deep", "survey", "sectioned"]
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
    # Survey Mode fields
    decomposed_themes: List[Dict]
    thematic_clusters: Dict
    per_theme_syntheses: Dict
    cross_theme_synthesis: str
    gap_analysis: str
    # Sectioned Mode fields
    section_plan: List[Dict]
    section_drafts: Dict[str, str]
    claim_ledger_json: str
    # Phase 11 fields
    community_data: Dict
    community_summaries: Dict
    relevant_communities: List[int]
```

### 10.2 Deep Mode Graph (17 nodes)

```text
InputRouter → Retrieve → Summarize → CategoryDiscovery
  → [Human Checkpoint: review/edit categories]
  → SciSpaCyNER → Extraction → KGBuilder
  → Drafter → Critic → Arbiter → AnchoringCheck1
  → (ArbiterPass2 if score < 0.85) → AnchoringCheck2
  → (HumanGate if score < 0.85) → Scrub → END
```

Conditional routing:
- Critic returns `NO_CRITIQUE` → skip Arbiter, go straight to anchoring
- Anchoring ≥ 0.85 on first pass → finalize
- Anchoring < 0.85 after second pass → human gate
- `query_scope="both"` → boundary scrub before output

### 10.3 Survey Mode Graph (8 nodes)

```text
survey_query_decompose → survey_retrieve → survey_community_route
  → survey_thematic_cluster → survey_per_document_extract
  → survey_per_theme_synthesize → survey_cross_theme_synthesize
  → [Human-in-the-loop gate] → survey_scrub → END
```

### 10.4 Interrupt & Resume

Two human checkpoints via `interrupt_before` with `MemorySaver` checkpointer:
1. After category discovery — review/edit categories before NER runs
2. `human_gate` / `survey_scrub` — final review when anchoring score is low

---

## 11. Background Daemon

The Phase 10 orchestrator runs as an autonomous background daemon, continuously growing the system's knowledge.

### 11.1 Daemon Pipeline

```
┌─ Web discovery ───────────────────────────────────────────────────┐
│  DuckDuckGo → discover_topics(seed_terms)                         │
│  Seed terms: static defaults + top‑N KG entities (by degree)       │
│  Results tagged source_type: "discovery" (never evidence)         │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ Query extraction ────────────────────────────────────────────────┐
│  Snippets / titles → deduplicated, ≥20‑char filtered, capped at 6 │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ Parallel EPMC fetch ─────────────────────────────────────────────┐
│  run_parallel(_fetch_and_parse_for_query, queries, max_workers=4) │
│  Per query: search(oa_only=True) → full_text_xml_batch()          │
│    → PMCXMLParser.parse() (EPMC REST → PMC OAI fallback)          │
│  Skips already‑ingested papers via IngestProgress.is_completed()  │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ Batch ingest ────────────────────────────────────────────────────┐
│  All chunks accumulated → one HybridRetriever.ingest() call       │
│  (avoids redundant BM25 corpus rebuilds from parallel threads)     │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ PreExtractor → KG ───────────────────────────────────────────────┐
│  Sequential per paper.  Batched extraction (8 chunks/call) with     │
│  evidence‑grouped line‑tagged format.  600s Python timeout,         │
│  max_retries=0, block‑level repetition detection.                  │
│  Between‑batch Ollama model resets (keep_alive=0 + /api/ps poll)    │
│  prevent Metal‑backend GPU memory fragmentation.                    │
│  Entity dicts → GraphBuilder → graph_storage.save()                │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ Community detection ─────────────────────────────────────────────┐
│  Louvain algorithm on updated KG. Caches to communities.json.     │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ Handoff + state ─────────────────────────────────────────────────┐
│  write_handoff() → projects/default/cycle_N_handoff.md            │
│  State: orchestrator_state.json (heartbeat, cycle, errors)         │
└───────────────────────────────────────────────────────────────────┘
```

### 11.2 Usage

```python
from src.graph import create_graph_storage
from src.agents.orchestrator import Orchestrator

gs = create_graph_storage(file_path="projects/default/project_graph.json")

# Dry run — see what WOULD happen, no API calls
orch = Orchestrator(graph_storage=gs, dry_run=True)
summary = orch.run_once()

# Single live cycle
orch = Orchestrator(graph_storage=gs)
orch.run_once()

# Daemon mode — runs every 60 min
orch = Orchestrator(graph_storage=gs, interval_minutes=60)
orch.start()   # non-blocking
orch.stop()    # clean shutdown
```

### 11.3 Key Features

- **Dry-run mode**: `Orchestrator(dry_run=True)` runs the full discovery→query cycle but skips all API calls beyond web search. Returns `would_have_queries` in summary.
- **Parallel fetch + batch ingest**: Parallelizes I/O-bound EPMC search/XML fetch, batches all chunks into one ingest call (avoids redundant BM25 rebuilds).
- **Batched extraction (8 chunks/call)**: Prevents prompt‑size hangs on large papers. Evidence‑grouped format saves 60‑70% output tokens. Streaming output provides real‑time visibility. Block‑level repetition detection aborts degraded batches. `max_retries=0` + 600s Python timeout prevent retry spirals.
- **Between‑batch Ollama model resets**: After each extraction batch, the model is unloaded (native Ollama API `keep_alive=0`) and reloaded — confirmed by polling `GET /api/ps` until the model disappears. Prevents llama.cpp Metal‑backend memory fragmentation across 100+ sequential inferences. Auditable via before/after model‑count logging.
- **State persistence**: `orchestrator_state.json` (cycle counter, heartbeat, total ingested, last error) read on restart for crash recovery.
- **PID management**: `orchestrator.pid` written on start, removed on clean shutdown.
- **Cycle handoffs**: `cycle_N_handoff.md` files with KG stats, ingest progress, cycle summary. Human `HANDOFF.md` never overwritten.
- **Log management**: Rotating file logs (5 backups × 5MB) at `projects/default/orchestrator.log`.
- **Handoff cleanup**: Auto-removes cycle handoff files older than 7 days.

---

## 12. Component Interfaces

### 12.1 Hybrid Retriever

```python
class HybridRetriever:
    def __init__(self, chroma: ChromaClient, bm25: BM25Index): ...
    def ingest(self, chunks: List[Dict]) -> None: ...
    def query(self, query: str, n_results: int = 10,
              filter_references: bool = True,
              include_figures: bool = False) -> List[Dict]: ...
```

### 12.2 PDF Parser

```python
class PDFParser:
    def parse(self, pdf_path: Path) -> List[Dict]:
        # Returns chunks with {"text": "...", "metadata": {...}}
```

### 12.3 Knowledge Graph Storage

```python
class BaseGraphStorage(ABC):
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None: ...
    def add_edge(self, source: str, target: str, relation: str, properties: dict) -> None: ...
    def get_neighbors(self, node_id: str, relation: str = None) -> list[dict]: ...
    def get_subgraph(self, node_ids: list[str], depth: int = 1) -> dict: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
```

### 12.4 LLM Provider

```python
def get_chat_model(model=None, temperature=0.0, max_tokens=None, timeout=None) -> ChatOpenAI: ...
def get_chat_model_for_scope(query_scope="public", model=None, ...) -> ChatOpenAI: ...
def resolve_model(model) -> str: ...  # "chat"/"small" → fast tier; "pro"/"large" → reasoning tier
```

### 12.5 Claim Ledger

```python
class ClaimLedger:
    def add_claim(claim_text, section, citations=None, grounded=True, metadata=None) -> Dict: ...
    def find_duplicates(claim_text) -> List[Dict]: ...
    def coverage_report(available_citations=None) -> Dict: ...
    def get_ungrounded_claims() -> List[Dict]: ...
    def save(path=None) -> None: ...
```

### 12.6 Orchestrator

```python
class Orchestrator:
    def __init__(self, *, interval_minutes=60, graph_storage=None,
                 max_papers_per_query=5, seed_terms=None, dry_run=False): ...
    def run_once() -> Dict[str, Any]: ...  # single cycle, blocking
    def start(cooldown_seconds=60) -> None: ...  # daemon loop, non-blocking
    def stop(timeout=30.0) -> None: ...
    def resolve_gaps(gap_analysis_text) -> Dict[str, Any]: ...
```

---

## 13. Testing Strategy

### 13.1 Test Suite

**307 tests passing, zero failures.** Coverage spans unit, integration, security, and vision tests.

| Category | Files | Tests |
|----------|-------|-------|
| Unit — agents | test_extraction_agent, test_gap_resolver, test_query_decomposer, test_community_summarizer, test_relevance_router, test_subagents, test_synthesis_agents | ~50 |
| Unit — retrieval | test_hybrid_retriever, test_retrieval, test_ingestion, test_coverage | ~40 |
| Unit — graph | test_graph_builder, test_langgraph_build_graph, test_survey_graph, test_thematic_clusterer | ~40 |
| Unit — synthesis | test_anchoring, test_synthesis/ | ~30 |
| Unit — infrastructure | test_state, test_unicode, test_citation_manager, test_spector2_cache, test_handoff | ~40 |
| Unit — orchestrator | test_orchestrator, test_scheduler | ~30 |
| Integration | test_phase3_integration, test_phase8_integration, test_phase9_phase10_integration, test_phase11_integration, test_orchestrator_integration | ~20 |
| Security | security/test_security | ~15 |
| Vision | vision/test_figure_* | ~50 |

### 13.2 Key Test Patterns

- **False-claim injection**: Fabricated claims planted in synthesis; 3/3 correctly flagged as ungrounded
- **Out-of-corpus detection**: 3/3 OOC queries scored below 0.40 anchoring threshold
- **Discussion-overlap**: Gap questions searched against paper Discussion sections; 80% genuinely novel
- **Calibrated LLM-as-Judge**: TRUE/FALSE/GRAY calibration before trusting judge scores
- **Security fuzzing**: 1000+ random PHI-like samples across 10 categories

### 13.3 Known Issue

6 pre-existing failures in `test_synthesis_agents.py` — mocks `langchain_ollama.ChatOllama` which was replaced by `langchain_openai.ChatOpenAI` during the Ollama→DeepSeek migration. Not yet updated. All 307 other tests pass.

---

## 14. Deployment

### 14.1 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and model choices

# Launch UI
streamlit run app.py

# Run a single daemon cycle (dry run)
python -c "
from src.graph import create_graph_storage
from src.agents.orchestrator import Orchestrator
gs = create_graph_storage(file_path='projects/default/project_graph.json')
orch = Orchestrator(graph_storage=gs, dry_run=True)
summary = orch.run_once()
print(summary)
"

# Run all tests
python -m pytest tests/ -q --tb=short
```

### 14.2 Docker Compose (Air-Gap)

Three services: orchestrator (port 8501), ollama-public (internet, port 11434), ollama-secure (air-gapped, `internal: true`).

```bash
docker-compose up -d
```

### 14.3 Local Model Setup

```bash
# Pull models (on M3 Max 36GB)
ollama pull gemma4:e4b        # fast tier (~9.6GB)
ollama pull qwen3.6:35b       # reasoning tier (~23GB)

# Configure parallelism
launchctl setenv OLLAMA_NUM_PARALLEL 4
launchctl setenv OLLAMA_MAX_QUEUE 8
```

Set `LLM_PROVIDER=ollama` in `.env` for fully local operation.

---

## 15. Current Performance

### 15.1 Benchmark Results (6-paper corpus, hybrid retrieval)

```
Tier A — Automated:
  Anchoring (hybrid BM25+ChromaDB):  0.993 mean (99.2% grounded)
  Claim density:                      118 claims / 22K chars
  Gap novelty:                        80% (don't match Discussion sections)
  Grounded/inferential:               88% grounded / 12% inferential
  Entity appearance:                  36.2% of pre-extracted entities surface in output
  Debate invocation:                  0% (no theme below 0.50 threshold)

Tier B — LLM-as-Judge (RAGAS):
  Faithfulness:                       4.7/5 grounded, 5.0/5 inferential
  Gap quality:                        4.5/5 novelty, 4.8/5 actionability

API vs Local (1:1 survey graph comparison):
  Metric                   DeepSeek (chat+v4-pro)    Local (gemma4+qwen3.6)
  ─────────────────────    ──────────────────────    ───────────────────────
  Avg Anchor Score         0.969                     0.947
  Per‑theme claims         119                       96
  Elapsed time             212s (3.5 min)            524s (8.7 min)
  Cost                     ~$0.50                    free / air‑gapped
```

> **Scale caveat**: Benchmarks are from a 6-paper corpus with topical overlap from a single lab. Anchoring scores at this scale measure traceability, not factual accuracy. Production-grade benchmarking requires 100+ diverse papers.

### 15.2 Latency by Mode (6 papers)

| Mode | LLM Calls | Latency (Dev/DeepSeek) | Latency (Target/Ollama) |
|------|-----------|----------------------|------------------------|
| Quick | 1-2 | ~5s | ~15s |
| Deep | 4-5 | ~30-60s | ~2-4 min |
| Survey | ~12 | ~3.5 min | ~5-8 min |
| Sectioned | 3-5 | ~50s | ~3-5 min |

---

## 16. Phase Evolution

The system was built in 10 phases over ~2 months (April–May 2026), progressing from basic RAG primitives through a complete autonomous research platform:

- **Phases 1-2**: Foundation — state management, Unicode handling, PDF ingestion, hybrid retrieval (ChromaDB + BM25 with RRF)
- **Phases 3-4**: Core intelligence — LangGraph orchestration, multi-agent debate chain, knowledge graph, Survey Mode with thematic clustering and cross-paper synthesis
- **Phases 5-6**: Production readiness — air-gap security (dual Ollama Docker), Streamlit UI, GLiNER-PII privacy, automated benchmarking, local model optimization
- **Phase 7**: Vision — figure extraction from PDFs, multimodal description via gemma4:e4b, claim/citation ledger with SHA-256 dedup
- **Phases 8-9**: Scale — Europe PMC API ingestion (replacing EZProxy/Playwright), SPECTER2 embeddings, corpus-scale retrieval at 22K+ documents
- **Phase 10**: Autonomy — background daemon (web discovery → EPMC → ingest → KG → handoff every 60 min), line-tagged extraction format, parallel subagents

Full build history with per-phase deliverables, lessons learned, and benchmark details: [`docs/phase-history.md`](docs/phase-history.md).

For current → next phase handoff: [`HANDOFF.md`](HANDOFF.md).

---

## 17. Architectural Constraints

These rules preserve the system's design integrity. **Do not violate them.**

### Extraction & Output
- Do NOT switch extraction output back to JSON — line-tagged format eliminates the 70% parse-failure rate on local models
- Do NOT switch extraction back to full‑prompt (all chunks in one call) — batched extraction (8 chunks/call) prevents 15 min+ hangs
- Do NOT remove the evidence‑grouped output format from the system prompt — saves 60‑70% tokens
- Do NOT remove `max_retries=0` from the extraction LLM instance — retrying hung Ollama requests wastes time in daemon context
- Do NOT remove streaming output (`streaming=True` + `TokenStreamHandler`) from extraction — real‑time visibility is critical for diagnosing degradation
- Do NOT remove the block‑level repetition detector in `_parse_line_tagged`'s `_commit()` — catches identical entity blocks
- Do NOT remove the token‑spam detector (`_detect_token_spam`) — catches word‑level (`Energy: Energy: Energy:`) AND character‑level (`e-coli-coli-coli…`) token repetition
- Do NOT remove the line-tagged parser (`_parse_line_tagged`) or formatters (`_categories_to_line_tagged`, `_entities_to_line_tagged`)
- Do NOT revert `DIRECTION`/`CLAIM` to the old constrained‑direction semantics — the `CLAIM` field captures qualitative changes, quantitative measurements, and states/roles; `DIRECTION` is mapped to `CLAIM` in the parser for backward compatibility
- Do NOT add keyword blacklists or grounding‑check heuristics to the parser — prompt constraints are the correct layer for output quality; programmatic filters create a maintenance arms race
- Plain ASCII output only — Unicode-to-ASCII substitution enforced at extraction, synthesis, and final scrub

### Debate & Synthesis
- Do NOT remove the heterogeneous debate chain (Drafter→Critic→Arbiter)
- Do NOT remove the evidence anchoring check (programmatic TF-IDF cosine scoring against source docs)
- Do NOT remove the 2-pass anchoring architecture (ArbiterPass2 + human gate for scores < 0.85)

### Knowledge Graph
- Do NOT couple consumers to a specific storage backend — all code uses `BaseGraphStorage` interface
- Do NOT remove per-paper source prefixes or `chunk_index` from chunk metadata

### Retrieval & Ingestion
- Do NOT wire `ingest()` into parallel threads — use batch accumulation + `_ingest_chunks_batch()` to avoid redundant BM25 rebuilds
- Do NOT add `Accept: application/json` to EPMC session defaults — breaks OAI-PMH fallback
- Do NOT switch SPECTER2 cache key from DOI to S2 paper_id

### Daemon
- Do NOT remove the dry-run flag from the orchestrator
- Do NOT make `_fetch_and_parse_for_query()` an instance method or closure — module-level functions are compatible with ThreadPoolExecutor
- Do NOT change `write_handoff()` default path to `HANDOFF.md` — always accept explicit `output_path`
- Do NOT remove `orchestrator_state.json` or `orchestrator.pid` management
- Do NOT remove `_reset_ollama()` or its between‑batch call site — it is the always‑safe fallback when process restart is impossible
- Do NOT remove `_restart_ollama_process()` — SIGKILL‑by‑port + `ollama serve` between papers/batches is the only reliable GPU memory flush
- Do NOT remove `_ensure_dedicated_ollama()` — launchd disarm at cycle start is required for process restarts to work on macOS (otherwise the Ollama.app watchdog respawns killed processes, creating ghost runners that each hold ~10 GB GPU memory)
- Do NOT remove the before/after model‑count logging in `_reset_ollama()` — only audit trail for memory reset effectiveness

### Security
- Do NOT route secure-scope queries to DeepSeek API
- Web search results are discovery-only (`source_type: "discovery"`) — never used as evidence
- Do NOT use `lstrip()` for prefix removal — use `removeprefix()` or explicit check

---

## 18. Planned Capabilities

### 18.1 What's Designed (Phases 11-13)

| Phase | Capability | Components |
|-------|-----------|------------|
| **11** | Community Routing | Relevance router gates KG community access; progressive disclosure tiers (system→community→paper); wire community routing into Survey Mode retrieval |
| **12** | Skills & Memory | Skill library (.md files from agent trajectories); JSONL trajectory logging; experiential memory; A/B skill evaluation before deployment |
| **13** | Output Templates | Grant proposal, paper, methods section, and review templates; evidence-anchored writing with auto-citation; pre-output anchoring gate |

### 18.2 Beyond the Current Roadmap

These capabilities are central to the North Star vision and will follow Phase 13:

| Capability | Description |
|-----------|-------------|
| **Persistent Belief Store** | Expand the claim ledger into a living hypothesis tracker. Each hypothesis carries: status (supported/challenged/contradicted/deprecated), confidence (0.0–1.0), evidence_for (citations + excerpts), evidence_against, version_history, first_seen, last_updated. Stored alongside the KG in `projects/default/`. |
| **Contradiction Detection Agent** | A new agent that runs during daemon cycles after ingestion: checks newly extracted entities/claims against existing beliefs, flags contradictions ("paper PMC-X challenges hypothesis #7"), updates confidences. This is the cognitive step that closes the loop between ingestion and belief evolution. |
| **Probabilistic KG Edges** | Edge weights in the knowledge graph carry confidence scores adjusted over time — more confirming papers increase weight, contradictory papers flag and branch. The KG becomes a probabilistic belief system, not a static fact database. |
| **Attention Router** | A lightweight model that decides what to load into active context for any task: which KG communities are relevant? Which hypotheses? Which papers? Loads compressed summaries for relevant content, skips everything else. Builds on Phase 11's community routing. |
| **Hypothesis Generator** | Proposes novel, testable hypotheses based on KG patterns, community intersections, and identified gaps — not just finding what's missing, but generating what's next. |
| **Instance Chaining Protocol** | Formal handoff when agent context fills during long-running tasks: compress state → structured handoff (what's done, pending, key findings, current position) → next agent picks up. Generalizes the cycle handoff pattern. |
| **Full Local AI Migration** | Complete transition from DeepSeek API to local Ollama for all workloads, with performance parity via model optimization and prompt engineering. |

---

## 19. Obsidian Knowledge Graph

An Obsidian vault exists at `docs/kg/` with 99 interconnected notes covering architectural components, decisions, gaps, benchmarks, models, and phases. **This vault is an archived capture from mid-project and is not actively maintained.** It may contain outdated information.

The canonical documentation is:
- **Architecture & current state**: This README
- **Next-phase handoff & constraints**: [`HANDOFF.md`](HANDOFF.md)
- **Full build history**: [`docs/phase-history.md`](docs/phase-history.md)

To explore the vault: `Obsidian → Open folder as vault → docs/kg/`. Useful plugins: Dataview, Templater.
