---
phase: [1, 3, 4, 7]
status: reference
tags: [architecture, state, langgraph]
created: 2026-05-10
links: [system-overview, deep-mode-graph, survey-mode-graph, sectioned-survey-graph]
---

# Agent State

Full AgentState TypedDict with all 34 fields organized by phase.

## Phase 1 Fields (Query & Routing)

| Field | Type | Description |
|-------|------|-------------|
| `user_query` | `str` | Raw user input |
| `query_scope` | `str` | `public`, `secure`, or `all` |
| `public_context` | `str` | Retrieved public chunks |
| `secure_context` | `str` | Retrieved secure chunks |
| `extracted_entities` | `list[dict]` | NER-extracted entities |
| `synthesis_draft` | `str` | Initial draft text |
| `citations_used` | `list[dict]` | Citation metadata |
| `final_output` | `str` | Rendered output |
| `human_approved` | `bool` | Human sign-off |
| `routes` | `list[str]` | Node routing history |

## Phase 3 Fields (Deep Mode)

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `str` | `quick`, `deep`, `survey` |
| `discovered_categories` | `list[dict]` | LLM-discovered themes |
| `knowledge_graph_snapshot` | `dict` | Serialized KG state |
| `critic_feedback` | `str` | Critic critique text |
| `synthesis_revised` | `str` | Arbiter-revised draft |
| `anchoring_score` | `float` | Evidence grounding score |
| `ungrounded_claims` | `list[str]` | Claims below threshold |
| `chunk_summary` | `str` | Condensed chunk text |
| `ner_entities` | `list[dict]` | SciSpaCy entities |

## Phase 4 Fields (Survey Mode)

| Field | Type | Description |
|-------|------|-------------|
| `decomposed_themes` | `list[str]` | Query sub-themes |
| `thematic_clusters` | `dict` | Paper → theme assignments |
| `per_paper_extractions` | `dict` | Entities per paper |
| `per_theme_syntheses` | `dict` | Synthesis per theme |
| `cross_theme_synthesis` | `str` | Final merged output |
| `gap_analysis` | `str` | Research gaps identified |

## Phase 7b Fields (Sectioned Survey)

| Field | Type | Description |
|-------|------|-------------|
| `section_plan` | `NotRequired[list[str]]` | IMRaD section list |
| `current_section_index` | `NotRequired[int]` | Active section pointer |
| `section_drafts` | `NotRequired[dict]` | Drafts per section |
| `section_feedback` | `NotRequired[dict]` | Reviewer feedback |
| `section_context` | `NotRequired[str]` | Prior section summaries |
| `claim_ledger_json` | `NotRequired[str]` | SHA-256 claim registry |
| `figure_context` | `NotRequired[str]` | Figure descriptions |

## Runtime Fields

| Field | Type | Description |
|-------|------|-------------|
| `num_ctx` | `NotRequired[int]` | Ollama context window |
| `client_kwargs` | `NotRequired[dict]` | LLM client parameters |
| `callback` | `NotRequired[callable]` | Streaming callback |

## Total Field Count

34 fields in TypedDict. Phase 7b and runtime fields marked `NotRequired`.
