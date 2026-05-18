from typing import TypedDict, List, Dict, Literal, Optional, NotRequired, Any

class Document(TypedDict):
    id: str
    content: str
    metadata: Dict

class AgentState(TypedDict):
    # Phase 1 fields
    user_query: str
    query_scope: Literal["public", "secure", "both"]
    public_context: List[Document]
    secure_context: List[Document]
    extracted_entities: Dict
    synthesis_draft: str
    citations_used: List[str]
    final_output: str
    human_approved: bool
    routes: Optional[Dict]
    
    # Phase 3 fields (new)
    mode: Literal["quick", "deep", "survey"]
    discovered_categories: Dict
    knowledge_graph_snapshot: Dict
    critic_feedback: str
    synthesis_revised: str
    anchoring_score: float
    ungrounded_claims: List[Dict]
    chunk_summary: str
    ner_entities: List[Dict]

    # Phase 4 Survey Mode fields
    decomposed_themes: List[Dict]
    thematic_clusters: Dict
    per_paper_extractions: Dict
    per_theme_syntheses: Dict
    cross_theme_synthesis: str
    gap_analysis: str

    # Phase 7b: Sectioned Survey Mode fields
    section_plan: NotRequired[List[Dict[str, str]]]
    current_section_index: NotRequired[int]
    section_drafts: NotRequired[Dict[str, str]]
    section_feedback: NotRequired[str]
    section_context: NotRequired[Dict[str, List[Dict]]]
    claim_ledger_json: NotRequired[str]
    figure_context: NotRequired[List[Dict]]

    # Phase 11: Community routing & progressive disclosure
    community_data: NotRequired[Dict]
    community_summaries: NotRequired[Dict]
    relevant_communities: NotRequired[List[int]]
    community_scores: NotRequired[Dict[str, float]]
    disclosure_tier: NotRequired[int]
    disclosure_map: NotRequired[Dict]

    # Runtime configuration (optional)
    num_ctx: NotRequired[int]
    client_kwargs: NotRequired[Dict[str, Any]]
    callback: NotRequired[Any]
