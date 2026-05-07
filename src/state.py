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

    # Runtime configuration (optional)
    num_ctx: NotRequired[int]
    client_kwargs: NotRequired[Dict[str, Any]]
    callback: NotRequired[Any]
