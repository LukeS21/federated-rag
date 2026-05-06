from typing import TypedDict, List, Dict, Literal, Optional

class Document(TypedDict):
    id: str
    content: str
    metadata: Dict

class AgentState(TypedDict):
    user_query: str
    query_scope: Literal["public", "secure", "both"]
    public_context: List[Document]
    secure_context: List[Document]
    extracted_entities: Dict
    synthesis_draft: str
    citations_used: List[str]          # e.g., ["@smith2023", "@jones2024"]
    final_output: str
    human_approved: bool
    # Optional fields for intermediate routing
    routes: Optional[Dict]