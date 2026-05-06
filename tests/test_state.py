from src.state import AgentState

def test_state_initialization():
    state: AgentState = {
        "user_query": "test",
        "query_scope": "public",
        "public_context": [],
        "secure_context": [],
        "extracted_entities": {},
        "synthesis_draft": "",
        "citations_used": [],
        "final_output": "",
        "human_approved": False,
        "routes": None
    }
    assert state["user_query"] == "test"