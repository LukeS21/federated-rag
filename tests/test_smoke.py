from src.state import AgentState
from src.unicode_map import scrub_unicode
from src.citation_manager.zotero_adapter import ZoteroAdapter
from src.retrieval.chroma_client import ChromaClient

def test_end_to_end_foundation():
    # 1. State
    state: AgentState = {
        "user_query": "TiO2 nanotubes",
        "query_scope": "public",
        "public_context": [],
        "secure_context": [],
        "extracted_entities": {},
        "synthesis_draft": "",
        "citations_used": [],
        "final_output": "",
        "human_approved": False,
        "routes": None,
        "mode": "deep",
        "discovered_categories": {},
        "knowledge_graph_snapshot": {},
        "critic_feedback": "",
        "synthesis_revised": "",
        "anchoring_score": 0.0,
    }
    assert state["query_scope"] == "public"

    # 2. Unicode
    clean = scrub_unicode("TiO₂ nanotubes")
    assert "TiO2" in clean

    # 3. Citation manager
    zm = ZoteroAdapter("fake", "fake")
    key = zm.add_item({"title": "Test Article"})
    assert key.startswith("@placeholder")

    # 4. Retrieval
    chroma = ChromaClient("smoke_test")
    chroma.add_documents(["1"], ["TiO2 nanotubes improve bone growth"])
    res = chroma.query("TiO2")
    assert len(res['documents'][0]) > 0