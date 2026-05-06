from src.citation_manager.zotero_adapter import ZoteroAdapter

def test_adapter_instantiation():
    adapter = ZoteroAdapter("123", "fakekey")
    assert adapter.format_citation_key("@smith2023") == "@smith2023"
