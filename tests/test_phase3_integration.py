"""Integration tests for Phase 3 pipeline improvements."""

from unittest.mock import MagicMock, patch

from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims
from src.agents.sci_ner import extract_ner_entities
from src.graph.graph_builder import build_graph
from src.graph.networkx_json_storage import NetworkXJSONStorage

# ---------------------------------------------------------------------------
#  Sample data
# ---------------------------------------------------------------------------
SAMPLE_CHUNKS = [
    {
        "text": "Surface roughness of titanium implants modulates macrophage polarization. "
        "Rough Ti surfaces promoted a pro-inflammatory phenotype, while rough-hydrophilic Ti "
        "shifted toward anti-inflammatory polarization. The proportion of lean-derived "
        "pro-inflammatory macrophages was lower in response to rough-hydrophilic Ti.",
        "metadata": {"source": "test.pdf", "chunk_index": 0},
    },
    {
        "text": "Obesity significantly affects the inflammatory response to modified Ti implants. "
        "Obese mice had significantly more neutrophils, pro-inflammatory macrophages, and T cells "
        "and fewer anti-inflammatory macrophages and mesenchymal stem cells (MSCs). "
        "Bone formation around Ti implants was reduced in obese mice.",
        "metadata": {"source": "test.pdf", "chunk_index": 1},
    },
    {
        "text": "Scanning electron microscopy and confocal microscopy were used to assess surface "
        "roughness. Contact angle goniometry determined surface hydrophilicity. "
        "X-ray photoelectron spectroscopy (XPS) determined the oxide layer composition.",
        "metadata": {"source": "test.pdf", "chunk_index": 2},
    },
]


# ---------------------------------------------------------------------------
#  Anchoring tests
# ---------------------------------------------------------------------------
def test_anchoring_cosine_grounded():
    """Claims that overlap with evidence should be grounded (cosine ≥ 0.35)."""
    claims = [
        "Rough Ti surfaces promote pro-inflammatory macrophage polarization.",
        "Obese mice had more pro-inflammatory macrophages and fewer anti-inflammatory macrophages.",
    ]
    score, ungrounded = compute_anchoring_score(claims, SAMPLE_CHUNKS)
    assert score == 1.0, f"Expected all claims grounded, got {score}"
    assert len(ungrounded) == 0


def test_anchoring_cosine_ungrounded():
    """Claims about unrelated topics should be ungrounded."""
    claims = [
        "Quantum entanglement enables faster-than-light communication between particles.",
    ]
    score, ungrounded = compute_anchoring_score(claims, SAMPLE_CHUNKS)
    assert score == 0.0, f"Expected no claims grounded, got {score}"
    assert len(ungrounded) == 1


def test_decompose_claims():
    """Sentence splitting should produce atomic claims."""
    text = (
        "Rough Ti surfaces promote pro-inflammatory macrophage polarization. "
        "Obesity significantly amplifies this immune response effect. "
    )
    claims = decompose_claims(text)
    assert len(claims) == 2, f"Expected 2 claims, got {len(claims)}"


# ---------------------------------------------------------------------------
#  SciSpaCy NER tests
# ---------------------------------------------------------------------------
def test_sci_ner_extracts_entities():
    """SciSpaCy should extract biomedical entities from chunks."""
    entities = extract_ner_entities(SAMPLE_CHUNKS)
    assert len(entities) > 0, "Expected at least one entity from SciSpaCy NER"
    entity_texts = {e["text"].lower() for e in entities}
    # Should find common biomedical terms
    assert any("macrophage" in t for t in entity_texts), "Expected macrophage entities"
    assert any("titanium" in t for t in entity_texts), "Expected titanium entities"


# ---------------------------------------------------------------------------
#  Graph compilation test
# ---------------------------------------------------------------------------
def test_graph_compiles_with_all_nodes():
    """Full graph with summarization, NER, and human checkpoints compiles."""
    hybrid = MagicMock()
    hybrid.query.return_value = SAMPLE_CHUNKS
    storage = NetworkXJSONStorage("data/tmp_integration_test_graph.json")
    app = build_graph(hybrid, storage)
    assert app is not None
    # Verify all expected nodes are in the graph
    nodes = list(app.get_graph().nodes)
    node_names = {n if isinstance(n, str) else n.name for n in nodes}
    expected = {
        "input_router", "retrieve", "summarize", "category_discovery",
        "sci_ner", "extraction", "kg_builder", "drafter", "critic",
        "arbiter", "arbiter_pass2", "anchoring_check_pass1",
        "anchoring_check_pass2", "scrub", "human_gate",
    }
    missing = expected - node_names
    assert not missing, f"Missing nodes: {missing}"
