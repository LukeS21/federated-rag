import os
import tempfile

import pytest

from src.graph.graph_builder import GraphBuilder
from src.graph.networkx_json_storage import NetworkXJSONStorage


@pytest.fixture
def temp_storage():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        # Start from an empty graph even though the file exists
        os.unlink(path)
    except FileNotFoundError:
        pass

    storage = NetworkXJSONStorage(path)
    yield storage

    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def test_build_adds_nodes_and_edges(temp_storage):
    entities = {
        "cytokines": [{"entity": "IL-6", "evidence": "IL-6 elevated", "source": "Chunk 0"}],
        "animal_models": [
            {"entity": "C57BL/6J mice", "evidence": "C57BL/6J mice used", "source": "Chunk 0"}
        ],
    }
    chunks = [
        {"text": "IL-6 was elevated in C57BL/6J mice.", "metadata": {"source": "paper1.pdf"}},
        {"text": "Another chunk unrelated.", "metadata": {"source": "paper1.pdf"}},
    ]

    GraphBuilder().build(entities, chunks, temp_storage)

    assert temp_storage._graph.has_node("cytokines:IL-6")
    assert temp_storage._graph.has_node("animal_models:C57BL/6J mice")

    node_data = temp_storage._graph.nodes["cytokines:IL-6"]
    assert node_data["node_type"] == "cytokines"
    assert node_data["source_paper"] == "paper1.pdf"

    edges = list(temp_storage._graph.edges(data=True))
    # GraphBuilder adds directed edges both ways for each co-occurrence pair
    assert len(edges) == 2
    for _, _, edge_data in edges:
        assert edge_data["relation"] == "co_occurs_with"
        assert edge_data["source_paper"] == "paper1.pdf"
        assert "extracted_at" in edge_data


def test_build_no_edges_if_different_chunks(temp_storage):
    entities = {
        "materials": [{"entity": "Ti-6Al-4V", "evidence": "...", "source": "Chunk 0"}],
        "cell_types": [{"entity": "macrophage", "evidence": "...", "source": "Chunk 1"}],
    }
    chunks = [
        {"text": "material only", "metadata": {"source": "p1.pdf"}},
        {"text": "cell only", "metadata": {"source": "p2.pdf"}},
    ]

    GraphBuilder().build(entities, chunks, temp_storage)

    assert temp_storage._graph.number_of_nodes() == 2
    assert temp_storage._graph.number_of_edges() == 0

