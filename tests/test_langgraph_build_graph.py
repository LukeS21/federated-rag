from unittest.mock import MagicMock

from src.graph.graph_builder import build_graph
from src.graph.networkx_json_storage import NetworkXJSONStorage


def test_build_graph_compiles():
    # HybridRetriever is only used via its `.query` method in the retrieve node,
    # so a small stub is enough for compilation.
    hybrid = MagicMock()
    hybrid.query.return_value = []
    storage = NetworkXJSONStorage("data/tmp_graph_test.json")

    app = build_graph(hybrid, storage)
    assert app is not None

