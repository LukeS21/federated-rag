"""
Graph storage factory — abstracts backend selection behind BaseGraphStorage.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from src.graph.base_graph import BaseGraphStorage

logger = logging.getLogger(__name__)


def create_graph_storage(
    file_path: str | Path | None = None,
    *,
    backend: str | None = None,
) -> BaseGraphStorage:
    """Create a graph storage backend based on ``GRAPH_BACKEND`` env var.

    Supported backends:
      - ``"networkx_json"`` (default): File-based NetworkX + JSON.
        Requires ``file_path`` pointing to a ``.json`` file.
      - ``"neo4j"``: Neo4j via bolt driver.  Reads ``NEO4J_URI``,
        ``NEO4J_USER``, ``NEO4J_PASSWORD`` from env (or defaults).

    Args:
        file_path: Path to the JSON file for ``networkx_json`` backend.
        backend: Override env var.  If None, reads ``GRAPH_BACKEND``.

    Returns:
        A concrete ``BaseGraphStorage`` instance.
    """
    backend = backend or os.getenv("GRAPH_BACKEND", "networkx_json")

    if backend == "networkx_json":
        from src.graph.networkx_json_storage import NetworkXJSONStorage
        path = file_path or os.getenv("GRAPH_FILE", "projects/default/project_graph.json")
        logger.info("Graph backend: NetworkXJSONStorage @ %s", path)
        return NetworkXJSONStorage(file_path=str(path))

    elif backend == "neo4j":
        from src.graph.neo4j_storage import Neo4jStorage
        storage = Neo4jStorage()
        logger.info("Graph backend: Neo4jStorage @ %s", storage._uri)
        return storage

    else:
        raise ValueError(
            f"Unknown GRAPH_BACKEND '{backend}'. "
            f"Supported: networkx_json, neo4j"
        )
