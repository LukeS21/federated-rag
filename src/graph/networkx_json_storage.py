import json
import networkx as nx
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from src.graph.base_graph import BaseGraphStorage


class NetworkXJSONStorage(BaseGraphStorage):
    """File‑based knowledge graph adapter backed by NetworkX.

    The graph is serialised via ``nx.node_link_data`` to a JSON file stored
    at ``file_path``.  All edge metadata includes the temporal fields required
    by the architecture.
    """

    def __init__(self, file_path: str) -> None:
        """Initialise the storage and load any existing graph.

        Args:
            file_path: Path to the persistent JSON file
                (e.g. ``projects/default/project_graph.json``).
        """
        self._file_path = Path(file_path)
        self._graph = nx.DiGraph()
        if self._file_path.exists():
            self.load()

    # ------------------------------------------------------------------
    #  Node management
    # ------------------------------------------------------------------
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        """Insert or update a node. The ``node_type`` is stored as a reserved
        property alongside any user‑supplied ``properties``."""
        safe_props = {"node_type": node_type, **properties}
        self._graph.add_node(node_id, **safe_props)

    # ------------------------------------------------------------------
    #  Edge management
    # ------------------------------------------------------------------
    def add_edge(
        self, source: str, target: str, relation: str, properties: dict
    ) -> None:
        """Add a directed edge. The ``relation`` and temporal metadata fields
        are stored on the edge.  Required keys in ``properties``:
        ``extracted_at``, ``source_paper``, ``evidence_phrase``."""
        # Ensure required temporal keys are present; fill with defaults if missing.
        now = datetime.now(timezone.utc).isoformat()
        edge_props = {
            "relation": relation,
            "extracted_at": properties.get("extracted_at", now),
            "source_paper": properties.get("source_paper", "unknown"),
            "evidence_phrase": properties.get("evidence_phrase", ""),
        }
        self._graph.add_edge(source, target, **edge_props)

    # ------------------------------------------------------------------
    #  Query methods
    # ------------------------------------------------------------------
    def get_neighbors(
        self, node_id: str, relation: Optional[str] = None
    ) -> List[dict]:
        """Return all direct successors of *node_id*, optionally filtered by
        edge type.  The result includes the neighbour node data and the
        edge attributes."""
        neighbors = []
        for _, neighbor_id, edge_data in self._graph.out_edges(
            node_id, data=True
        ):
            if relation and edge_data.get("relation") != relation:
                continue
            neighbor_data = dict(self._graph.nodes[neighbor_id])
            neighbor_data["node_id"] = neighbor_id
            neighbor_data["edge"] = edge_data
            neighbors.append(neighbor_data)
        return neighbors

    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> dict:
        """Return a node‑link representation of the sub‑graph induced by
        the given *node_ids* and their neighbours up to *depth* hops."""
        # Start with the seed nodes
        nodes_to_include = set(node_ids)
        frontier = set(node_ids)
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                for _, v in self._graph.out_edges(n):
                    if v not in nodes_to_include:
                        nodes_to_include.add(v)
                        next_frontier.add(v)
                for u, _ in self._graph.in_edges(n):
                    if u not in nodes_to_include:
                        nodes_to_include.add(u)
                        next_frontier.add(u)
            frontier = next_frontier
            if not frontier:
                break
        subg = self._graph.subgraph(nodes_to_include)
        return nx.node_link_data(subg, edges="edges")

    def query_relationships(
        self, source_type: str, relation: str, target_type: str
    ) -> List[dict]:
        """Return all edges where the source node has *node_type* == *source_type*,
        the target node has *node_type* == *target_type*, and the edge’s
        ``relation`` matches exactly."""
        results = []
        for u, v, edge_data in self._graph.edges(data=True):
            if edge_data.get("relation") != relation:
                continue
            u_type = self._graph.nodes[u].get("node_type")
            v_type = self._graph.nodes[v].get("node_type")
            if u_type == source_type and v_type == target_type:
                results.append(
                    {
                        "source": u,
                        "source_properties": dict(self._graph.nodes[u]),
                        "target": v,
                        "target_properties": dict(self._graph.nodes[v]),
                        "edge": edge_data,
                    }
                )
        return results

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Serialise the graph to JSON using the ``node_link`` format."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph, edges="edges")
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Load the graph from the JSON file."""
        with open(self._file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data, edges="edges")