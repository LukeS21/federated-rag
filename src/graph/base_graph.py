from abc import ABC, abstractmethod
from typing import Any, Optional, List


class BaseGraphStorage(ABC):
    """Abstract interface for a persistent knowledge graph.

    All components that read or write the graph code against this interface,
    never against a concrete backend.
    """

    @abstractmethod
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        """Insert or update a node.

        Args:
            node_id: Unique identifier for the node.
            node_type: Category of the node (e.g., 'material', 'cytokine').
            properties: Arbitrary key‑value metadata.
        """
        ...

    @abstractmethod
    def add_edge(
        self, source: str, target: str, relation: str, properties: dict
    ) -> None:
        """Add a directed relationship between two existing nodes.

        Args:
            source: ID of the source node.
            target: ID of the target node.
            relation: Type of edge (e.g., 'measured_via', 'observed_in').
            properties: Metadata dict, must include temporal fields
                ('extracted_at', 'source_paper', 'evidence_phrase').
        """
        ...

    @abstractmethod
    def get_neighbors(
        self, node_id: str, relation: Optional[str] = None
    ) -> List[dict]:
        """Return the direct neighbours of a node.

        Args:
            node_id: Node to expand.
            relation: Optionally filter by edge type.

        Returns:
            List of dictionaries, each representing a neighbouring node with its
            properties and the connecting edge's properties.
        """
        ...

    @abstractmethod
    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> dict:
        """Extract a sub‑graph containing the given nodes and neighbours up to *depth*.

        Returns a dict that can be serialised for prompt injection.
        """
        ...

    @abstractmethod
    def query_relationships(
        self, source_type: str, relation: str, target_type: str
    ) -> List[dict]:
        """Return all edges matching the exact source‑node‑type / relation /
        target‑node‑type pattern.

        Useful for hypothesis generation.
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist the graph to the configured storage backend."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the graph from the configured storage backend."""
        ...