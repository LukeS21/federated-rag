"""
Neo4j-backed implementation of BaseGraphStorage for publication-scale graphs.

Uses the official ``neo4j`` Python driver.  Set ``GRAPH_BACKEND=neo4j`` in
``.env`` (or set ``NEO4J_URI``) to activate — all consumers are swapped
transparently through the abstract ``BaseGraphStorage`` interface.

Usage::

    from src.graph.neo4j_storage import Neo4jStorage

    kg = Neo4jStorage(uri="bolt://localhost:7687", user="neo4j", password="pass")
    kg.add_node("IL-6", "cytokine", {"evidence": "..."})
    kg.add_edge("IL-6", "obese_mice", "elevated_in", {"source_paper": "a.pdf"})
    neighbors = kg.get_neighbors("IL-6")
    kg.save()   # no-op (Neo4j is always persistent)


Environment variables:
  - NEO4J_URI:  bolt://localhost:7687 (default)
  - NEO4J_USER: neo4j (default)
  - NEO4J_PASSWORD: password123 (default)
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from src.graph.base_graph import BaseGraphStorage

logger = logging.getLogger(__name__)


class Neo4jStorage(BaseGraphStorage):
    """Neo4j-backed knowledge graph satisfying BaseGraphStorage."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        from neo4j import GraphDatabase

        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints and indexes if they don't exist."""
        try:
            with self._driver.session() as session:
                session.run(
                    "CREATE CONSTRAINT node_id_unique IF NOT EXISTS "
                    "FOR (n:Entity) REQUIRE n.node_id IS UNIQUE"
                )
                session.run(
                    "CREATE INDEX node_type_idx IF NOT EXISTS "
                    "FOR (n:Entity) ON (n.node_type)"
                )
        except Exception as e:
            logger.warning("Could not create Neo4j constraints: %s", e)

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ── BaseGraphStorage interface ──────────────────────────────────────────

    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        with self._driver.session() as session:
            session.run(
                "MERGE (n:Entity {node_id: $node_id}) "
                "SET n.node_type = $node_type, n += $props",
                node_id=node_id,
                node_type=node_type,
                props=properties,
            )

    def add_edge(
        self, source: str, target: str, relation: str, properties: dict
    ) -> None:
        # Ensure nodes exist
        self.add_node(source, "unknown", {})
        self.add_node(target, "unknown", {})

        if "extracted_at" not in properties:
            properties["extracted_at"] = self._now_iso()
        if "source_paper" not in properties:
            properties["source_paper"] = ""
        if "evidence_phrase" not in properties:
            properties["evidence_phrase"] = ""

        with self._driver.session() as session:
            session.run(
                "MATCH (a:Entity {node_id: $source}) "
                "MATCH (b:Entity {node_id: $target}) "
                "MERGE (a)-[r:RELATES {relation: $relation}]->(b) "
                "SET r += $props",
                source=source,
                target=target,
                relation=relation,
                props=properties,
            )

    def get_neighbors(
        self, node_id: str, relation: Optional[str] = None
    ) -> List[dict]:
        if relation:
            query = (
                "MATCH (a:Entity {node_id: $node_id})-[r:RELATES {relation: $rel}]->(b:Entity) "
                "RETURN b.node_id AS node_id, b.node_type AS node_type, "
                "properties(b) AS props, r.relation AS relation, properties(r) AS edge_props"
            )
            params = {"node_id": node_id, "rel": relation}
        else:
            query = (
                "MATCH (a:Entity {node_id: $node_id})-[r:RELATES]->(b:Entity) "
                "RETURN b.node_id AS node_id, b.node_type AS node_type, "
                "properties(b) AS props, r.relation AS relation, properties(r) AS edge_props"
            )
            params = {"node_id": node_id}

        with self._driver.session() as session:
            results = session.run(query, params)
            return [
                {
                    "node_id": record["node_id"],
                    "node_type": record["node_type"],
                    "properties": record.get("props", {}),
                    "edge_relation": record["relation"],
                    "edge_properties": record.get("edge_props", {}),
                }
                for record in results
            ]

    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> dict:
        """Extract a subgraph using variable-length path expansion."""
        if not node_ids:
            return {"nodes": [], "edges": []}

        with self._driver.session() as session:
            results = session.run(
                "MATCH (a:Entity)-[r:RELATES*1..%d]-(b:Entity) "
                "WHERE a.node_id IN $node_ids "
                "RETURN DISTINCT a.node_id AS source, b.node_id AS target, "
                "a.node_type AS source_type, b.node_type AS target_type, "
                "length(r) AS distance" % depth,
                node_ids=node_ids,
            )
            nodes = {}
            edges = []
            for record in results:
                src = record["source"]
                tgt = record["target"]
                nodes[src] = record.get("source_type", "")
                nodes[tgt] = record.get("target_type", "")
                edges.append({
                    "source": src,
                    "target": tgt,
                    "distance": record["distance"],
                })
            return {
                "nodes": [
                    {"node_id": nid, "node_type": ntype}
                    for nid, ntype in nodes.items()
                ],
                "edges": edges,
            }

    def query_relationships(
        self, source_type: str, relation: str, target_type: str
    ) -> List[dict]:
        with self._driver.session() as session:
            results = session.run(
                "MATCH (a:Entity {node_type: $st})-[r:RELATES {relation: $rel}]->(b:Entity {node_type: $tt}) "
                "RETURN a.node_id AS source_id, b.node_id AS target_id, "
                "properties(r) AS edge_props",
                st=source_type,
                rel=relation,
                tt=target_type,
            )
            return [
                {
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "edge_properties": record.get("edge_props", {}),
                }
                for record in results
            ]

    def save(self) -> None:
        """No-op: Neo4j is always persistent."""

    def load(self) -> None:
        """No-op: Neo4j graph is loaded via the driver on each query."""

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()

    def node_count(self) -> int:
        """Return total number of Entity nodes."""
        with self._driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS cnt")
            return result.single()["cnt"]

    def edge_count(self) -> int:
        """Return total number of RELATES edges."""
        with self._driver.session() as session:
            result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS cnt")
            return result.single()["cnt"]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
