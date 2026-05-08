"""Graph reasoning layer — extracts meaningful insights from the knowledge graph
for the synthesis drafter.

Instead of dumping raw node‑link JSON, this module finds:
  - Central entities (highest degree) — concepts that connect everything
  - Bridge entities (high betweenness) — cross‑cutting themes linking clusters
  - 2‑hop neighbourhood — related entities for a given set of query‑matching nodes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


def _build_nx_from_snapshot(snapshot: Dict[str, Any]) -> nx.Graph:
    """Convert a node‑link snapshot dict into a NetworkX graph."""
    g = nx.Graph()
    for node in snapshot.get("nodes", []):
        nid = node.get("id", node.get("node_id", str(node)))
        g.add_node(nid, **{k: v for k, v in node.items() if k not in ("id", "node_id")})
    for edge in snapshot.get("edges", snapshot.get("links", [])):
        src = edge.get("source", edge.get("u", ""))
        tgt = edge.get("target", edge.get("v", ""))
        if src and tgt:
            g.add_edge(src, tgt)
    return g


def compute_graph_insights(
    snapshot: Dict[str, Any],
    query: str = "",
    top_n_central: int = 8,
    top_n_bridge: int = 5,
) -> str:
    """Produce a structured text summary of key graph insights.

    Args:
        snapshot: node‑link dict from ``NetworkXJSONStorage.get_subgraph()``.
        query: the user's original research question (for entity matching).
        top_n_central: number of highest‑degree nodes to report.
        top_n_bridge: number of bridge nodes to report (betweenness centrality).

    Returns:
        A plain‑text summary suitable for injection into the drafter's prompt.
    """
    if not snapshot or not snapshot.get("nodes"):
        return ""

    g = _build_nx_from_snapshot(snapshot)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    if n_nodes == 0:
        return ""

    lines = [f"Knowledge Graph: {n_nodes} entities, {n_edges} relationships."]

    # 1. Central entities (highest degree)
    if n_edges > 0:
        degrees = sorted(g.degree(weight=None), key=lambda x: x[1], reverse=True)
        central = degrees[:top_n_central]
        central_strs = []
        for node_id, deg in central:
            node_type = g.nodes[node_id].get("node_type", "entity")
            evidence = str(g.nodes[node_id].get("evidence", ""))[:100]
            central_strs.append(
                f"  - {node_id} [{node_type}] — {deg} connections"
                + (f"; evidence: {evidence}" if evidence else "")
            )
        if central_strs:
            lines.append("\nCentral concepts (most connected):")
            lines.extend(central_strs)

    # 2. Bridge entities (betweenness centrality, only if graph is non‑trivial)
    if n_nodes > 2 and n_edges > 1:
        try:
            bc = nx.betweenness_centrality(g)
            bridges = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_n_bridge]
            if bridges and bridges[0][1] > 0:
                bridge_strs = []
                for node_id, score in bridges:
                    node_type = g.nodes[node_id].get("node_type", "entity")
                    bridge_strs.append(f"  - {node_id} [{node_type}] — betweenness {score:.3f}")
                if bridge_strs:
                    lines.append("\nBridge entities (connect otherwise‑separate clusters):")
                    lines.extend(bridge_strs)
        except Exception:
            logger.debug("Betweenness centrality computation skipped (graph too small or disconnected).")

    # 3. 2‑hop neighbourhood for query‑matching entities
    if query:
        query_lower = query.lower()
        matching: Set[str] = set()
        for node_id in g.nodes:
            label = str(node_id).lower()
            node_data = g.nodes[node_id]
            evidence = str(node_data.get("evidence", "")).lower()
            if any(term in label or term in evidence for term in query_lower.split() if len(term) > 3):
                matching.add(node_id)

        if matching:
            # Collect 2‑hop neighbourhood
            neighbourhood: Set[str] = set(matching)
            for seed in list(matching):
                for hop1 in g.neighbors(seed):
                    neighbourhood.add(hop1)
                    for hop2 in g.neighbors(hop1):
                        neighbourhood.add(hop2)
            # Show up to 15 related entities (excluding the matching seeds)
            related = neighbourhood - matching
            if related:
                related_list = sorted(related)[:15]
                lines.append(
                    f"\nEntities within 2 hops of query‑matching concepts "
                    f"({len(matching)} matched, {len(related)} related):"
                )
                for node_id in related_list:
                    node_type = g.nodes[node_id].get("node_type", "entity")
                    lines.append(f"  - {node_id} [{node_type}]")

    return "\n".join(lines)
