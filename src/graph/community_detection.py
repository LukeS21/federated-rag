"""Community detection on the knowledge graph via Louvain algorithm (Phase 11).

Detects research communities in the entity co-occurrence graph, mapping
every node to a community ID. Results are cached to disk for reuse across
query cycles. The graph is converted to undirected for community detection
since co_occurs_with edges are bidirectional in the KG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.community import louvain_communities

from src.graph.base_graph import BaseGraphStorage

logger = logging.getLogger(__name__)

_DEFAULT_COMMUNITY_PATH = Path("projects/default/communities.json")


def detect_communities(
    graph_storage: BaseGraphStorage,
    *,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Run Louvain community detection on the KG.

    Converts the directed entity graph to undirected, runs Louvain,
    and returns community assignments with metadata.

    Args:
        graph_storage: The knowledge graph backend.
        cache_path: Path to read/write cached community data.
            Defaults to ``projects/default/communities.json``.
        force_recompute: If True, skip loading from cache.

    Returns:
        {
            "algorithm": "louvain",
            "modularity": 0.42,
            "n_communities": 5,
            "n_nodes": 232,
            "community_sizes": {0: 48, 1: 55, ...},
            "node_to_community": {"T cell activation:CD4+ T cell": 0, ...},
            "community_nodes": {0: ["T cell activation:CD4+ T cell", ...], ...},
        }
    """
    cache_path = cache_path or _DEFAULT_COMMUNITY_PATH

    if not force_recompute and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("algorithm") and cached.get("node_to_community"):
                logger.info("Loaded community detection from cache: %d communities, %d nodes",
                             cached.get("n_communities", 0), cached.get("n_nodes", 0))
                return cached
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Community cache corrupted, recomputing: %s", e)

    result = _run_detection(graph_storage)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Community detection cached to %s", cache_path)

    return result


def _run_detection(graph_storage: BaseGraphStorage) -> Dict[str, Any]:
    """Core Louvain detection on the graph's internal NetworkX object."""
    g = _get_undirected_graph(graph_storage)
    n_nodes = g.number_of_nodes()
    if n_nodes < 3:
        logger.info("Graph too small for community detection (%d nodes)", n_nodes)
        return {
            "algorithm": "louvain",
            "modularity": 0.0,
            "n_communities": 0,
            "n_nodes": n_nodes,
            "community_sizes": {},
            "node_to_community": {},
            "community_nodes": {},
        }

    communities = list(louvain_communities(g, seed=42))
    modularity = nx.community.modularity(g, communities)

    node_to_community: Dict[str, int] = {}
    community_nodes: Dict[int, List[str]] = {}
    community_sizes: Dict[int, int] = {}

    for cid, nodeset in enumerate(communities):
        community_sizes[cid] = len(nodeset)
        nodes_list = sorted(nodeset)
        community_nodes[cid] = nodes_list
        for nid in nodes_list:
            node_to_community[nid] = cid

    result = {
        "algorithm": "louvain",
        "modularity": round(modularity, 4),
        "n_communities": len(communities),
        "n_nodes": n_nodes,
        "community_sizes": community_sizes,
        "node_to_community": node_to_community,
        "community_nodes": community_nodes,
    }

    logger.info(
        "Louvain detection: %d communities, modularity=%.4f, %d nodes",
        len(communities), modularity, n_nodes,
    )

    return result


def get_community_papers(
    community_data: Dict[str, Any],
    graph_storage: BaseGraphStorage,
) -> Dict[int, List[str]]:
    """Map community IDs to the source papers whose entities belong to that community.

    Args:
        community_data: Output from ``detect_communities()``.
        graph_storage: The knowledge graph backend.

    Returns:
        {community_id: [paper_id, ...]} mapping.
    """
    community_papers: Dict[int, Set[str]] = {
        cid: set() for cid in community_data.get("community_nodes", {})
    }
    node_to_community = community_data.get("node_to_community", {})

    for node_id, cid in node_to_community.items():
        try:
            node_data = graph_storage._graph.nodes[node_id]
            source_paper = node_data.get("source_paper", "")
            if source_paper:
                community_papers.setdefault(cid, set()).add(source_paper)
        except (KeyError, AttributeError):
            continue

    return {cid: sorted(papers) for cid, papers in community_papers.items()}


def get_community_entities(
    community_data: Dict[str, Any],
    graph_storage: BaseGraphStorage,
) -> Dict[int, List[Dict[str, Any]]]:
    """Collect entity details for each community from the KG.

    Args:
        community_data: Output from ``detect_communities()``.
        graph_storage: The knowledge graph backend.

    Returns:
        {community_id: [{node_id, node_type, evidence, source_paper, ...}, ...]}
    """
    community_entities: Dict[int, List[Dict[str, Any]]] = {}
    node_to_community = community_data.get("node_to_community", {})

    for node_id, cid in node_to_community.items():
        try:
            node_data = dict(graph_storage._graph.nodes[node_id])
            entity_info = {
                "node_id": node_id,
                "node_type": node_data.get("node_type", "unknown"),
                "evidence": node_data.get("evidence", ""),
                "source_paper": node_data.get("source_paper", ""),
            }
            community_entities.setdefault(cid, []).append(entity_info)
        except (KeyError, AttributeError):
            continue

    return community_entities


def _get_undirected_graph(graph_storage: BaseGraphStorage) -> nx.Graph:
    """Convert the storage's DiGraph to an undirected copy for community detection."""
    if hasattr(graph_storage, "_graph") and isinstance(graph_storage._graph, nx.Graph):
        return graph_storage._graph.to_undirected()
    # Fallback: create a new undirected graph from node/edge data
    g = nx.Graph()
    try:
        for node_id, node_data in graph_storage._graph.nodes(data=True):
            g.add_node(node_id, **dict(node_data))
        for u, v, edge_data in graph_storage._graph.edges(data=True):
            g.add_edge(u, v, **dict(edge_data))
    except AttributeError:
        logger.warning("Graph storage does not expose internal _graph attribute")
    return g
