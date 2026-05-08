#!/usr/bin/env python3
"""Phase 4 Visualization — inspect the knowledge graph, entity flows,
thematic clustering, and synthesis quality of the Survey Mode pipeline.

Run AFTER phase4_demo.py to visualize what was built.

Usage:
    python phase4_viz.py                          # full visualization suite
    python phase4_viz.py --graph-only             # just the knowledge graph
    python phase4_viz.py --stats-only             # just entity/theme statistics
    python phase4_viz.py --dataflow               # pipeline data flow diagram
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")  # interactive window by default
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


# ── colour palette per entity type ─────────────────────────────────────────────
TYPE_COLORS = {
    "material":       "#E74C3C",  # red
    "cell_type":      "#3498DB",  # blue
    "cytokine":       "#2ECC71",  # green
    "model_system":   "#9B59B6",  # purple
    "method":         "#F39C12",  # orange
    "finding":        "#1ABC9C",  # teal
    "paper":          "#95A5A6",  # grey
}
DEFAULT_COLOR = "#7F8C8D"


def _node_color(node_type: str) -> str:
    return TYPE_COLORS.get(str(node_type).lower(), DEFAULT_COLOR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Knowledge Graph Loader
# ═══════════════════════════════════════════════════════════════════════════════
def load_knowledge_graph(path: str = "projects/default/project_graph.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]No knowledge graph found at {path}. Run phase4_demo.py first.[/red]")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def build_nx_graph(data: Dict[str, Any]) -> nx.Graph:
    g = nx.Graph()
    for node in data.get("nodes", []):
        nid = node.get("id", node.get("node_id", str(node)))
        node_type = node.get("node_type", "unknown")
        evidence = str(node.get("evidence", ""))[:80]
        g.add_node(nid, node_type=node_type, evidence=evidence)
    for edge in data.get("edges", data.get("links", [])):
        src = edge.get("source", edge.get("u", ""))
        tgt = edge.get("target", edge.get("v", ""))
        rel = edge.get("relation", edge.get("label", "co_occurs_with"))
        if src and tgt:
            g.add_edge(src, tgt, relation=rel)
    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Entity Statistics
# ═══════════════════════════════════════════════════════════════════════════════
def load_entity_stats() -> Dict[str, Dict[str, int]]:
    """Count entities per paper from pre-extraction cache."""
    stats: Dict[str, Dict[str, int]] = {}
    extractions_dir = Path("projects/default/extractions")
    if not extractions_dir.exists():
        return stats
    for path in sorted(extractions_dir.glob("*.json")):
        paper_id = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            counts: Dict[str, int] = {}
            for cat, ent_list in data.items():
                if isinstance(ent_list, list):
                    counts[cat] = len(ent_list)
            stats[paper_id] = counts
        except (json.JSONDecodeError, OSError):
            pass
    return stats


def load_theme_results() -> Optional[Dict[str, Any]]:
    result_path = Path("projects/default/survey_result.json")
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Knowledge Graph Visual
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_node_label(node_id: str) -> str:
    """Strip paper prefixes and keep only the entity name."""
    # Remove paper prefix like "test.pdf::"
    if "::" in node_id:
        node_id = node_id.split("::", 1)[1]
    # For "Category:Entity" format, show just the entity name
    if ":" in node_id:
        parts = node_id.split(":")
        # If last part is very short (like a gene symbol), keep more context
        name = parts[-1].strip()
        if len(name) < 4 and len(parts) >= 2:
            name = ":".join(parts[-2:])
        node_id = name
    return node_id[:40]


def _clean_node_type(node_type: str) -> str:
    """Strip paper prefix from node types for cleaner legend."""
    if "::" in node_type:
        return node_type.rsplit("::", 1)[-1]
    return node_type


def _detect_communities(g: nx.Graph) -> Dict[str, List[str]]:
    """Group nodes into communities using label propagation.

    Returns mapping community_id → list of node IDs.
    """
    try:
        from networkx.algorithms.community import label_propagation_communities
        raw = list(label_propagation_communities(g))
    except (ImportError, Exception):
        # Fallback: group by cleaned node type
        groups: Dict[str, list] = {}
        for node, ndata in g.nodes(data=True):
            t = _clean_node_type(ndata.get("node_type", "unknown"))
            groups.setdefault(t, []).append(node)
        return groups

    return {f"cluster_{i}": sorted(c) for i, c in enumerate(raw) if len(c) >= 2}


def _community_dominant_type(g: nx.Graph, nodes: List[str]) -> str:
    """Return the most common cleaned node type in a community."""
    counts: Dict[str, int] = {}
    for n in nodes:
        t = _clean_node_type(g.nodes[n].get("node_type", "unknown"))
        counts[t] = counts.get(t, 0) + 1
    return max(counts, key=counts.get) if counts else "unknown"


def _community_top_terms(g: nx.Graph, nodes: List[str], top_n: int = 5) -> str:
    """Extract top entity terms from node IDs in a community."""
    terms: Dict[str, int] = {}
    for n in nodes:
        label = _clean_node_label(str(n))
        for word in label.lower().split():
            w = word.strip("()[],;:")
            if len(w) > 3 and w not in ("with", "from", "that", "this", "were", "been"):
                terms[w] = terms.get(w, 0) + 1
    top = sorted(terms, key=terms.get, reverse=True)[:top_n]
    return ", ".join(top)


def viz_knowledge_graph(g: nx.Graph, data: Dict[str, Any]):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    if n_nodes == 0:
        console.print("[yellow]Knowledge graph is empty — no entities extracted yet.[/yellow]")
        return

    console.print(Panel(f"[bold]Knowledge Graph: {n_nodes} entities, {n_edges} relationships[/bold]"))

    # ── community detection ──
    communities = _detect_communities(g)
    n_communities = len(communities)

    # ── community summary table ──
    comm_table = Table(title=f"Knowledge Graph Communities ({n_communities} clusters)", box=box.ROUNDED)
    comm_table.add_column("Cluster", style="bold", width=12)
    comm_table.add_column("Nodes", justify="right", width=6)
    comm_table.add_column("Edges", justify="right", width=6)
    comm_table.add_column("Dominant Type", width=30)
    comm_table.add_column("Key Terms", width=35)
    for cid, nodes in sorted(communities.items(), key=lambda x: -len(x[1])):
        sub = g.subgraph(nodes)
        dom_type = _community_dominant_type(g, nodes)
        terms = _community_top_terms(g, nodes)
        color = _node_color(dom_type)
        comm_table.add_row(
            cid, str(len(nodes)), str(sub.number_of_edges()),
            f"[{color}]{dom_type[:28]}[/{color}]", terms[:33],
        )
    console.print(comm_table)

    # ── network figure with community coloring ──
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.set_title(f"Knowledge Graph — {n_nodes} entities, {n_edges} relationships, {n_communities} clusters",
                 color="white", fontsize=14, fontweight="bold", pad=15)

    # Community colors
    community_colors = plt.cm.tab20(np.linspace(0, 1, max(n_communities, 1)))

    # Build node → community index + color map
    node_to_comm: Dict[str, int] = {}
    for idx, (cid, nodes) in enumerate(sorted(communities.items())):
        for n in nodes:
            node_to_comm[n] = idx

    # Spring layout with stronger repulsion for larger graphs
    k_val = 1.5 if n_nodes > 300 else 0.8
    pos = nx.spring_layout(g, k=k_val, iterations=80, seed=42)

    # Draw communities as background regions
    for idx, (cid, nodes) in enumerate(sorted(communities.items())):
        if len(nodes) < 2:
            continue
        pts = np.array([pos[n] for n in nodes if n in pos])
        if len(pts) < 3:
            continue
        # Convex hull for community
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            color = community_colors[idx % len(community_colors)]
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    color=color, alpha=0.08, linewidth=1.5,
                    edgecolor=color, linestyle="--")
            # Label community at centroid
            centroid = hull_pts.mean(axis=0)
            top_terms = _community_top_terms(g, nodes, 3)
            ax.annotate(f"cluster_{idx}\n{top_terms}",
                        xy=centroid, fontsize=5, color="white", alpha=0.5,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d1a",
                                  edgecolor=color, alpha=0.6, linewidth=0.5))
        except Exception:
            pass

    # Draw nodes colored by community
    for n in g.nodes:
        if n in node_to_comm:
            idx = node_to_comm[n]
            c = community_colors[idx % len(community_colors)]
        else:
            c = "#555555"
        npos = pos.get(n)
        if npos is None:
            continue
        deg = g.degree(n)
        size = max(80, min(500, deg * 40))
        ax.scatter(npos[0], npos[1], s=size, c=[c], edgecolors="#ffffff",
                   linewidths=0.3, alpha=0.85, zorder=2)

    # Edges
    edge_lines = []
    for u, v in g.edges():
        if u in pos and v in pos:
            edge_lines.append([pos[u], pos[v]])
    for seg in edge_lines:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                color="#444444", alpha=0.08, linewidth=0.3, zorder=1)

    # Label nodes with degree >= threshold (more labels than before)
    degree_threshold = max(3, sorted(dict(g.degree()).values(), reverse=True)[:80][-1] if len(g) >= 80 else 0)
    label_nodes = {n for n, d in g.degree() if d >= degree_threshold}
    labels = {n: _clean_node_label(str(n)) for n in label_nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=5, font_color="white",
                            font_weight="bold", ax=ax)

    # Legend — top 15 communities
    legend_elements = []
    top_comms = sorted(communities.items(), key=lambda x: -len(x[1]))[:15]
    for idx, (cid, nodes) in enumerate(top_comms):
        dom_type = _community_dominant_type(g, nodes)
        top = _community_top_terms(g, nodes, 2)
        c = community_colors[idx % len(community_colors)]
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=8, label=f"{cid} | {dom_type[:25]} | {top}")
        )
    if legend_elements:
        legend = ax.legend(handles=legend_elements, loc="upper right",
                          fontsize=6, facecolor="#1a1a2e", edgecolor="#555555",
                          labelcolor="white", title="Clusters (id | dominant type | top terms)",
                          title_fontsize=7)
        legend.get_frame().set_alpha(0.9)
        legend.get_title().set_color("white")

    ax.axis("off")
    plt.tight_layout()

    # ── top entities by degree ──
    top_table = Table(title="Top Entities by Degree (most connected)", box=box.ROUNDED)
    top_table.add_column("Entity", style="bold")
    top_table.add_column("Type", width=25)
    top_table.add_column("Degree", justify="right")
    for n, deg in sorted(g.degree(), key=lambda x: -x[1])[:25]:
        t = _clean_node_type(g.nodes[n].get("node_type", "unknown"))
        color = _node_color(t)
        top_table.add_row(_clean_node_label(str(n)), f"[{color}]{t[:23]}[/{color}]", str(deg))
    console.print(top_table)

    console.print("\n[dim]Close the graph window to continue...[/dim]")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entity Statistics Per Paper
# ═══════════════════════════════════════════════════════════════════════════════
def viz_entity_stats(stats: Dict[str, Dict[str, int]]):
    if not stats:
        console.print("[yellow]No pre-extraction data found in projects/default/extractions/[/yellow]")
        return

    console.print(Panel(f"[bold]Entity Extraction: {len(stats)} papers[/bold]"))

    # Stacked bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Collect all categories
    all_cats: set[str] = set()
    for counts in stats.values():
        all_cats.update(counts.keys())
    cat_list = sorted(all_cats)

    papers = sorted(stats.keys())
    x = np.arange(len(papers))
    width = 0.7
    bottom = np.zeros(len(papers))

    for cat in cat_list:
        values = [stats[p].get(cat, 0) for p in papers]
        color = _node_color(cat)
        bars = ax.bar(x, values, width, bottom=bottom, label=cat, color=color, edgecolor="#333", linewidth=0.3)
        # Label totals
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i, b + val / 2, str(val), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += values

    ax.set_title("Entities Extracted Per Paper (by Category)", color="white", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Entity Count", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([p[:35] + "..." if len(p) > 35 else p for p in papers],
                       rotation=25, ha="right", fontsize=8, color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")

    legend = ax.legend(loc="upper right", fontsize=8, facecolor="#2d2d44",
                       edgecolor="#555", labelcolor="white")
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()

    # Table
    table = Table(title="Entities Per Paper", box=box.ROUNDED)
    table.add_column("Paper", style="bold", width=40)
    table.add_column("Total Entities", justify="right")
    table.add_column("Categories", justify="right")
    for paper in papers:
        counts = stats[paper]
        total = sum(counts.values())
        table.add_row(paper[:40], str(total), str(len(counts)))
    console.print(table)

    console.print("\n[dim]Close the chart window to continue...[/dim]")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  Thematic Clustering Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def viz_thematic_clusters():
    from src.agents.thematic_clusterer import _get_embedder, _SIMILARITY_THRESHOLD
    from src.ingestion.pre_extractor import PreExtractor
    import numpy as np

    # Load cached paper embeddings
    cached = PreExtractor.load_all_embeddings()
    if not cached:
        console.print("[yellow]No paper embeddings cached. Run phase4_demo.py first.[/yellow]")
        return

    console.print(Panel(f"[bold]Thematic Clustering: {len(cached)} papers, similarity threshold={_SIMILARITY_THRESHOLD}[/bold]"))

    paper_ids = sorted(cached.keys())
    embeddings = np.array([cached[pid] for pid in paper_ids])

    # Compute pairwise cosine similarity between papers
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    emb_norm = embeddings / norms
    sim_matrix = np.dot(emb_norm, emb_norm.T)

    # Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(paper_ids)))
    ax.set_yticks(range(len(paper_ids)))
    ax.set_xticklabels([pid[:20] for pid in paper_ids], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([pid[:20] for pid in paper_ids], fontsize=7)
    ax.set_title(f"Paper Similarity Matrix (cosine, threshold={_SIMILARITY_THRESHOLD})",
                 fontsize=12, fontweight="bold")

    # Annotate high-similarity pairs
    for i in range(len(paper_ids)):
        for j in range(len(paper_ids)):
            if i != j and sim_matrix[i, j] >= _SIMILARITY_THRESHOLD:
                ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if sim_matrix[i,j] > 0.6 else "black",
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=10)

    plt.tight_layout()

    # Similarity table
    table = Table(title="Paper Similarity Matrix", box=box.ROUNDED)
    table.add_column("Paper A", style="bold", width=30)
    table.add_column("Paper B", width=30)
    table.add_column("Similarity", justify="right")
    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            sim = sim_matrix[i, j]
            color = "green" if sim >= _SIMILARITY_THRESHOLD else "dim"
            table.add_row(
                paper_ids[i][:30], paper_ids[j][:30],
                f"[{color}]{sim:.3f}[/{color}]",
            )
    console.print(table)

    console.print("\n[dim]Close the chart window to continue...[/dim]")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-Theme Synthesis Quality
# ═══════════════════════════════════════════════════════════════════════════════
def viz_theme_scores(theme_results: Dict[str, Any]):
    themes = theme_results.get("per_theme_syntheses", {})
    if not themes:
        console.print("[yellow]No theme results found in survey_result.json[/yellow]")
        return

    console.print(Panel(f"[bold]Per-Theme Synthesis Quality: {len(themes)} themes[/bold]"))

    # Extract scores
    theme_names = []
    scores = []
    papers_per_theme = []
    for name, ts in sorted(themes.items()):
        theme_names.append(name[:50])
        scores.append(ts.get("anchoring_score", 0))
        papers_per_theme.append(ts.get("num_papers", 0))

    # Bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    colors = []
    for s, n in zip(scores, papers_per_theme):
        if n == 1:
            colors.append("#3498DB")  # blue = single-paper (entity formatting)
        elif s >= 0.85:
            colors.append("#2ECC71")  # green = well-grounded
        elif s >= 0.35:
            colors.append("#F39C12")  # orange = moderately grounded
        else:
            colors.append("#E74C3C")  # red = poorly grounded

    bars = ax.barh(theme_names, scores, color=colors, edgecolor="#444", linewidth=0.5, height=0.7)

    # Threshold lines
    ax.axvline(x=0.85, color="#2ECC71", linestyle="--", linewidth=1, alpha=0.7, label="Well-grounded (0.85)")
    ax.axvline(x=0.35, color="#F39C12", linestyle="--", linewidth=1, alpha=0.7, label="Moderate (0.35)")

    # Annotate papers per theme
    for i, (bar, n) in enumerate(zip(bars, papers_per_theme)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}  ({n}p)", va="center",
                fontsize=9, color="white")

    ax.set_title("Per-Theme Anchoring Scores", color="white", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Anchoring Score", color="white")
    ax.set_xlim(0, 1.1)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.legend(loc="lower right", fontsize=8, facecolor="#2d2d44",
              edgecolor="#555", labelcolor="white")

    plt.tight_layout()

    # Quality summary table
    table = Table(title="Theme Synthesis Quality", box=box.ROUNDED)
    table.add_column("Theme", style="bold", width=40)
    table.add_column("Papers", justify="right")
    table.add_column("Anchoring", justify="right")
    table.add_column("Status")
    for name, score, n in zip(theme_names, scores, papers_per_theme):
        if n == 1:
            status = "[blue]entity-formatted[/blue]"
        elif score >= 0.85:
            status = "[green]well-grounded[/green]"
        elif score >= 0.35:
            status = "[yellow]moderately grounded[/yellow]"
        else:
            status = "[red]poorly grounded[/red]"
        table.add_row(name, str(n), f"{score:.3f}", status)
    console.print(table)

    console.print("\n[dim]Close the chart window to continue...[/dim]")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Flow Pipeline Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def viz_dataflow_diagram():
    """Display the full Survey Mode data flow as a formatted terminal diagram."""

    diagram = Text("""\
                         ┌─────────────────────────────┐
                         │     PDF Files (data/)        │
                         │  6 papers, ~1000 chunks      │
                         └─────────────┬───────────────┘
                                       │ Docling PDF parser
                         ┌─────────────▼───────────────┐
                         │    TF-IDF PreSummarizer      │
                         │  Extractive chunk summaries  │
                         │  Stored in ChromaDB metadata │
                         └─────────────┬───────────────┘
                                       │ PreExtractor (ingest-time)
                         ┌─────────────▼───────────────┐
                         │   Entity Extraction (cache)  │
                         │  26 entity groups total      │
                         │  ─ per paper JSON ─          │
                         │  ─ paper embedding (.npy) ─  │
                         └───────┬──────────┬──────────┘
                                 │          │
              ┌──────────────────┘          └──────────────────┐
    ┌─────────▼──────────┐                          ┌─────────▼──────────┐
    │  Knowledge Graph   │                          │ Embedding Cache    │
    │  NetworkX JSON     │                          │ .npy per paper     │
    │  ~150 nodes        │                          │ all-MiniLM-L6-v2   │
    │  ~300 edges         │                          └──────────┬─────────┘
    └─────────┬──────────┘                                     │
              │           ┌──────────────────────────────────┘
              │           │
    ┌─────────▼───────────▼──────────────────────────────────────────┐
    │                     SURVEY MODE QUERY                          │
    │  "What mechanisms support/challenge this hypothesis?"          │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S1: Query Decomposer  } deepseek-v4-pro  } cache-hit <1ms    │
    │  ── 6 themed sub-queries ──                                    │
    │  • Macrophage senescence                                       │
    │  • Macrophage plasticity                                       │
    │  • Aging + osseointegration                                    │
    │  • Adoptive transfer                                           │
    │  • Inflammatory microenvironment                               │
    │  • Risks and failure modes                                      │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S2: Broad Retrieval  } hybrid (ChromaDB + BM25)  } 0.8s      │
    │  ── L2 ≤ 1.5, max 50 chunks across all papers ──               │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S3: Thematic Clustering  } embeddings  } 2s                   │
    │  ── 6 papers × 6 themes, cosine similarity ≥ 0.35 ──           │
    │  ── 0 unassigned ──                                            │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S4: Per-Document Extraction  } pre-cached from disk  } 0.0s   │
    │  ── 6 papers loaded, 0 needed LLM extraction ──                │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S5: Per-Theme Synthesis  } deepseek-chat  } PARALLEL  } 9.1s  │
    │                                                                 │
    │  ┌─ Theme 1: 1 paper  →  entity-formatted (no LLM)  ──┐       │
    │  │─ Theme 2: 5 papers →  drafter → score 0.46 ────────│       │
    │  │─ Theme 3: 6 papers →  drafter → score 0.58 ────────│───┐   │
    │  │─ Theme 4: 5 papers →  drafter → score 0.55 ────────│───┤   │
    │  │─ Theme 5: 6 papers →  drafter → score 0.67 ────────│───┤   │
    │  └─ Theme 6: 5 papers →  drafter → score 0.36 ────────┘   │   │
    │                                                            │   │
    │  ── 0 Critic calls (all drafts ≥ 0.35 threshold) ──        │   │
    │  ── 0 Arbiter calls                                        │   │
    │  ── KG insights injected into per-theme Drafter prompts    │   │
    └─────────────────────────┬──────────────────────────────────┘   │
                              │                                       │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S6: Cross-Theme Synthesis  } deepseek-v4-pro  } PARALLEL  }  │
    │                                                                 │
    │  ┌─ Cross-theme narrative ── (v4-pro Drafter)  ──────────┐     │
    │  │─ Gap analysis ── (v4-pro Drafter)  ───────────────────│     │
    │  └─ Both run in parallel (no dependency)  ───────────────┘     │
    │                                                                 │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │  S7: Scrub  } ASCII enforcement  } 0.0s                        │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   FINAL OUTPUT     │
                    │   Survey + Gaps    │
                    └────────────────────┘
""")

    console.print(Panel(diagram, title="[bold]Survey Mode Data Flow Pipeline[/bold]", title_align="left",
                        border_style="bright_blue"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 4 Survey Mode Visualization")
    parser.add_argument("--graph-only", action="store_true", help="Show only the knowledge graph")
    parser.add_argument("--stats-only", action="store_true", help="Show only entity statistics")
    parser.add_argument("--themes-only", action="store_true", help="Show only thematic clustering")
    parser.add_argument("--scores-only", action="store_true", help="Show only theme anchoring scores")
    parser.add_argument("--dataflow", action="store_true", help="Show only the data flow diagram")
    parser.add_argument("--graph-path", default="projects/default/project_graph.json",
                        help="Path to project_graph.json")
    args = parser.parse_args()

    # Determine mode
    run_all = not any([args.graph_only, args.stats_only, args.themes_only,
                       args.scores_only, args.dataflow])

    console.print(Panel("[bold bright_blue]Phase 4 Survey Mode Visualization[/bold bright_blue]"),
                  justify="center")
    console.print()

    # Always show dataflow first
    if run_all or args.dataflow:
        viz_dataflow_diagram()
        if not run_all:
            return

    # Knowledge graph
    if run_all or args.graph_only or args.themes_only:
        try:
            data = load_knowledge_graph(args.graph_path)
            g = build_nx_graph(data)
        except SystemExit:
            return

        if run_all or args.graph_only:
            viz_knowledge_graph(g, data)

    # Entity stats
    if run_all or args.stats_only:
        try:
            stats = load_entity_stats()
            viz_entity_stats(stats)
        except Exception as e:
            console.print(f"[red]Error loading stats: {e}[/red]")

    # Thematic clustering (similarity heatmap)
    if run_all or args.themes_only:
        try:
            viz_thematic_clusters()
        except Exception as e:
            console.print(f"[red]Error viz themes: {e}[/red]")

    # Theme scores
    if run_all or args.scores_only:
        theme_results = load_theme_results()
        if theme_results:
            viz_theme_scores(theme_results)
        else:
            console.print("[yellow]No survey_result.json found — scores can't be shown."
                          " Run phase4_demo.py with a query, then re-run this tool.[/yellow]")

    console.print(Panel("[green]Visualization complete.[/green]"), justify="center")


if __name__ == "__main__":
    main()
