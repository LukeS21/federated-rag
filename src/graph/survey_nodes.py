"""Survey Mode node functions for the LangGraph pipeline (Phase 4).

These nodes implement the two-stage hybrid Survey Mode architecture:
  1. Query decomposition → thematic clusters
  2. Broad retrieval → per-document parallel extraction → per-theme debate → cross-theme synthesis
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import tiktoken

from src.agents.arbiter import Arbiter
from src.agents.extraction_agent import ExtractionAgent
from src.agents.query_decomposer import QueryDecomposer
from src.agents.socratic_critic import SocraticCritic
from src.agents.synthesis_drafter import SynthesisDrafter
from src.agents.thematic_clusterer import ThematicClusterer
from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims
from src.graph.base_graph import BaseGraphStorage
from src.graph.graph_builder import GraphBuilder
from src.graph.graph_reasoning import compute_graph_insights
from src.graph.networkx_json_storage import NetworkXJSONStorage
from src.cache.query_cache import (
    cache_query_decomposition, load_query_decomposition,
    cache_theme_synthesis, load_theme_synthesis,
    cache_cross_theme, load_cross_theme,
)
from src.graph.community_detection import detect_communities, get_community_papers
from src.ingestion.pre_extractor import PreExtractor
from src.retrieval.hybrid_retriever import HybridRetriever
from src.scrubber import final_scrub
from src.security.audit_log import get_audit_logger
from src.security.boundary_scrubber import default_boundary_scrubber
from src.state import AgentState
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

# Threshold below which the Critic is invoked in multi-paper themes.
# Drafts with anchoring score >= this threshold skip Critic (EGSR pattern).
# Raised from 0.35 to 0.50 for local models — fewer themes trigger the
# expensive full debate chain.  Tune with OLLAMA_NUM_PARALLEL for speed.
CONDITIONAL_CRITIC_THRESHOLD = 0.50

# Default model tiering: per-theme tasks use the fast/cheap tier.
# Dual-model parallelism (Phase 5.5): themes are split across two
# different models, giving true GPU parallelism without memory
# multiplication (different models = different weights = independent
# GPU streams).  gemma4:e4b handles half the themes, medgemma:4b
# handles the other half.  Falls back to single-model if alt is unset.
PER_THEME_DRAFTER_MODEL = os.getenv("OLLAMA_SMALL_MODEL", "gemma4:e4b")
PER_THEME_MODEL_B = os.getenv("OLLAMA_ALT_MODEL", "")
CROSS_THEME_DRAFTER_MODEL = os.getenv("OLLAMA_LARGE_MODEL", "deepseek-v4-pro")
GAP_ANALYSIS_MODEL = os.getenv("GAP_ANALYSIS_MODEL", PER_THEME_DRAFTER_MODEL)
PER_THEME_MAX_WORKERS = int(os.getenv("PER_THEME_MAX_WORKERS", "2"))

# Tokenizer for accurate context-window estimation (lazy init)
_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


# ---------------------------------------------------------------------------
#  Memoized agent constructors (shared across themes — avoids redundant
#  ChatOpenAI instantiations for identical config)
# ---------------------------------------------------------------------------
_drafter_cache: Dict[str, SynthesisDrafter] = {}
_critic_cache: Dict[str, SocraticCritic] = {}
_arbiter_cache: Dict[str, Arbiter] = {}


def _clear_agent_caches() -> None:
    """Clear memoized agent instances (useful for testing)."""
    _drafter_cache.clear()
    _critic_cache.clear()
    _arbiter_cache.clear()


def _get_drafter(
    num_ctx: int,
    client_kwargs: dict | None,
    model: str | None = None,
) -> SynthesisDrafter:
    key = f"drafter:{model or 'default'}"
    if key not in _drafter_cache:
        _drafter_cache[key] = SynthesisDrafter(
            num_ctx=num_ctx, client_kwargs=client_kwargs, model=model,
        )
    return _drafter_cache[key]


def _get_critic(
    num_ctx: int,
    client_kwargs: dict | None,
) -> SocraticCritic:
    if "critic" not in _critic_cache:
        _critic_cache["critic"] = SocraticCritic(
            num_ctx=num_ctx, client_kwargs=client_kwargs,
        )
    return _critic_cache["critic"]


def _get_arbiter(
    num_ctx: int,
    client_kwargs: dict | None,
) -> Arbiter:
    if "arbiter" not in _arbiter_cache:
        _arbiter_cache["arbiter"] = Arbiter(
            num_ctx=num_ctx, client_kwargs=client_kwargs,
        )
    return _arbiter_cache["arbiter"]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _fit_summaries_to_context(
    summaries: List[str],
    num_ctx: int,
    max_ratio: float = 0.7,
    overhead_tokens: int = 3000,
) -> List[str]:
    """Dynamically cap summaries to fit within the context window.

    Replaces hardcoded ``summaries[:20]`` with a dynamic cap based on
    available context window size.  Fills summaries until approximately
    *max_ratio* of the context window is consumed (default 70%), reserving
    *overhead_tokens* for the system prompt, entity JSON, and citations.

    Uses tiktoken (cl100k_base) for exact token counting.
    """
    if not summaries:
        return []
    enc = _get_tokenizer()
    max_tokens = max(1, int(num_ctx * max_ratio) - overhead_tokens)
    selected: List[str] = []
    token_count = 0
    for s in summaries:
        est_tokens = len(enc.encode(s))
        if token_count + est_tokens > max_tokens and selected:
            break
        selected.append(s)
        token_count += est_tokens
    return selected


# ---------------------------------------------------------------------------
#  Node S1: Query Decomposition
# ---------------------------------------------------------------------------
def survey_query_decompose_node(state: AgentState) -> Dict[str, Any]:
    """Decompose broad research question into thematic sub-queries.

    Uses Level 1 query cache — if the same query has been decomposed
    before (same text + same # of papers), the cached decomposition is
    returned instantly.
    """
    query = state["user_query"]
    scope = state.get("query_scope", "public")

    try:
        audit = get_audit_logger()
        audit.log_scope_routing(
            query_scope=scope,
            mode="survey",
            routing_decision="survey_pipeline",
            context_keys=["public_context"],
        )
        audit.log_access(operation="query_decompose", resource="survey", query=query)
    except Exception:
        pass

    # Count unique papers for cache invalidation
    from pathlib import Path as _Path
    doc_count = 0
    extractions_dir = _Path("projects/default/extractions")
    if extractions_dir.exists():
        doc_count = len(list(extractions_dir.glob("*.json")))

    # Level 1 cache check
    cached = load_query_decomposition(query, doc_count)
    if cached is not None:
        return {"decomposed_themes": cached}

    decomposer = QueryDecomposer()
    result = decomposer.decompose(query)
    themes = result.get("themes", [])
    logger.info("Query decomposed into %d themes: %s", len(themes),
                 [t.get("theme", "?") for t in themes])

    # Cache for future queries
    cache_query_decomposition(query, doc_count, themes)

    return {
        "decomposed_themes": themes,
    }


# ---------------------------------------------------------------------------
#  Node S2: Broad Retrieval
# ---------------------------------------------------------------------------
def survey_retrieve_node(
    state: AgentState, hybrid_retriever: HybridRetriever
) -> Dict[str, Any]:
    """Broad retrieval — fetch chunks from all matching papers, no exclusions.

    Uses a looser threshold (L2 ≤ 1.5) and higher max (50 chunks) to ensure
    full paper coverage for thematic clustering.  Respects query_scope:
    populates ``public_context`` and/or ``secure_context`` based on scope.
    """
    query = state["user_query"]
    scope = state.get("query_scope", "public")
    chunks = hybrid_retriever.query(
        query,
        similarity_threshold=1.5,
        max_chunks=50,
        filter_references=True,
        include_figures=True,
    )
    logger.info("Survey retrieve: %d chunks across papers (scope=%s)", len(chunks), scope)

    # Separate figure chunks for downstream use
    figure_chunks = [c for c in chunks if (c.get("metadata", {}) or {}).get("chunk_type") == "figure"]
    text_chunks = [c for c in chunks if (c.get("metadata", {}) or {}).get("chunk_type") != "figure"]
    logger.info("Survey retrieve: %d text + %d figure chunks", len(text_chunks), len(figure_chunks))

    updates: Dict[str, Any] = {}
    if scope in ("public", "both"):
        updates["public_context"] = chunks
    if scope in ("secure", "both"):
        updates["secure_context"] = chunks
    return updates


# ---------------------------------------------------------------------------
#  Node S2b: Community Route (Phase 11)
# ---------------------------------------------------------------------------
def survey_community_route_node(
    state: AgentState,
    graph_storage: BaseGraphStorage | None = None,
) -> Dict[str, Any]:
    """Detect communities and route query to relevant research clusters.

    Runs Louvain community detection on the KG (cached to disk), loads
    community summaries, and determines which communities are relevant to
    the user's query. Chunks from irrelevant communities are filtered out
    of the retrieval results.

    Graceful degradation: if community detection fails or no communities
    are found, passes through all chunks unchanged.
    """
    chunks = state.get("public_context") or []
    query = state["user_query"]

    if graph_storage is None or not chunks:
        return {}

    updates: Dict[str, Any] = {}

    try:
        community_data = detect_communities(graph_storage)
        n_comm = community_data.get("n_communities", 0)
        if n_comm == 0:
            logger.info("Community routing: no communities detected — pass-through")
            return {"community_data": community_data}

        updates["community_data"] = community_data

        # Load or generate community summaries
        try:
            from src.agents.community_summarizer import CommunitySummarizer
            summarizer = CommunitySummarizer()
            summaries = summarizer.summarize(graph_storage, community_data=community_data)
            updates["community_summaries"] = summaries
        except Exception as e:
            logger.warning("Community summarization failed: %s", e)
            summaries = {}

        # Route query to relevant communities
        try:
            from src.agents.relevance_router import RelevanceRouter
            router = RelevanceRouter()
            routing = router.route(query, summaries)
            relevant = routing.get("relevant_communities", [])
            scores = routing.get("scores", {})
            method = routing.get("method", "embedding")

            updates["relevant_communities"] = relevant
            updates["community_scores"] = scores

            logger.info(
                "Community routing (%s): %d/%d communities relevant — %s",
                method, len(relevant), n_comm,
                [f"c{cid}:{scores.get(cid, 0):.2f}" for cid in relevant[:5]],
            )

            # Filter chunks to relevant communities' papers
            if relevant:
                community_papers = get_community_papers(community_data, graph_storage)
                relevant_papers: set = set()
                for cid in relevant:
                    relevant_papers.update(community_papers.get(cid, []))

                if relevant_papers:
                    filtered = [
                        ch for ch in chunks
                        if (ch.get("metadata", {}) or {}).get("source", "unknown") in relevant_papers
                    ]
                    dropped = len(chunks) - len(filtered)
                    if dropped > 0:
                        logger.info("Community routing: filtered %d/%d chunks (%d relevant papers)",
                                      dropped, len(chunks), len(relevant_papers))
                        updates["public_context"] = filtered
                else:
                    logger.info("Community routing: no papers found for relevant communities — pass-through")

        except Exception as e:
            logger.warning("Relevance routing failed: %s — pass-through", e)

    except Exception as e:
        logger.warning("Community detection failed: %s — pass-through", e)

    return updates


# ---------------------------------------------------------------------------
#  Node S3: Thematic Clustering
# ---------------------------------------------------------------------------
def survey_thematic_cluster_node(state: AgentState) -> Dict[str, Any]:
    """Assign every paper to one or more themes."""
    chunks = state.get("public_context") or []
    themes = state.get("decomposed_themes", [])

    if not themes:
        logger.warning("No themes to cluster — skipping.")
        return {"thematic_clusters": {}}

    # Build paper summaries from pre-summarized chunks, excluding unknown sources
    papers_by_source: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        src = meta.get("source", "unknown")
        if not src or src == "unknown":
            continue
        if src not in papers_by_source:
            papers_by_source[src] = {
                "id": src,
                "title": src,
                "summary_parts": [],
            }
        summary = meta.get("chunk_summary", ch.get("text", "")[:300])
        papers_by_source[src]["summary_parts"].append(summary)

    papers = []
    paper_id_by_index: Dict[str, str] = {}  # "0" → "test.pdf" mapping for fallback
    for idx, (src, info) in enumerate(papers_by_source.items()):
        papers.append({
            "id": src,
            "title": src,
            "summary": " ".join(info["summary_parts"][:10]),
        })
        paper_id_by_index[str(idx)] = src

    clusterer = ThematicClusterer()
    result = clusterer.cluster(papers, themes)
    clusters = result.get("clusters", {})
    unassigned = result.get("unassigned", [])

    # Fix numeric IDs → actual paper IDs (LLM sometimes returns indices despite prompt)
    fixed_clusters: Dict[str, list] = {}
    for theme_name, paper_ids in clusters.items():
        if isinstance(paper_ids, list):
            fixed = []
            for pid in paper_ids:
                pid_str = str(pid)
                if pid_str in paper_id_by_index:
                    fixed.append(paper_id_by_index[pid_str])
                elif pid_str in papers_by_source:
                    fixed.append(pid_str)
                else:
                    logger.warning("Unknown paper ID in cluster '%s': %s", theme_name, pid_str)
            if fixed:
                fixed_clusters[theme_name] = fixed
        else:
            fixed_clusters[theme_name] = paper_ids

    logger.info("Thematic clustering: %d clusters, %d unassigned papers",
                 len(fixed_clusters), len(unassigned))
    if unassigned:
        logger.info("Unassigned papers: %s", unassigned)

    return {"thematic_clusters": fixed_clusters}


# ---------------------------------------------------------------------------
#  Node S4: Per-Document Extraction (pre-extracted entities priority)
# ---------------------------------------------------------------------------
def _extract_one_paper(
    chunks: List[Dict[str, Any]],
    query: str,
    extraction_model: str,
) -> Dict[str, Any]:
    """LLM extraction fallback for papers without pre-extracted entities."""
    agent = ExtractionAgent(model=extraction_model)
    summary_chunks = []
    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        s = meta.get("chunk_summary", ch.get("text", "")[:200])
        summary_chunks.append({"text": s, "metadata": meta})
    categories = agent.discover_categories(summary_chunks, query)
    entities = agent.extract_entities_batched(chunks, categories, query)
    return entities


def survey_per_document_extract_node(
    state: AgentState, graph_storage: BaseGraphStorage
) -> Dict[str, Any]:
    """Retrieve per-paper entities — pre-extracted from disk first, LLM fallback.

    Pre-extraction at ingest time eliminates ~60% of query-time LLM calls.
    Only papers without pre-extracted entities trigger fresh LLM extraction.
    """
    chunks = state.get("public_context") or []
    query = state["user_query"]

    # Group chunks by source paper (exclude unknown sources)
    chunks_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        src = meta.get("source", "") or ""
        if not src or src == "unknown":
            continue
        if src not in chunks_by_source:
            chunks_by_source[src] = []
        chunks_by_source[src].append(ch)

    paper_ids = sorted(chunks_by_source.keys())
    if not paper_ids:
        logger.info("Per-document extraction: no papers to process.")
        return {"per_paper_extractions": {}, "extracted_entities": {}}

    per_paper: Dict[str, Any] = {}
    all_entities: Dict[str, List[Dict[str, Any]]] = {}
    papers_needing_extraction: List[str] = []

    # Load pre-extracted entities for each paper
    for src in paper_ids:
        cached = PreExtractor.load(src)
        if cached is not None:
            per_paper[src] = cached
            for cat, ent_list in cached.items():
                key = f"{src}::{cat}"
                if key not in all_entities:
                    all_entities[key] = []
                all_entities[key].extend(ent_list if isinstance(ent_list, list) else [])
            logger.info("  %s: loaded %d pre-extracted entity groups", src, len(cached))
        else:
            papers_needing_extraction.append(src)

    # LLM extraction for papers not pre-extracted
    if papers_needing_extraction:
        logger.info("Per-document extraction: %d paper(s) need LLM extraction",
                     len(papers_needing_extraction))
        extraction_model = "deepseek-chat"
        max_workers = min(len(papers_needing_extraction), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for src in papers_needing_extraction:
                future = executor.submit(
                    _extract_one_paper,
                    chunks_by_source[src],
                    query,
                    extraction_model,
                )
                futures[future] = src

            for future in as_completed(futures):
                src = futures[future]
                try:
                    entities = future.result()
                    per_paper[src] = entities
                    for cat, ent_list in entities.items():
                        key = f"{src}::{cat}"
                        if key not in all_entities:
                            all_entities[key] = []
                        all_entities[key].extend(ent_list if isinstance(ent_list, list) else [])
                    # Cache for future queries
                    PreExtractor()._save(src, entities)
                    # Feed to KG
                    try:
                        GraphBuilder().build(entities, chunks_by_source[src], graph_storage)
                    except Exception:
                        pass
                    logger.info("  %s: %d entity groups extracted + cached", src, len(entities))
                except Exception as e:
                    logger.error("  %s: extraction failed — %s", src, e)
                    per_paper[src] = {}

    logger.info("Per-document extraction complete: %d papers, %d entity groups",
                 len(per_paper), len(all_entities))

    return {
        "per_paper_extractions": per_paper,
        "extracted_entities": all_entities,
    }


# ---------------------------------------------------------------------------
#  Node S5: Per-Theme Deep Synthesis
# ---------------------------------------------------------------------------
def _run_debate_for_theme(
    theme_name: str,
    theme_chunks: List[Dict[str, Any]],
    theme_entities: Dict[str, Any],
    query: str,
    num_ctx: int,
    client_kwargs: dict | None,
    num_papers: int = 1,
    drafter_model: str | None = None,
    graph_storage: BaseGraphStorage | None = None,
) -> Dict[str, Any]:
    """Run Drafter→[Critic→Arbiter]→Anchoring for one theme.

    Optimizations:
      - Single-paper themes: format pre-extracted entities directly,
        skipping the Drafter LLM call entirely.
      - Configurable Drafter model (deepseek-chat for per-theme,
        v4-pro for cross-theme).
      - Debate regression guard: if the debate chain produces a worse
        anchoring score than the draft, the draft is kept.
      - KG insights injected into the drafter prompt when available.
    """
    single_paper = num_papers <= 1

    # Build summary from chunk summaries, entity evidence, or fallback text
    summaries: List[str] = []
    for ch in theme_chunks:
        meta = ch.get("metadata", {}) or {}
        cs = meta.get("chunk_summary", ch.get("text", "")[:300])
        if cs:
            summaries.append(cs)

    if not summaries and theme_entities:
        for key, ent_list in theme_entities.items():
            if isinstance(ent_list, list):
                for ent in ent_list:
                    if isinstance(ent, dict):
                        ev = ent.get("evidence", "")
                        if ev:
                            summaries.append(f"[{ent.get('entity', '?')}] {ev}")
        if summaries:
            logger.info("  '%s': using %d entity evidence phrases", theme_name, len(summaries))

    # ── Figure descriptions ─────────────────────────────────────────────
    figure_descs = _extract_figure_descriptions(theme_chunks)
    if figure_descs:
        summaries.extend(figure_descs)
        logger.info("  '%s': added %d figure descriptions to evidence", theme_name, len(figure_descs))

    # Dynamic evidence cap
    fitted_summaries = _fit_summaries_to_context(summaries, num_ctx)
    logger.debug("  '%s': %d/%d summaries fit in context window (num_ctx=%d)",
                 theme_name, len(fitted_summaries), len(summaries), num_ctx)
    summary_text = "\n\n".join(fitted_summaries)
    summary_chunks = [{"text": summary_text, "metadata": {"source": f"theme:{theme_name}"}}]

    citations = sorted({(ch.get("metadata", {}) or {}).get("cite_key") or
                         (ch.get("metadata", {}) or {}).get("source", "unknown")
                         for ch in theme_chunks})
    if not citations and theme_entities:
        for key, ent_list in theme_entities.items():
            if isinstance(ent_list, list):
                for ent in ent_list:
                    if isinstance(ent, dict):
                        src = ent.get("source_paper") or ent.get("cite_key") or ""
                        if src:
                            citations.append(src)
        citations = sorted(set(citations))

    # ── Level 2 cache: theme synthesis ──────────────────────────────────
    # Paper IDs for this theme (from the chunks, unique sources)
    theme_paper_ids = sorted({(ch.get("metadata", {}) or {}).get("source", "unknown")
                               for ch in theme_chunks if (ch.get("metadata", {}) or {}).get("source")})
    cached_theme = load_theme_synthesis(theme_name, theme_paper_ids, query, theme_chunks)
    if cached_theme is not None:
        return cached_theme

    def _return_with_cache(result: Dict[str, Any]) -> Dict[str, Any]:
        cache_theme_synthesis(theme_name, theme_paper_ids, query, theme_chunks, result)
        return result

    # ── Single-paper themes: format entities directly, no Drafter ──────
    if single_paper:
        logger.info("  '%s': single-paper theme — formatting entities (no Drafter)", theme_name)
        synthesis = _format_single_paper_synthesis(theme_name, theme_entities, summary_text, citations)
        claims = decompose_claims(synthesis)
        score, ungrounded = compute_anchoring_score(claims, theme_chunks or summary_chunks)
        return _return_with_cache({
            "theme": theme_name,
            "synthesis": synthesis,
            "anchoring_score": round(score, 3),
            "ungrounded_claims": ungrounded,
            "num_papers": num_papers,
        })

    # ── Multi-paper: Drafter with KG insights ──────────────────────────
    kg_context: Dict[str, Any] | str = {}
    if graph_storage is not None:
        try:
            kg_snapshot = graph_storage.get_subgraph([])
            if kg_snapshot and kg_snapshot.get("nodes"):
                kg_context = compute_graph_insights(
                    kg_snapshot, query=query,
                    top_n_central=5, top_n_bridge=3,
                )
                if kg_context:
                    logger.debug("  '%s': KG insights (%d chars) injected into Drafter",
                                 theme_name, len(_kg_str(kg_context)))
        except Exception:
            pass

    # Compress entities — the Drafter already receives full evidence summaries,
    # so the entity list only needs to cue key entities and their attributes
    compressed_entities = _compress_entities_for_drafter(theme_entities)

    drafter = _get_drafter(num_ctx, client_kwargs, model=drafter_model)
    draft = drafter.draft(
        query=f"{query} [Theme: {theme_name}]",
        entities=compressed_entities,
        chunks=summary_chunks,
        citations=citations,
        kg_context=kg_context,
    )

    # ── EGSR: Conditional Critic ────────────────────────────────────────
    draft_claims = decompose_claims(draft)
    draft_score, draft_ungrounded = compute_anchoring_score(
        draft_claims, theme_chunks or summary_chunks,
    )
    logger.info("  '%s': draft anchoring=%.3f (%d claims, %d ungrounded)",
                theme_name, draft_score, len(draft_claims), len(draft_ungrounded))

    if draft_score >= 0.85:
        logger.info("  '%s': draft is well-grounded (>=0.85) — skipping Critic/Arbiter",
                     theme_name)
        return _return_with_cache({
            "theme": theme_name,
            "synthesis": draft,
            "anchoring_score": round(draft_score, 3),
            "ungrounded_claims": draft_ungrounded,
            "num_papers": num_papers,
        })

    if draft_score >= CONDITIONAL_CRITIC_THRESHOLD:
        logger.info("  '%s': draft is moderately grounded (>=%.2f) — skipping Critic/Arbiter",
                     theme_name, CONDITIONAL_CRITIC_THRESHOLD)
        return _return_with_cache({
            "theme": theme_name,
            "synthesis": draft,
            "anchoring_score": round(draft_score, 3),
            "ungrounded_claims": draft_ungrounded,
            "num_papers": num_papers,
        })

    # Poorly grounded — run full debate chain
    logger.info("  '%s': draft is poorly grounded (<%.2f) — invoking Critic→Arbiter",
                theme_name, CONDITIONAL_CRITIC_THRESHOLD)
    critic = _get_critic(num_ctx, client_kwargs)
    critique = critic.critique(draft, summary_chunks, theme_entities)

    arbiter = _get_arbiter(num_ctx, client_kwargs)
    if critique.startswith("NO_CRITIQUE"):
        revised = draft
    else:
        revised = arbiter.revise(draft, critique, summary_chunks)

    claims = decompose_claims(revised)
    score, ungrounded = compute_anchoring_score(claims, theme_chunks or summary_chunks)

    # ── Debate regression guard ────────────────────────────────────────
    if score < draft_score:
        logger.info("  '%s': debate regression (%.3f → %.3f) — keeping draft",
                     theme_name, draft_score, score)
        return _return_with_cache({
            "theme": theme_name,
            "synthesis": draft,
            "anchoring_score": round(draft_score, 3),
            "ungrounded_claims": draft_ungrounded,
            "num_papers": num_papers,
        })

    return _return_with_cache({
        "theme": theme_name,
        "synthesis": revised,
        "anchoring_score": round(score, 3),
        "ungrounded_claims": ungrounded,
        "num_papers": num_papers,
    })


def _kg_str(kg: Any) -> str:
    """Extract string from KG context regardless of type."""
    if isinstance(kg, str):
        return kg
    if isinstance(kg, dict):
        return json.dumps(kg, default=str)
    return str(kg)


def _compress_entities_for_drafter(theme_entities: Dict[str, Any]) -> Dict[str, Any]:
    """Lightly compress entity JSON for the Drafter prompt — keep evidence but drop metadata.

    The Drafter already receives full evidence summaries separately, so redundant
    entity metadata (source_paper, chunk_index, cite_key) is stripped.  Evidence
    phrases are preserved because the Drafter uses them to ground specific claims.

    Returns a dict (same structure) with trimmed entities, not a plain string.
    The ``SynthesisDrafter.draft()`` method expects a dict for the entities arg.
    """
    if not theme_entities:
        return {}

    compressed: Dict[str, Any] = {}
    for key, ent_list in theme_entities.items():
        items = ent_list if isinstance(ent_list, list) else [ent_list]
        trimmed: List[Dict[str, str]] = []
        for ent in items:
            if not isinstance(ent, dict):
                continue
            keep = {}
            for field in ("entity", "name", "direction", "conditions", "context",
                          "model_system", "evidence"):
                v = ent.get(field, "")
                if v and str(v).strip():
                    keep[field] = str(v).strip()
            if keep:
                trimmed.append(keep)
        if trimmed:
            compressed[key] = trimmed[:12]  # cap per category

    return compressed


def _extract_figure_descriptions(theme_chunks: List[Dict[str, Any]]) -> List[str]:
    """Extract figure descriptions from theme_chunks for injection into evidence.

    Returns a list of formatted figure description strings.
    """
    figure_texts = []
    for ch in theme_chunks:
        meta = ch.get("metadata", {}) or {}
        if meta.get("chunk_type") != "figure":
            continue
        desc = ch.get("text", "").strip()
        if not desc:
            continue
        caption = meta.get("caption", "")
        page = meta.get("page_no", "?")
        source = meta.get("source", "?")
        if caption:
            figure_texts.append(
                f"[Figure from {source} (page {page}): {caption}] {desc}"
            )
        else:
            figure_texts.append(f"[Figure from {source} (page {page})] {desc}")
    return figure_texts


def _format_single_paper_synthesis(
    theme_name: str,
    theme_entities: Dict[str, Any],
    summary_text: str,
    citations: List[str],
) -> str:
    """Format pre-extracted entities into a structured synthesis paragraph.

    Used for single-paper themes where there is no cross-paper evidence
    to reconcile — a formatted entity listing is faster and equally
    informative as a Drafter call for one-paper coverage.
    """
    source = citations[0] if citations else "unknown"
    parts = [f"Key findings from {source} related to '{theme_name}':"]

    for category, ent_list in sorted((theme_entities or {}).items()):
        if not isinstance(ent_list, list) or not ent_list:
            continue
        item_strs: List[str] = []
        for ent in ent_list:
            if isinstance(ent, dict):
                name = ent.get("entity", str(ent))
                evidence = str(ent.get("evidence", ""))[:200]
                item_strs.append(f"{name} ({evidence})" if evidence else name)
        if item_strs:
            parts.append(f"\n{category}: {'; '.join(item_strs[:8])}")

    if len(parts) == 1:
        parts.append(f"\nSummary: {summary_text[:500]}")
    return "\n".join(parts)


def survey_per_theme_synthesize_node(
    state: AgentState,
    graph_storage: BaseGraphStorage | None = None,
) -> Dict[str, Any]:
    """Run per-theme deep synthesis in parallel across all themes.

    Themes are processed concurrently via ThreadPoolExecutor.  Each
    theme still runs its Drafter→[Critic→Arbiter] chain sequentially,
    but all themes run in parallel.  The wall-clock time is bounded
    by the slowest single theme rather than the sum of all themes.
    """
    clusters = state.get("thematic_clusters", {})
    per_paper = state.get("per_paper_extractions", {})
    all_chunks = state.get("public_context") or []
    query = state["user_query"]
    num_ctx = int(state.get("num_ctx", 16384) or 16384)
    client_kwargs = state.get("client_kwargs")

    if not clusters:
        logger.warning("No clusters — running single synthesis on all papers.")
        clusters = {"all_papers": list(per_paper.keys())}

    theme_syntheses: Dict[str, Dict[str, Any]] = {}

    def _prepare_and_run(theme_name: str, paper_ids: list,
                         drafter_model: str | None = None) -> Dict[str, Any] | None:
        """Gather chunks/entities for one theme and invoke the debate chain."""
        if not paper_ids:
            return None
        if drafter_model is None:
            drafter_model = PER_THEME_DRAFTER_MODEL

        theme_chunks = [
            ch for ch in all_chunks
            if (ch.get("metadata", {}) or {}).get("source") in paper_ids
        ]
        theme_entities: Dict[str, Any] = {}
        for pid in paper_ids:
            if pid in per_paper:
                for cat, ent_list in per_paper[pid].items():
                    key = f"{pid}::{cat}"
                    theme_entities[key] = ent_list if isinstance(ent_list, list) else []
            else:
                logger.warning("  '%s': paper '%s' not found in per_paper (keys: %s)",
                               theme_name, pid, sorted(per_paper.keys()))

        logger.info("Per-theme synthesis: '%s' (%d papers) [model=%s]",
                     theme_name, len(paper_ids), drafter_model)
        logger.debug("  '%s': %d chunks, %d entity groups",
                      theme_name, len(theme_chunks), len(theme_entities))

        import time as _time
        _t0 = _time.time()
        try:
            result = _run_debate_for_theme(
                theme_name, theme_chunks, theme_entities, query,
                num_ctx, client_kwargs,
                num_papers=len(paper_ids),
                drafter_model=drafter_model,
                graph_storage=graph_storage,
            )
            _elapsed = _time.time() - _t0
            logger.info("  '%s': score=%.2f, %d chars, %.1fs",
                         theme_name, result["anchoring_score"],
                         len(result["synthesis"]), _elapsed)
            return result
        except Exception as e:
            logger.error("  '%s': synthesis failed — %s", theme_name, e)
            return {
                "theme": theme_name,
                "synthesis": f"Synthesis failed for theme '{theme_name}': {e}",
                "anchoring_score": 0.0,
                "ungrounded_claims": [],
                "num_papers": len(paper_ids),
            }

    # ── Dual-model parallelism (Phase 5.5) ──────────────────────────────
    # Themes are split across two different models running in parallel.
    # Different models = different weights = true GPU parallelism without
    # memory multiplication.  Falls back to single-model sequential if
    # PER_THEME_MODEL_B is unset.
    theme_items = list(clusters.items())
    model_b = PER_THEME_MODEL_B.strip() if PER_THEME_MODEL_B else ""

    if model_b and len(theme_items) >= 2:
        mid = max(1, len(theme_items) // 2)
        group_a = theme_items[:mid]
        group_b = theme_items[mid:]

        def _process_group(themes: list, model: str) -> None:
            for theme_name, paper_ids in themes:
                result = _prepare_and_run(theme_name, paper_ids, drafter_model=model)
                if result is not None:
                    theme_syntheses[theme_name] = result

        logger.info("Dual-model per-theme: %d themes split %d/%d (model_a=%s, model_b=%s)",
                     len(theme_items), len(group_a), len(group_b),
                     PER_THEME_DRAFTER_MODEL, model_b)
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_a = executor.submit(_process_group, group_a, PER_THEME_DRAFTER_MODEL)
            f_b = executor.submit(_process_group, group_b, model_b)
            for f in (f_a, f_b):
                try:
                    f.result()
                except Exception as e:
                    logger.error("Dual-model group failed: %s", e)
    else:
        # Same-model parallel (Phase 6.5): when only one per‑theme model is configured,
        # use ThreadPoolExecutor to pipeline concurrent HTTP requests to Ollama.
        # Same model = same KV cache → no memory multiplication.
        # Ollama queues requests internally; pipeline overlap reduces per‑theme wall‑clock.
        import time as _time
        _parallel_t0 = _time.time()

        max_w = max(1, PER_THEME_MAX_WORKERS)
        if max_w > 1 and len(theme_items) > 1:
            logger.info("Same-model parallel per-theme: %d themes, %d workers (model=%s)",
                         len(theme_items), max_w, PER_THEME_DRAFTER_MODEL)
            futures: Dict[Any, str] = {}
            with ThreadPoolExecutor(max_workers=max_w) as executor:
                for theme_name, paper_ids in theme_items:
                    future = executor.submit(_prepare_and_run, theme_name, paper_ids)
                    futures[future] = theme_name
                for future in as_completed(futures):
                    theme_name = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            theme_syntheses[theme_name] = result
                    except Exception as e:
                        logger.error("  '%s': synthesis failed — %s", theme_name, e)
        else:
            logger.info("Single-model sequential per-theme: %d themes (model=%s)",
                         len(theme_items), PER_THEME_DRAFTER_MODEL)
            for theme_name, paper_ids in theme_items:
                result = _prepare_and_run(theme_name, paper_ids)
                if result is not None:
                    theme_syntheses[theme_name] = result

        _parallel_elapsed = _time.time() - _parallel_t0
        logger.info("Per-theme wall-clock: %.1fs (%d themes, %s)",
                     _parallel_elapsed, len(theme_items),
                     "parallel" if (max_w > 1 and len(theme_items) > 1) else "sequential")

    return {"per_theme_syntheses": theme_syntheses}


# ---------------------------------------------------------------------------
#  Helper: combine per-theme syntheses for cross-theme prompt
# ---------------------------------------------------------------------------
def _combine_syntheses(theme_syntheses: Dict[str, Dict[str, Any]]) -> str:
    """Concatenate per-theme claim syntheses with theme headers.

    The Drafter now produces dense evidence-backed claims (one per line)
    instead of verbose prose paragraphs.  This format is directly usable
    by the cross-theme synthesis model without additional compression.
    No truncation is applied — all claims are preserved.
    """
    parts: List[str] = []
    for theme_name, ts in theme_syntheses.items():
        score = ts.get("anchoring_score", 0)
        text = ts.get("synthesis", "")
        if text.startswith("Synthesis failed"):
            parts.append(f"## {theme_name} (score: {score:.2f})\n(no synthesis)")
        else:
            parts.append(f"## {theme_name} (score: {score:.2f})\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
#  Node S6: Cross-Theme Synthesis & Gap Analysis
# ---------------------------------------------------------------------------
def survey_cross_theme_synthesize_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize across all per-theme outputs + identify research gaps.

    Cross-theme synthesis and gap analysis run in parallel —
    gap analysis uses the per-theme syntheses directly rather than
    waiting for the cross-theme output.
    """
    theme_syntheses = state.get("per_theme_syntheses", {})
    query = state["user_query"]
    num_ctx = int(state.get("num_ctx", 16384) or 16384)
    client_kwargs = state.get("client_kwargs")

    if not theme_syntheses:
        return {
            "cross_theme_synthesis": "No per-theme syntheses to combine.",
            "gap_analysis": "",
        }

    combined_text = _combine_syntheses(theme_syntheses)
    citations = sorted(theme_syntheses.keys())

    # ── Level 3 cache: cross-theme synthesis ────────────────────────────
    cached_cross = load_cross_theme(query, theme_syntheses)
    if cached_cross is not None:
        return cached_cross

    # Cross-theme prompt chunks
    cross_chunks = [{"text": combined_text, "metadata": {"source": "cross_theme"}}]
    # Gap prompt chunks (uses per-theme syntheses directly — no dependency on cross-theme output)
    gap_chunks = [{
        "text": f"Individual theme syntheses:\n{combined_text}",
        "metadata": {"source": "gap_analysis"},
    }]

    cross_synthesis = ""
    gap_analysis = ""

    def _run_cross_theme() -> str:
        drafter = _get_drafter(num_ctx, client_kwargs, model=CROSS_THEME_DRAFTER_MODEL)
        return drafter.draft(
            query=f"{query}\n\nSynthesize the per-theme claims below into a unified narrative. "
                  "Identify agreements, contradictions, and gaps in the evidence. "
                  "Include inline citations. Output plain text.",
            entities={},
            chunks=cross_chunks,
            citations=citations,
            kg_context={},
        )

    def _run_gap_analysis() -> str:
        gap_drafter = _get_drafter(num_ctx, client_kwargs, model=GAP_ANALYSIS_MODEL)
        logger.info("Gap analysis using model=%s (CROSS_THEME_DRAFTER_MODEL=%s)",
                     GAP_ANALYSIS_MODEL, CROSS_THEME_DRAFTER_MODEL)
        return gap_drafter.draft(
            query=(
                "Based on the per-theme syntheses below, identify specific research gaps, "
                "unanswered questions, and areas where evidence is contradictory or missing. "
                "List each gap as a clear, actionable research question."
            ),
            entities={},
            chunks=gap_chunks,
            citations=citations,
            kg_context={},
        )

    # Run cross-theme synthesis and gap analysis in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        cross_future = executor.submit(_run_cross_theme)
        gap_future = executor.submit(_run_gap_analysis)
        try:
            cross_synthesis = cross_future.result()
        except Exception as e:
            logger.error("Cross-theme synthesis failed: %s", e)
            cross_synthesis = f"Cross-theme synthesis failed: {e}"
        try:
            gap_analysis = gap_future.result()
        except Exception as e:
            logger.error("Gap analysis failed: %s", e)
            gap_analysis = f"Gap analysis failed: {e}"

    cache_cross_theme(query, theme_syntheses, cross_synthesis, gap_analysis)

    return {
        "cross_theme_synthesis": cross_synthesis,
        "gap_analysis": gap_analysis,
    }


# ---------------------------------------------------------------------------
#  Node S7: Survey Scrub
# ---------------------------------------------------------------------------
def survey_scrub_node(state: AgentState) -> Dict[str, Any]:
    """Final ASCII scrub and output formatting for Survey Mode.

    Applies boundary scrubbing when query_scope is 'both', redacting
    sensitive content from secure corpus data before public output.
    """
    cross_synth = state.get("cross_theme_synthesis", "")
    gap = state.get("gap_analysis", "")
    theme_syntheses = state.get("per_theme_syntheses", {})
    scope = state.get("query_scope", "public")

    # Build final output
    parts = ["# SURVEY SYNTHESIS\n"]
    parts.append(final_scrub(cross_synth))

    # Phase 11: Community context (progressive disclosure tier 1)
    community_data = state.get("community_data", {})
    relevant = state.get("relevant_communities", [])
    community_summaries = state.get("community_summaries", {})
    if relevant and community_summaries:
        parts.append("\n\n# RESEARCH COMMUNITIES\n")
        parts.append(f"Query mapped to {len(relevant)} relevant research communities.\n")
        for cid in relevant:
            info = community_summaries.get(cid, community_summaries.get(str(cid), {}))
            summary = info.get("summary", "") if isinstance(info, dict) else ""
            n_entities = info.get("n_entities", "?") if isinstance(info, dict) else "?"
            if summary:
                parts.append(f"\n## Community {cid} ({n_entities} entities)\n{summary[:300]}")

    parts.append("\n\n# RESEARCH GAPS\n")
    parts.append(final_scrub(gap))
    parts.append("\n\n# PER-THEME DETAILS\n")
    for name, ts in theme_syntheses.items():
        parts.append(f"\n## {name} (score: {ts.get('anchoring_score', '?')})\n")
        parts.append(final_scrub(ts.get("synthesis", "")))

    final = "\n".join(parts)

    # Boundary scrubbing for both scope
    if scope == "both":
        scrubber = default_boundary_scrubber()
        before_len = len(final)
        final = scrubber.scrub(final)
        after_len = len(final)
        if scrubber.redaction_count > 0:
            logger.info("Survey boundary scrub: %d redactions (%d → %d chars)",
                         scrubber.redaction_count, before_len, after_len)
            try:
                audit = get_audit_logger()
                audit.log_boundary_crossing(
                    direction="secure_to_public",
                    redaction_count=scrubber.redaction_count,
                    output_chars_before=before_len,
                    output_chars_after=after_len,
                )
            except Exception:
                pass

    return {"final_output": final, "human_approved": True}
