#!/usr/bin/env python3
"""Phase 6 Streamlit UI — Production interface for the Federated RAG system.

Localhost only.  Provides project management, query execution, result review,
session history, and export.  Supports both Deep Mode and Survey Mode.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Must be before streamlit import for env
from dotenv import load_dotenv
load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Federated RAG — Biomedical Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .score-good { color: #4CAF50; font-weight: bold; }
    .score-warn { color: #FF9800; font-weight: bold; }
    .score-bad { color: #f44336; font-weight: bold; }
    .citation { color: #2196F3; font-family: monospace; }
    .entity { color: #9C27B0; font-weight: bold; }
    .log-line { font-size: 0.8em; color: #666; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ── State initialization ───────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history: List[Dict[str, Any]] = []
if "current_result" not in st.session_state:
    st.session_state.current_result: Optional[Dict[str, Any]] = None
if "logs" not in st.session_state:
    st.session_state.logs: List[str] = []


def _add_log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-100:]


# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "projects/default"))
SURVEY_RESULT_PATH = PROJECT_DIR / "survey_result.json"
BENCHMARK_OUTPUT_PATH = PROJECT_DIR / "benchmark_scorecard.json"

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Federated RAG")
    st.caption("Biomedical Research Assistant")

    st.divider()

    mode = st.radio(
        "Mode",
        ["Survey", "Deep", "Quick", "Sectioned"],
        index=0,
        help="Survey: multi-theme synthesis | Deep: full debate | Quick: fast lookup | Sectioned: IMRaD section writing",
    )

    scope = st.radio(
        "Scope",
        ["public", "secure", "both"],
        index=0,
        help="public: open literature | secure: internal data | both: cross-corpus",
    )

    st.divider()

    num_ctx = st.number_input("Context Window", value=16384, step=4096,
                               help="LLM context window size in tokens")
    max_tokens = st.number_input("Max Output Tokens", value=4096, step=512,
                                  help="Max tokens per LLM response")

    st.divider()

    st.caption(f"Project: {PROJECT_DIR.name}")
    st.caption(f"Provider: {os.getenv('LLM_PROVIDER', 'ollama')}")
    st.caption(f"Fast tier: {os.getenv('OLLAMA_SMALL_MODEL', 'gemma4:e4b')}")
    st.caption(f"Reasoning tier: {os.getenv('OLLAMA_LARGE_MODEL', 'qwen3.6:35b')}")

# ── Main area ───────────────────────────────────────────────────────────────
st.title("Federated RAG — Biomedical Literature Synthesis")

tab_query, tab_results, tab_benchmarks, tab_history, tab_logs = st.tabs(
    ["Query", "Results", "Benchmarks", "History", "Logs"]
)

# ── Tab 1: Query ───────────────────────────────────────────────────────────
with tab_query:
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_area(
            "Research Question",
            placeholder="e.g., How do T cells and macrophages coordinate bone healing around titanium implants?",
            height=100,
            key="query_input",
        )
    with col2:
        st.caption("Quick queries:")
        quick_queries = [
            "Map immune response to titanium implants",
            "How does obesity affect biomaterial integration?",
            "Cytokine profiles in macrophage polarization",
            "T cell subsets in bone healing",
            "Surface modifications for osseointegration",
        ]
        for q in quick_queries:
            if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
                st.session_state.query_input = q
                st.rerun()

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        run_btn = st.button("🔍 Run Query", type="primary", use_container_width=True)
    with col2:
        if st.button("📊 Benchmark", use_container_width=True):
            _add_log("Running Tier A benchmark...")
            try:
                from phase5_benchmark import load_survey_result, compute_all_metrics
                result = load_survey_result()
                if result:
                    metrics = compute_all_metrics(
                        result.get("per_theme_syntheses", {}),
                        result.get("cross_theme_synthesis", ""),
                        result.get("gap_analysis", ""),
                    )
                    st.session_state.current_result = {"type": "benchmark", "metrics": metrics}
                    _add_log("Benchmark complete")
                else:
                    st.warning("No cached results. Run a query first.")
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

    if run_btn and query:
        _add_log(f"Running {mode} mode query: {query[:80]}...")
        with st.spinner(f"Running {mode} mode synthesis..."):
            try:
                # Set env for this run
                os.environ["LLM_MAX_TOKENS"] = str(max_tokens)

                # Ingest/check PDFs
                from src.ingestion.pdf_parser import PDFParser
                from src.retrieval.chroma_client import ChromaClient
                from src.retrieval.bm25_index import BM25Index
                from src.retrieval.hybrid_retriever import HybridRetriever
                from src.graph.networkx_json_storage import NetworkXJSONStorage
                from src.unicode_map import scrub_unicode

                CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
                GRAPH_PATH = str(PROJECT_DIR / "project_graph.json")

                chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
                bm25 = BM25Index()
                # Rebuild BM25 index from ChromaDB data
                all_docs = chroma.collection.get(include=["documents", "metadatas"])
                if all_docs.get("documents"):
                    bm25.add_documents([d for d, m in zip(all_docs["documents"],
                                            all_docs.get("metadatas") or [])
                                        if (m or {}).get("chunk_type") != "reference"])
                hybrid = HybridRetriever(chroma_client=chroma, bm25_index=bm25)
                graph_storage = NetworkXJSONStorage(file_path=GRAPH_PATH)

                # Check if we need to ingest
                data_dir = Path("data")
                if data_dir.exists():
                    parser = PDFParser()
                    new_count = 0
                    for pdf_path in sorted(data_dir.glob("*.pdf")):
                        try:
                            existing = chroma.collection.get(
                                where={"source": pdf_path.name}, limit=1
                            )
                            if existing and existing.get("ids"):
                                continue
                            chunks = parser.parse(pdf_path)
                            if chunks:
                                hybrid.ingest(chunks)
                                new_count += 1
                                _add_log(f"Ingested {pdf_path.name}: {len(chunks)} chunks")
                        except Exception as e:
                            _add_log(f"Skip {pdf_path.name}: {e}")
                    if new_count:
                        st.success(f"Ingested {new_count} new PDF(s)")

                    # Phase 7a: Vision pipeline — extract, describe, embed figures
                    _add_log("Running vision pipeline on ingested PDFs...")
                    vision_enabled = os.getenv("VISION_MODEL", "gemma4:e4b").lower() not in ("0", "none", "false")
                    for pdf_path in sorted(data_dir.glob("*.pdf")):
                        try:
                            from src.vision.vision_ingest import vision_ingest_pdf
                            vresult = vision_ingest_pdf(pdf_path, hybrid, describe=vision_enabled)
                            if vresult.get("embedded", 0) > 0:
                                _add_log(
                                    f"Vision: {pdf_path.name} → {vresult['kept']}/{vresult['extracted']} figures kept, "
                                    f"{vresult['embedded']} embedded"
                                )
                        except Exception as e:
                            _add_log(f"Vision skip {pdf_path.name}: {e}")

                # Build and run graph
                from src.graph.graph_builder import build_survey_graph, build_graph
                from langgraph.errors import GraphRecursionError

                if mode == "Survey":
                    graph = build_survey_graph(hybrid, graph_storage)
                elif mode == "Sectioned":
                    from src.graph.sectioned_survey_graph import build_sectioned_survey_graph
                    graph = build_sectioned_survey_graph(hybrid)
                else:
                    graph = build_graph(hybrid, graph_storage)

                initial_state = {
                    "user_query": query,
                    "query_scope": scope,
                    "mode": mode.lower(),
                    "num_ctx": num_ctx,
                    "client_kwargs": {"timeout": int(os.getenv("LLM_TIMEOUT", "900"))},
                    "public_context": [],
                    "secure_context": [],
                }

                _add_log(f"Invoking LangGraph ({mode} mode)...")
                t0 = time.time()
                config = {"configurable": {"thread_id": f"ui-{int(t0)}"}}

                try:
                    result = graph.invoke(initial_state, config)
                except GraphRecursionError:
                    result = graph.invoke(initial_state, {"recursion_limit": 200})

                elapsed = time.time() - t0
                _add_log(f"Completed in {elapsed:.0f}s")

                # Store result
                final_output = str(result.get("final_output", "") or result.get("cross_theme_synthesis", ""))
                section_drafts = result.get("section_drafts", {})
                section_plan = result.get("section_plan", [])

                st.session_state.current_result = {
                    "type": "query",
                    "mode": mode,
                    "query": query,
                    "scope": scope,
                    "elapsed": round(elapsed, 1),
                    "final_output": final_output,
                    "cross_theme_synthesis": result.get("cross_theme_synthesis", ""),
                    "gap_analysis": result.get("gap_analysis", ""),
                    "per_theme_syntheses": result.get("per_theme_syntheses", {}),
                    "decomposed_themes": result.get("decomposed_themes", []),
                    "thematic_clusters": result.get("thematic_clusters", {}),
                    "anchoring_score": result.get("anchoring_score", 0),
                    "section_drafts": section_drafts,
                    "section_plan": section_plan,
                }

                # Add to history
                st.session_state.query_history.insert(0, {
                    "query": query,
                    "mode": mode,
                    "scope": scope,
                    "elapsed": round(elapsed, 1),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                # Save survey result
                if mode == "Survey":
                    SURVEY_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    save_data = {
                        "decomposed_themes": result.get("decomposed_themes", []),
                        "thematic_clusters": result.get("thematic_clusters", {}),
                        "per_theme_syntheses": result.get("per_theme_syntheses", {}),
                        "cross_theme_synthesis": result.get("cross_theme_synthesis", ""),
                        "gap_analysis": result.get("gap_analysis", ""),
                    }
                    SURVEY_RESULT_PATH.write_text(
                        json.dumps(save_data, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )

                st.success(f"Query completed in {elapsed:.0f}s")
                st.rerun()

            except Exception as e:
                _add_log(f"ERROR: {e}")
                st.error(f"Query failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── Tab 2: Results ─────────────────────────────────────────────────────────
with tab_results:
    result = st.session_state.current_result

    if result is None:
        st.info("Run a query to see results here.")
        # Show cached survey if available
        if SURVEY_RESULT_PATH.exists():
            with st.expander("📋 Cached Survey Result", expanded=False):
                try:
                    cached = json.loads(SURVEY_RESULT_PATH.read_text(encoding="utf-8"))
                    themes = cached.get("per_theme_syntheses", {})
                    for name, ts in themes.items():
                        score = ts.get("anchoring_score", 0)
                        color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                        st.markdown(f"**{name}** :{color}[score: {score:.2f}]")
                        with st.expander(f"  Show synthesis ({len(ts.get('synthesis',''))} chars)"):
                            st.text(ts.get("synthesis", "")[:2000])
                except Exception:
                    st.caption("Could not load cached result.")
    else:
        if result["type"] == "benchmark":
            st.subheader("📊 Benchmark Scorecard")
            metrics = result.get("metrics", {})
            overall = metrics.get("_overall", {})

            col1, col2 = st.columns(2)
            with col1:
                grade = overall.get("grade", "?")
                color = "green" if grade == "PASS" else "orange" if grade == "WARN" else "red"
                st.markdown(f"### Overall: :{color}[{grade}]")
                st.caption(f"Pass: {overall.get('pass_count',0)} | "
                           f"Warn: {overall.get('warn_count',0)} | "
                           f"Fail: {overall.get('fail_count',0)}")

            for name, m in metrics.items():
                if name.startswith("_"):
                    continue
                g = m.get("grade", "?")
                color = "green" if g == "PASS" else "orange" if g == "WARN" else "red"
                with st.expander(f":{color}[{g}] {name}"):
                    for k, v in m.items():
                        if k != "grade":
                            st.caption(f"{k}: {v}")

        elif result["type"] == "query":
            st.subheader(f"Results — {result['mode']} mode ({result['elapsed']:.0f}s)")
            st.caption(f"Query: {result['query']}")

            # Tabs for different result sections
            if result.get("section_drafts"):
                # Sectioned survey display
                rt1, rt2 = st.tabs(["Manuscript", "Sections"])
                with rt1:
                    final = result.get("final_output", "")
                    if final:
                        st.markdown(final)
                    else:
                        # Build from section drafts
                        for sec in result.get("section_plan", []):
                            name = sec.get("name", "?")
                            draft = result["section_drafts"].get(name, "")
                            if draft:
                                st.markdown(f"### {name.upper()}")
                                st.text(draft[:3000])
                with rt2:
                    for sec in result.get("section_plan", []):
                        name = sec.get("name", "?")
                        draft = result["section_drafts"].get(name, "")
                        if draft:
                            claims = [l for l in draft.split("\n") if l.strip()]
                            st.markdown(f"**{name}**: {len(claims)} claims, {len(draft)} chars")
                rt3, rt4 = st.tabs(["Gap Analysis", "Export"])
            else:
                rt1, rt2, rt3, rt4 = st.tabs(["Synthesis", "Per-Theme", "Gap Analysis", "Export"])

            with rt1:
                final = result.get("final_output", "")
                if final:
                    st.markdown(final)
                else:
                    cross = result.get("cross_theme_synthesis", "")
                    if cross:
                        st.markdown(cross)
                    else:
                        st.info("No synthesis output.")

            with rt2:
                themes = result.get("per_theme_syntheses", {})
                if themes:
                    for name, ts in themes.items():
                        score = ts.get("anchoring_score", 0)
                        color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                        st.markdown(f"### {name} :{color}[{score:.2f}]")
                        st.text(ts.get("synthesis", "")[:3000])
                        if len(ts.get("synthesis", "")) > 3000:
                            st.caption("(truncated)")
                        ungrounded = ts.get("ungrounded_claims", [])
                        if ungrounded:
                            with st.expander(f"{len(ungrounded)} ungrounded claims"):
                                for ug in ungrounded:
                                    st.caption(f"- {ug.get('claim', '?')[:200]}")
                else:
                    st.info("No per-theme results.")

            with rt3:
                gap = result.get("gap_analysis", "")
                if gap:
                    st.markdown(gap)
                else:
                    st.info("No gap analysis.")

            with rt4:
                st.subheader("Export")
                export_format = st.selectbox("Format", ["Markdown", "Plain Text", "JSON"])

                def _get_export_text():
                    final = str(result.get("final_output", "") or result.get("cross_theme_synthesis", ""))
                    gap = str(result.get("gap_analysis", ""))
                    if export_format == "JSON":
                        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
                    elif export_format == "Markdown":
                        parts = ["# Synthesis\n", final, "\n\n## Research Gaps\n", gap]
                        themes = result.get("per_theme_syntheses", {})
                        if themes:
                            parts.append("\n\n## Per-Theme Details\n")
                            for name, ts in themes.items():
                                parts.append(f"\n### {name} (score: {ts.get('anchoring_score','?')})\n")
                                parts.append(ts.get("synthesis", ""))
                        return "\n".join(parts)
                    else:
                        return final

                st.download_button(
                    "Download",
                    data=_get_export_text(),
                    file_name=f"synthesis_{int(time.time())}.{'md' if export_format == 'Markdown' else 'txt' if export_format == 'Plain Text' else 'json'}",
                    mime="text/plain",
                )

# ── Tab 3: Benchmarks ──────────────────────────────────────────────────────
with tab_benchmarks:
    st.subheader("📊 Benchmark Dashboard")

    if st.button("Run Tier A Benchmark", type="secondary"):
        _add_log("Running Tier A benchmark...")
        try:
            from phase5_benchmark import load_survey_result, compute_all_metrics, print_report
            result = load_survey_result()
            if result:
                metrics = compute_all_metrics(
                    result.get("per_theme_syntheses", {}),
                    result.get("cross_theme_synthesis", ""),
                    result.get("gap_analysis", ""),
                )
                overall = metrics.get("_overall", {})

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    a = metrics.get("anchoring_distribution", {})
                    st.metric("Mean Anchor", f"{a.get('mean',0):.3f}",
                              delta=f"Min: {a.get('min',0):.3f}")
                with col2:
                    d = metrics.get("claim_density", {})
                    st.metric("Total Claims", d.get("total_claims", 0))
                with col3:
                    e = metrics.get("entity_appearance", {})
                    st.metric("Entity Rate", f"{e.get('rate',0):.0%}")
                with col4:
                    grade = overall.get("grade", "?")
                    color = "green" if grade == "PASS" else "orange" if grade == "WARN" else "red"
                    st.metric("Overall", f":{color}[{grade}]")

                st.divider()

                for name, m in metrics.items():
                    if name.startswith("_"):
                        continue
                    g = m.get("grade", "?")
                    color = "green" if g == "PASS" else "orange" if g == "WARN" else "red"
                    with st.expander(f":{color}[{g}] {name}"):
                        for k, v in sorted(m.items()):
                            if k != "grade":
                                st.caption(f"{k}: {v}")

                # Save
                BENCHMARK_OUTPUT_PATH.write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                st.success(f"Scorecard saved to {BENCHMARK_OUTPUT_PATH}")
            else:
                st.warning("No cached survey result. Run a Survey Mode query first.")
        except Exception as e:
            st.error(f"Benchmark failed: {e}")

# ── Tab 4: History ─────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Session History")

    if not st.session_state.query_history:
        st.info("No queries run this session.")
    else:
        for i, entry in enumerate(st.session_state.query_history):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{entry['query'][:100]}**")
                st.caption(f"{entry['mode']} · {entry['scope']} · {entry['elapsed']:.0f}s · "
                           f"{entry.get('timestamp', '')[:16]}")
            with col2:
                if st.button("Re-run", key=f"rerun_{i}"):
                    st.session_state.query_input = entry["query"]
                    st.rerun()
            st.divider()

    if st.button("Clear History"):
        st.session_state.query_history = []
        st.rerun()

# ── Tab 5: Logs ────────────────────────────────────────────────────────────
with tab_logs:
    st.subheader("System Log")
    st.caption("Last 100 log entries")

    log_text = "\n".join(st.session_state.logs[-100:])
    st.code(log_text if log_text else "(no logs)", language="text")

    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()
