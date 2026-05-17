"""
Orchestrator — Phase 10 background daemon loop.
Autonomously runs: web discovery → EPMC search → ingest → extract → KG → handoff.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("projects/default")
GRAPH_PATH = PROJECT_DIR / "project_graph.json"
STATE_PATH = PROJECT_DIR / "orchestrator_state.json"
PID_PATH = PROJECT_DIR / "orchestrator.pid"
LOG_PATH = PROJECT_DIR / "orchestrator.log"
YIELD_PATH = PROJECT_DIR / "daemon_yield"
DEFAULT_INTERVAL_MINUTES = 60
DEFAULT_MAX_PAPERS_PER_QUERY = 5
DEFAULT_SEED_TERMS = [
    "biomaterial surface modification immune response",
    "titanium implant osseointegration",
    "macrophage polarization biomaterials",
]

_FILE_HANDLER_SETUP = False


def _ensure_file_logging() -> None:
    """Add a RotatingFileHandler to the orchestrator logger (Phase 10 Gap C).

    Called once per process lifetime.  Writes to
    ``projects/default/orchestrator.log`` with 5 backups × 5 MB each.
    """
    global _FILE_HANDLER_SETUP
    if _FILE_HANDLER_SETUP:
        return
    _FILE_HANDLER_SETUP = True

    try:
        from logging.handlers import RotatingFileHandler
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            str(LOG_PATH),
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.info("File logging initialized: %s", LOG_PATH)
    except Exception:
        pass  # don't crash if we can't set up logging


class Orchestrator:
    """Autonomous background daemon that discovers and ingests new research.

    Usage::

        from src.graph import create_graph_storage
        gs = create_graph_storage(file_path=GRAPH_PATH)

        # Dry run — see what WOULD happen, no API calls beyond web discovery
        orch = Orchestrator(graph_storage=gs, dry_run=True)
        summary = orch.run_once()

        # Live run — actually ingests papers
        orch = Orchestrator(graph_storage=gs, interval_minutes=60)
        orch.run_once()        # single cycle, blocking
        # orch.start()         # daemon loop, non-blocking

    Pipeline per cycle:
        1. web discovery → topic list
        2. EPMC search → full-text XML → parse → chunks  (skipped if dry_run)
        3. ChromaDB + BM25 ingest (deduped)               (skipped if dry_run)
        4. PreExtractor → KG update                       (skipped if dry_run)
        5. graph_storage.save() + handoff write
    """

    def __init__(
        self,
        *,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        graph_storage: Any = None,
        max_papers_per_query: int = DEFAULT_MAX_PAPERS_PER_QUERY,
        seed_terms: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> None:
        self.interval_minutes = interval_minutes
        self.graph_storage = graph_storage
        self.max_papers_per_query = max_papers_per_query
        self.dry_run = dry_run
        self._seed_terms = seed_terms or list(DEFAULT_SEED_TERMS)
        self._cycle = 0
        self._total_ingested = 0
        self._scheduler = None

        # Phase 10 Gap A: resume from state file on restart
        self._load_state()

    def _load_state(self) -> None:
        """Restore cycle counter and ingestion total from state file.

        If the daemon crashed or was killed, restoring state prevents
        resetting to cycle 0 on restart.  IngestProgress checkpoints
        still ensure idempotency — re-ingesting the same paper is a no-op.
        """
        if not STATE_PATH.exists():
            return
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            saved_cycle = data.get("last_cycle", 0)
            saved_total = data.get("total_ingested", 0)
            if isinstance(saved_cycle, int) and saved_cycle > 0:
                self._cycle = saved_cycle
                logger.info("Resumed daemon state: cycle=%d, total_ingested=%d",
                             self._cycle, saved_total)
            if isinstance(saved_total, int):
                self._total_ingested = saved_total
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("State file unreadable, starting fresh: %s", exc)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run_once(self) -> Dict[str, Any]:
        """Execute one full pipeline cycle (blocking).

        Returns a summary dict suitable for logging and handoff generation.
        """
        logger.info("=== Orchestrator cycle %d starting ===", self._cycle + 1)
        return self._run_cycle()

    def start(self, cooldown_seconds: int = 60) -> None:
        """Start the daemon loop (non-blocking).

        Runs ``run_once()`` in a continuous chain: execute → *cooldown_seconds*
        → execute → ...  A long-running cycle doesn't delay the next start —
        the cooldown begins only after the cycle finishes.

        Call ``stop()`` to terminate.
        """
        from src.agents.scheduler import Scheduler

        if self._scheduler is not None:
            logger.warning("Orchestrator already started")
            return

        self._write_pid()
        self._write_state("started")
        _ensure_file_logging()
        self._scheduler = Scheduler()
        self._scheduler.schedule(
            self.run_once, cooldown_seconds=cooldown_seconds,
        )
        logger.info("Orchestrator daemon started (pid=%d, cooldown=%ds)",
                     os.getpid(), cooldown_seconds)

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the daemon loop and wait for the current cycle to finish."""
        if self._scheduler is not None:
            self._scheduler.stop(timeout=timeout)
            self._scheduler = None
        self._write_state("stopped")
        self._remove_pid()
        logger.info("Orchestrator daemon stopped")

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.is_running

    # ------------------------------------------------------------------
    #  Core cycle
    # ------------------------------------------------------------------

    def _run_cycle(self) -> Dict[str, Any]:
        self._cycle += 1
        summary: Dict[str, Any] = {
            "cycle": self._cycle,
            "mode": "dry_run" if self.dry_run else "live",
            "discovered_topics": 0,
            "epmc_queries_run": 0,
            "papers_fetched": 0,
            "papers_ingested": 0,
            "would_have_queries": [],
            "errors": [],
        }

        # ── Bootstrap: take ownership of Ollama (disarm launchd) ────────
        if not self.dry_run:
            try:
                from src.ingestion.pre_extractor import PreExtractor
                PreExtractor._ensure_dedicated_ollama()
            except Exception:
                pass

        # 1 ─ Web discovery
        seed_terms = self._derive_seed_terms()
        discovered = self._discover_topics(seed_terms)
        summary["discovered_topics"] = len(discovered)

        # 2 ─ Build search queries from discovery results
        queries = self._queries_from_discovery(discovered, seed_terms)
        summary["epmc_queries_run"] = len(queries)

        if self.dry_run:
            summary["would_have_queries"] = queries
            if queries:
                logger.info(
                    "DRY RUN: would search EPMC with %d queries: %s",
                    len(queries),
                    ", ".join(q[:60] for q in queries[:3]),
                )
            else:
                logger.info("DRY RUN: no queries generated (nothing to search)")
        else:
            # 3 ─ Search, fetch, parse, ingest, extract (parallel fetch, batch ingest)
            if queries:
                ingested = self._search_and_ingest(queries)
                summary["papers_fetched"] = sum(
                    r.get("fetched", 0) for r in ingested.values()
                )
                summary["papers_ingested"] = sum(
                    r.get("ingested", 0) for r in ingested.values()
                )

        # 4 ─ Persist graph (safe even with dry_run — just saves current state)
        if self.graph_storage is not None:
            try:
                self.graph_storage.save()
                logger.info("Knowledge graph saved.")
            except Exception as exc:
                logger.error("KG save failed: %s", exc)
                summary["errors"].append(f"kg_save: {exc}")

            # 4.5 ─ Phase 11: Update community detection (offline)
            try:
                community_result = self._update_communities()
                summary["communities"] = community_result
            except Exception as exc:
                logger.warning("Community detection update skipped: %s", exc)

        # 5 ─ Write handoff (cycle-specific file, never overwrites human HANDOFF.md)
        self._write_handoff(summary)

        # 5.5 ─ Phase 10 Gap B: rotate old handoff files
        self._cleanup_handoffs()

        self._total_ingested += summary["papers_ingested"]

        # 6 ─ Persist daemon state for crash recovery
        self._write_state("cycle_complete", last_error=summary["errors"][0] if summary["errors"] else None)
        logger.info(
            "=== Orchestrator cycle %d complete [%s]: %d discovered, "
            "%d fetched, %d ingested (total: %d) ===",
            self._cycle,
            "DRY RUN" if self.dry_run else "LIVE",
            summary["discovered_topics"],
            summary["papers_fetched"],
            summary["papers_ingested"],
            self._total_ingested,
        )
        return summary

    # ------------------------------------------------------------------
    #  Step 1: Web discovery
    # ------------------------------------------------------------------

    def _derive_seed_terms(self) -> List[str]:
        """Collect seed terms from KG top entities + static defaults."""
        terms = list(self._seed_terms)
        if self.graph_storage is not None:
            try:
                top = self._top_kg_entities(max_terms=4)
                for t in top:
                    if t not in terms:
                        terms.append(t)
            except Exception:
                pass
        return terms[:8]

    def _top_kg_entities(self, max_terms: int = 4) -> List[str]:
        """Extract entity names with the highest edge degree from the KG."""
        gs = self.graph_storage
        if gs is None:
            return []
        try:
            degrees: Dict[str, int] = {}
            graph = getattr(gs, "_graph", None)
            if graph is None:
                return []
            for node in graph.nodes:
                deg = graph.degree(node)
                degrees[node] = deg
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            # Extract the entity name after the category prefix (e.g. "material:Ti-6Al-4V")
            names = []
            for n in sorted_nodes:
                if ":" in n:
                    _, name = n.split(":", 1)
                else:
                    name = n
                if name and name not in names:
                    names.append(name)
                    if len(names) >= max_terms:
                        break
            return names
        except Exception:
            return []

    def _discover_topics(self, seed_terms: List[str]) -> List[Dict[str, Any]]:
        """Run web discovery across seed terms. Returns list of discovery dicts."""
        try:
            from src.retrieval.web_search import WebSearchClient

            wsc = WebSearchClient()
            results = wsc.discover_topics(seed_terms, results_per_term=3)
            logger.info("Web discovery: %d results from %d seed terms",
                        len(results), len(seed_terms))
            return results
        except Exception as exc:
            logger.warning("Web discovery failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    #  Step 2-3: Query extraction + EPMC search + ingest
    # ------------------------------------------------------------------

    @staticmethod
    def _queries_from_discovery(
        discovered: List[Dict[str, Any]], seed_terms: List[str]
    ) -> List[str]:
        """Extract unique search queries from discovery results.

        Uses discovery snippets/titles; falls back to seed terms.
        Always returns a deduplicated list.
        """
        queries: List[str] = []
        seen: set = set()
        for d in discovered:
            title = str(d.get("title", "")).strip()
            snippet = str(d.get("snippet", "")).strip()
            candidate = title if len(title) >= 20 else snippet
            if not candidate or len(candidate) < 20:
                continue
            candidate = candidate[:200]
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                queries.append(candidate)
        # Supplement with seed terms if discovery returned little
        if len(queries) < 2:
            for term in seed_terms:
                key = term.lower()
                if key not in seen:
                    seen.add(key)
                    queries.append(term)
                    if len(queries) >= 3:
                        break
        return queries[:6]

    def _search_and_ingest(self, queries: List[str]) -> Dict[str, Dict[str, int]]:
        """Parallel EPMC search + XML fetch → batch ingest → sequential extraction.

        Phase 1: ``run_parallel`` fetches and parses all queries concurrently
        (I/O-bound — EPMC HTTP + OAI fallback).  Phase 2: all chunks are
        batched into a single ChromaDB + BM25 ingest (avoids redundant BM25
        rebuilds).  Phase 3: PreExtractor runs sequentially per paper
        (Ollama/LLM is the bottleneck — parallelising here does not help).

        Returns a dict mapping query → {"fetched": N, "ingested": N}.
        """
        from src.agents.subagents import run_parallel
        from src.utils.ingest_progress import IngestProgress

        progress = IngestProgress()
        completed_set = progress.get_completed()

        results = run_parallel(
            _fetch_and_parse_for_query,
            queries,
            max_workers=min(4, len(queries)),
            max_papers=self.max_papers_per_query,
            completed_pmcids=completed_set,
        )

        # Collect all parsed chunks + paper metadata from successful results
        all_paper_data: List[Dict[str, Any]] = []
        per_query: Dict[str, Dict[str, int]] = {}

        for r in results:
            if r["error"] is not None:
                logger.warning("Parallel fetch failed for %r: %s", r["item"], r["error"])
                per_query[r["item"]] = {"fetched": 0, "ingested": 0, "parse_error": True}
                continue

            data = r["result"]
            query = data["query"]
            per_query[query] = {"fetched": len(data["paper_data"]), "ingested": 0}
            for pd in data["paper_data"]:
                all_paper_data.append(pd)

        if not all_paper_data:
            return per_query

        # Batch ingest — one ChromaDB + BM25 call for all new chunks
        batch_chunks: List[Dict[str, Any]] = []
        for pd in all_paper_data:
            batch_chunks.extend(pd["chunks"])

        try:
            self._ingest_chunks_batch(batch_chunks)
        except Exception as exc:
            logger.error("Batch ingest failed: %s", exc)
            return per_query

        # Mark all as ingested and run extraction sequentially per paper
        for pd in all_paper_data:
            pmcid = pd["pmcid"]
            progress.checkpoint(pmcid)
            # Find which query this paper came from and increment its counter
            for query, stats in per_query.items():
                if query == pd.get("query"):
                    stats["ingested"] += 1
                    break

            if self.graph_storage is not None:
                self._extract_to_kg(pmcid, pd["chunks"])

            # Check for yield request from UI between papers
            self._check_yield()

        progress.finalize()
        return per_query

    def _ingest_chunks(self, pmcid: str, chunks: List[Dict[str, Any]]) -> None:
        """Ingest parsed chunks into ChromaDB + BM25 (deduped)."""
        try:
            from src.retrieval.chroma_client import ChromaClient
            from src.retrieval.bm25_index import BM25Index
            from src.retrieval.hybrid_retriever import HybridRetriever

            chroma = ChromaClient(
                collection_name="public_corpus",
                persist_directory=str(PROJECT_DIR / "chroma_data"),
            )
            bm25 = BM25Index(persist_dir=PROJECT_DIR / "bm25_index")
            bm25.load()
            retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)
            retriever.ingest(chunks)
            bm25.save()
            logger.info("Ingested %d chunks from %s", len(chunks), pmcid)
        except Exception as exc:
            logger.error("Ingest failed for %s: %s", pmcid, exc)

    def _ingest_chunks_batch(self, chunks: List[Dict[str, Any]]) -> None:
        """Ingest a batch of chunks from multiple papers in a single BM25 rebuild.

        Avoids redundant corpus tokenisation when many papers arrive in one cycle.
        """
        if not chunks:
            return
        try:
            from src.retrieval.chroma_client import ChromaClient
            from src.retrieval.bm25_index import BM25Index
            from src.retrieval.hybrid_retriever import HybridRetriever

            chroma = ChromaClient(
                collection_name="public_corpus",
                persist_directory=str(PROJECT_DIR / "chroma_data"),
            )
            bm25 = BM25Index(persist_dir=PROJECT_DIR / "bm25_index")
            bm25.load()
            retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)
            retriever.ingest(chunks)
            bm25.save()
            logger.info("Batch ingested %d chunks from %d papers",
                        len(chunks),
                        len({c["metadata"].get("pmcid", "") for c in chunks if c.get("metadata")}))
        except Exception as exc:
            logger.error("Batch ingest failed: %s", exc)
            raise

    def _extract_to_kg(
        self, pmcid: str, chunks: List[Dict[str, Any]],
    ) -> None:
        """Run PreExtractor to update the knowledge graph."""
        try:
            from src.ingestion.pre_extractor import PreExtractor

            pre = PreExtractor()
            pre.extract_paper(
                paper_id=pmcid, chunks=chunks,
                graph_storage=self.graph_storage,
            )
        except Exception as exc:
            logger.debug("PreExtractor failed for %s: %s", pmcid, exc)

    # ------------------------------------------------------------------
    #  Step 4.5: Phase 11 Community Detection
    # ------------------------------------------------------------------

    def _update_communities(self) -> Dict[str, Any]:
        """Run community detection on the KG and cache results.

        Runs after every cycle so communities stay current as new entities
        are added to the graph.  Uses Louvain algorithm via NetworkX.
        """
        try:
            from src.graph.community_detection import detect_communities
            result = detect_communities(
                self.graph_storage,
                force_recompute=True,
            )
            n_comm = result.get("n_communities", 0)
            mod = result.get("modularity", 0.0)
            logger.info(
                "Phase 11 community detection: %d communities (modularity=%.3f, %d nodes)",
                n_comm, mod, result.get("n_nodes", 0),
            )
            return {
                "n_communities": n_comm,
                "modularity": mod,
                "n_nodes": result.get("n_nodes", 0),
            }
        except Exception as exc:
            logger.warning("Community detection failed (non-fatal): %s", exc)
            return {"n_communities": 0, "modularity": 0.0, "error": str(exc)}

    # ------------------------------------------------------------------
    #  Step 5: Handoff
    # ------------------------------------------------------------------

    def _write_handoff(self, summary: Dict[str, Any]) -> None:
        """Write a cycle-specific handoff file.

        Writes to ``projects/default/cycle_N_handoff.md`` so the human-maintained
        ``HANDOFF.md`` is never overwritten.
        """
        try:
            from src.agents.handoff import write_handoff

            cycle = summary.get("cycle", self._cycle)
            output_path = PROJECT_DIR / f"cycle_{cycle}_handoff.md"
            write_handoff(
                graph_storage=self.graph_storage,
                orchestrator_summary=summary,
                output_path=output_path,
            )
        except Exception as exc:
            logger.warning("Handoff write failed: %s", exc)

    def _check_yield(self, timeout: float = 600.0) -> None:
        """Pause extraction if a user query is waiting for the GPU.

        Checks for the presence of ``projects/default/daemon_yield`` — a
        sentinel file created by the Streamlit UI or manually by the user.
        When found, the daemon unloads gemma4 (freeing GPU memory for qwen),
        yields, and polls until the file is removed.

        This allows user queries to complete during daemon extraction without
        killing the daemon or exceeding the 36 GB unified-memory budget.
        Only one model can be loaded at a time on this hardware.
        """
        if not YIELD_PATH.exists():
            return

        logger.info(
            "Daemon yield — sentinel file detected at %s. "
            "Unloading gemma4 to free GPU for user query.",
            YIELD_PATH,
        )
        self._write_state("yielding")

        try:
            from src.ingestion.pre_extractor import PreExtractor
            PreExtractor._reset_ollama(timeout=10.0)
        except Exception:
            pass

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not YIELD_PATH.exists():
                elapsed = time.monotonic() - t0
                logger.info(
                    "Daemon yield complete after %.1fs — sentinel removed. "
                    "Resuming extraction.",
                    elapsed,
                )
                self._write_state("extracting")
                return
            time.sleep(1)

        logger.warning(
            "Daemon yield timed out after %.0fs — sentinel still present. "
            "Resuming extraction anyway.",
            timeout,
        )
        self._write_state("extracting")

    def _cleanup_handoffs(self, max_age_days: int = 7) -> None:
        """Remove handoff files older than *max_age_days*.

        Prevents unbounded accumulation of ``cycle_N_handoff.md`` files
        in the project directory.
        """
        try:
            now = datetime.now(timezone.utc)
            for path in PROJECT_DIR.glob("cycle_*_handoff.md"):
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    age_days = (now - mtime).days
                    if age_days > max_age_days:
                        path.unlink()
                        logger.info("Cleaned up old handoff: %s (%d days old)", path.name, age_days)
                except OSError:
                    pass
        except Exception as exc:
            logger.debug("Handoff cleanup failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    #  State file + PID management
    # ------------------------------------------------------------------

    def _write_state(self, status: str, last_error: Optional[str] = None) -> None:
        """Persist orchestrator state for crash recovery and monitoring."""
        try:
            state = {
                "pid": os.getpid(),
                "status": status,
                "last_cycle": self._cycle,
                "total_ingested": self._total_ingested,
                "dry_run": self.dry_run,
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                "last_error": last_error,
            }
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("State file write failed: %s", exc)

    def _write_pid(self) -> None:
        """Write PID file for external daemon management (e.g. ``kill $(cat orchestrator.pid)``)."""
        try:
            PID_PATH.parent.mkdir(parents=True, exist_ok=True)
            PID_PATH.write_text(str(os.getpid()))
        except Exception as exc:
            logger.debug("PID file write failed: %s", exc)

    def _remove_pid(self) -> None:
        """Remove PID file on clean shutdown."""
        try:
            if PID_PATH.exists():
                PID_PATH.unlink()
        except Exception as exc:
            logger.debug("PID file removal failed: %s", exc)

    # ------------------------------------------------------------------
    #  Convenience: gap-driven resolution (reuses GapResolver)
    # ------------------------------------------------------------------

    def resolve_gaps(
        self,
        gap_analysis_text: str,
    ) -> Dict[str, Any]:
        """Run gap-driven paper discovery and ingestion.

        Convenience wrapper around ``GapResolver.resolve_gaps()`` that
        passes the orchestrator's graph_storage and enables ingestion.
        """
        from src.agents.gap_resolver import GapResolver

        resolver = GapResolver(max_papers_per_gap=self.max_papers_per_query)
        return resolver.resolve_gaps(
            gap_analysis_text,
            graph_storage=self.graph_storage,
            ingest=True,
        )


# ── Module-level parallel worker ────────────────────────────────────────────


def _fetch_and_parse_for_query(
    query: str,
    *,
    max_papers: int = 5,
    completed_pmcids: set | None = None,
) -> Dict[str, Any]:
    """Search EPMC for *query*, fetch full-text XML, and parse into chunks.

    Designed to run in parallel via ``run_parallel``.  Each invocation
    creates its own ``EuropePMCClient`` / ``PMCXMLParser`` so threads
    do not share state.

    Args:
        query: EPMC search query string.
        max_papers: Maximum papers to return from EPMC search.
        completed_pmcids: Set of already-ingested PMCIDs to skip.

    Returns:
        Dict with ``query`` and ``paper_data`` keys.
    """
    from src.retrieval.europe_pmc import EuropePMCClient
    from src.ingestion.pmc_xml_parser import PMCXMLParser

    epmc = EuropePMCClient()
    parser = PMCXMLParser()
    completed = completed_pmcids or set()

    papers = epmc.search(query, oa_only=True, max_results=max_papers)
    if not papers:
        return {"query": query, "paper_data": []}

    pmcids = [p.get("pmcid", "") for p in papers if p.get("pmcid")]
    xml_docs = epmc.full_text_xml_batch(pmcids) if pmcids else {}

    paper_data: List[Dict[str, Any]] = []
    for p in papers:
        pmcid = p.get("pmcid", "")
        if not pmcid or pmcid in completed:
            continue
        xml = xml_docs.get(pmcid)
        if not xml:
            continue
        try:
            chunks = parser.parse(xml, pmcid=pmcid, doi=p.get("doi", ""))
        except Exception:
            continue
        if chunks:
            paper_data.append({
                "pmcid": pmcid,
                "doi": p.get("doi", ""),
                "title": (p.get("title") or "")[:100],
                "chunks": chunks,
                "query": query,
            })

    return {"query": query, "paper_data": paper_data}
