"""Pre-extraction at ingest time — runs entity extraction on a paper's chunks
during PDF ingestion and stores results for query-time reuse.

Also pre-computes paper embeddings (via sentence-transformers) for
deterministic thematic clustering at query time.

Eliminates the need for per-document LLM extraction at query time, which is
the dominant query cost in Survey Mode (~60% of LLM calls).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.extraction_agent import ExtractionAgent
from src.graph.graph_builder import GraphBuilder
from src.graph.base_graph import BaseGraphStorage
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

EXTRACTIONS_DIR = "projects/default/extractions"
EMBEDDINGS_DIR = "projects/default/embeddings"


def _ensure_dir() -> Path:
    d = Path(EXTRACTIONS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_embeddings_dir() -> Path:
    d = Path(EMBEDDINGS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


class PreExtractor:
    """Runs per-document entity extraction at ingest time.

    Extracted entities are stored on disk as JSON and fed into the
    shared knowledge graph. At query time, entities are loaded from
    disk instead of re-running LLM extraction per paper.
    """

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.model = model
        self.agent = ExtractionAgent(model=model)

    def extract_paper(
        self,
        paper_id: str,
        chunks: List[Dict[str, Any]],
        query: Optional[str] = None,
        graph_storage: Optional[BaseGraphStorage] = None,
    ) -> Dict[str, Any]:
        """Extract entities from a paper's chunks and persist to disk + KG.

        Also pre-computes and caches the paper embedding for fast
        thematic clustering at query time.

        Args:
            paper_id: Unique paper identifier (e.g., filename).
            chunks: Pre-summarized chunks from this paper.
            query: Optional query to condition extraction. If None, uses a
                   default broad query to extract all entities.
            graph_storage: Optional KG storage to feed entities into.

        Returns:
            The extracted entities dict (category → list of entity objects).
        """
        if not chunks:
            return {}

        if query is None:
            query = (
                "What are the key findings, materials, cell types, cytokines, "
                "experimental methods, model systems, and results described "
                "in this paper?"
            )

        logger.info("Pre-extracting entities from %s (%d chunks)...", paper_id, len(chunks))

        # Build summary from chunk summaries for embedding
        summary_parts = []
        for ch in chunks:
            meta = ch.get("metadata", {}) or {}
            cs = meta.get("chunk_summary", str(ch.get("text", ""))[:200])
            if cs:
                summary_parts.append(cs)
        paper_summary = " ".join(summary_parts[:20])  # capped for embedding quality

        # Category discovery on summaries
        summary_chunks = []
        for ch in chunks:
            meta = ch.get("metadata", {}) or {}
            s = meta.get("chunk_summary", ch.get("text", "")[:200])
            summary_chunks.append({"text": s, "metadata": meta})

        categories = self.agent.discover_categories(summary_chunks, query)
        entities = self.agent.extract_entities_batched(chunks, categories, query)

        logger.info("  %s: %d entity groups extracted", paper_id, len(entities))

        # Feed into knowledge graph if available
        if graph_storage is not None and entities:
            try:
                GraphBuilder().build(entities, chunks, graph_storage)
                logger.debug("  KG updated with entities from %s", paper_id)
            except Exception as e:
                logger.warning("  KG update failed for %s: %s", paper_id, e)

        # Persist entities to disk
        self._save(paper_id, entities)

        # Pre-compute and cache paper embedding
        if paper_summary:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embedding = model.encode([paper_summary[:2000]], show_progress_bar=False)[0]
                self.save_embedding(paper_id, embedding)
                logger.debug("  embedding cached for %s", paper_id)
            except Exception as e:
                logger.debug("  embedding cache skipped for %s: %s", paper_id, e)

        # Force Ollama to unload and reload the model between papers.
        # Prevents Metal backend memory fragmentation that accumulates across
        # 100+ sequential inference requests, causing late-cycle hangs and
        # garbage output ("TYPE: TYPE: TYPE:").
        PreExtractor._reset_ollama()

        return entities

    def _save(self, paper_id: str, entities: Dict[str, Any]) -> None:
        path = _ensure_dir() / f"{paper_id}.json"
        # Convert to serializable form
        serializable = {}
        for key, ent_list in entities.items():
            serializable[key] = [
                {k: (str(v)[:500] if isinstance(v, str) else v) for k, v in e.items()}
                for e in (ent_list if isinstance(ent_list, list) else [])
                if isinstance(e, dict)
            ]
        path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False, default=str))

    @staticmethod
    def load(paper_id: str) -> Optional[Dict[str, Any]]:
        """Load pre-extracted entities from disk for a given paper.

        Returns None if the extraction file does not exist.
        """
        path = Path(EXTRACTIONS_DIR) / f"{paper_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load extraction for %s: %s", paper_id, e)
            return None

    @staticmethod
    def is_extracted(paper_id: str) -> bool:
        """Check whether a paper has already been pre-extracted."""
        return (Path(EXTRACTIONS_DIR) / f"{paper_id}.json").exists()

    @staticmethod
    def save_embedding(paper_id: str, embedding: np.ndarray) -> None:
        """Store pre-computed paper embedding to disk."""
        d = _ensure_embeddings_dir()
        path = d / f"{paper_id}.npy"
        np.save(str(path), embedding)

    @staticmethod
    def load_embedding(paper_id: str) -> Optional[np.ndarray]:
        """Load a pre-computed paper embedding, or None if not cached."""
        path = Path(EMBEDDINGS_DIR) / f"{paper_id}.npy"
        if not path.exists():
            return None
        try:
            return np.load(str(path))
        except (OSError, ValueError) as e:
            logger.warning("Failed to load embedding for %s: %s", paper_id, e)
            return None

    @staticmethod
    def load_all_embeddings() -> Dict[str, np.ndarray]:
        """Load all cached paper embeddings.

        Returns mapping of paper_id → embedding array.
        """
        d = Path(EMBEDDINGS_DIR)
        if not d.exists():
            return {}
        result: Dict[str, np.ndarray] = {}
        for path in sorted(d.glob("*.npy")):
            paper_id = path.stem
            try:
                result[paper_id] = np.load(str(path))
            except (OSError, ValueError) as e:
                logger.warning("Failed to load embedding for %s: %s", paper_id, e)
        return result

    @staticmethod
    def _reset_ollama(
        model_name: str = "gemma4:e4b",
        ollama_host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        """Unload *model_name* from Ollama and poll until confirmed gone.

        Step 1: POST ``/api/generate`` with ``keep_alive=0`` to request unload.
        Step 2: Poll ``GET /api/ps`` (0.5 s interval) until *model_name*
           disappears from the running‑models list — confirming GPU memory
           is fully reclaimed by Metal.

        This prevents Memory-Allocation-Fragmentation (MAF) in llama.cpp's
        Metal backend that accumulates across 100+ sequential inference
        requests during long daemon cycles.  Fragmented GPU buffers cause
        indefinite hangs, garbage output, and progressive slowdown.

        A fresh model load gives every batch / paper a clean slate.
        Typical latency: 1–5 s.  Worst‑case safety valve: *timeout* seconds.
        """
        try:
            import time as _time
            import urllib.request

            def _running_models() -> list[str]:
                """Return the list of model names currently loaded in Ollama."""
                try:
                    r = urllib.request.Request(
                        f"{ollama_host}/api/ps", method="GET",
                    )
                    data = json.loads(urllib.request.urlopen(r, timeout=5).read())
                    return [m.get("name", "") for m in data.get("models", [])]
                except Exception:
                    return []

            before = _running_models()
            logger.info(
                "Resetting Ollama — %d model(s) loaded before: %s",
                len(before), ", ".join(before) if before else "(none)",
            )
            t0 = _time.monotonic()

            # Step 1 — request unload
            body = json.dumps({
                "model": model_name,
                "prompt": ".",
                "keep_alive": 0,
                "options": {"num_predict": 1},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{ollama_host}/api/generate",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)

            # Step 2 — poll until confirmed gone
            while _time.monotonic() - t0 < timeout:
                _time.sleep(0.5)
                try:
                    ps_req = urllib.request.Request(
                        f"{ollama_host}/api/ps",
                        method="GET",
                    )
                    resp_data = json.loads(
                        urllib.request.urlopen(ps_req, timeout=5).read()
                    )
                    running = [
                        m.get("name", "")
                        for m in resp_data.get("models", [])
                    ]
                    if model_name not in running:
                        after = _running_models()
                        logger.info(
                            "Ollama reset complete in %.1fs — %d model(s) loaded now: %s",
                            _time.monotonic() - t0, len(after),
                            ", ".join(after) if after else "none — GPU memory cleared",
                        )
                        return
                except Exception:
                    continue  # transient ps failure, keep polling

            after = _running_models()
            logger.warning(
                "Ollama reset timed out after %.0fs — %d model(s) still loaded: %s",
                timeout, len(after), ", ".join(after) if after else "(none)",
            )
        except Exception:
            logger.debug("Ollama reset failed (non-fatal)")
            pass
