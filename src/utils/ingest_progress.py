"""
Progress persistence for Phase 9 ingestion.

Saves a checkpoint to projects/default/ingest_progress.json every N papers
so that a crashed (or interrupted) pipeline can resume without re-ingesting
papers that were already committed to ChromaDB and the BM25 index.

Usage::

    from src.utils.ingest_progress import IngestProgress

    progress = IngestProgress()
    for paper in papers:
        pmcid = paper["pmcid"]
        if progress.is_completed(pmcid):
            continue
        chunks = parser.parse(xml, pmcid=pmcid, doi=paper.get("doi", ""))
        retriever.ingest(chunks)
        progress.checkpoint(pmcid)
    progress.finalize()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

DEFAULT_PROGRESS_PATH = Path("projects/default/ingest_progress.json")
CHECKPOINT_INTERVAL = 10  # save every N newly-ingested papers


class IngestProgress:
    """Tracks which papers have been ingested to avoid duplicate work."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_PROGRESS_PATH
        self.completed: Set[str] = set()
        self._unsaved = 0
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.completed = set(data.get("completed_pmcids", []))
                logger.info("Loaded progress: %d papers already ingested", len(self.completed))
            except (json.JSONDecodeError, KeyError):
                self.completed = set()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({
                "completed_pmcids": sorted(self.completed),
                "total_ingested": len(self.completed),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_completed(self, pmcid: str) -> bool:
        return pmcid in self.completed

    def mark_completed(self, pmcid: str) -> None:
        self.completed.add(pmcid)
        self._unsaved += 1

    def checkpoint(self, pmcid: str) -> None:
        """Mark a paper ingested and persist every CHECKPOINT_INTERVAL."""
        self.mark_completed(pmcid)
        if self._unsaved >= CHECKPOINT_INTERVAL or len(self.completed) % CHECKPOINT_INTERVAL == 0:
            self.save()
            self._unsaved = 0

    def finalize(self) -> None:
        self.save()
        logger.info("Finalised progress: %d papers ingested", len(self.completed))
