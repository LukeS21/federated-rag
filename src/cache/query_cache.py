"""Multi-level query result cache for Survey Mode.

Caches intermediate and final results of the Survey Mode pipeline
to avoid redundant LLM calls on repeated queries.

Level 1: Query decomposition (hash of query text + doc count)
Level 2: Per-theme synthesis (hash of theme + papers + evidence)
Level 3: Cross-theme synthesis + gap analysis (hash of all per-theme outputs)

Visible logging for debugging; TTL-based auto-invalidation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.cache import CACHE_VERSION

logger = logging.getLogger(__name__)

CACHE_DIR = "projects/default/query_cache"
DEFAULT_TTL = 7 * 86400  # 7 days


def _ensure_dir() -> Path:
    d = Path(CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(*parts: str) -> str:
    raw = f"{CACHE_VERSION}|{'|'.join(parts)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write(key: str, data: dict, ttl: int = DEFAULT_TTL) -> None:
    path = _ensure_dir() / f"{key}.json"
    data["_cached_at"] = time.time()
    data["_ttl"] = ttl
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read(key: str) -> Optional[dict]:
    path = _ensure_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data.get("_cached_at", 0) > data.get("_ttl", DEFAULT_TTL):
            path.unlink(missing_ok=True)
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


# ── Level 1: Query Decomposition ──────────────────────────────────────────────
def cache_query_decomposition(query: str, doc_count: int, themes: List[Dict[str, Any]]) -> None:
    """Store decomposed themes for a query."""
    key = _hash(query.strip().lower(), str(doc_count))
    _write(key, {"type": "decomposition", "query": query, "themes": themes})
    logger.info("[query-cache] STORED decomposition | query='%s...' | %d themes | key=%s",
                query[:60], len(themes), key[:12])


def load_query_decomposition(query: str, doc_count: int) -> Optional[List[Dict[str, Any]]]:
    """Retrieve cached decomposition, or None."""
    key = _hash(query.strip().lower(), str(doc_count))
    data = _read(key)
    if data is None:
        logger.info("[query-cache] MISS decomposition | query='%s...' | key=%s", query[:60], key[:12])
        return None
    themes = data.get("themes", [])
    age_h = (time.time() - data.get("_cached_at", 0)) / 3600
    logger.info("[query-cache] HIT decomposition | query='%s...' | %d themes | age=%.1fh | key=%s",
                query[:60], len(themes), age_h, key[:12])
    return themes


# ── Level 2: Per-Theme Synthesis ──────────────────────────────────────────────
def _theme_cache_key(
    theme_name: str,
    paper_ids: List[str],
    query_hash: str,
    evidence_hash: str,
) -> str:
    return _hash(theme_name, ",".join(sorted(paper_ids)), query_hash, evidence_hash)


def _compute_evidence_hash(chunks: List[Dict[str, Any]]) -> str:
    """Hash the evidence text for staleness detection."""
    texts = sorted(
        str((ch.get("metadata", {}) or {}).get("chunk_summary", ch.get("text", "")[:200]))
        for ch in chunks
    )
    return _hash(*texts[:100])  # cap at 100 chunks for speed


def cache_theme_synthesis(
    theme_name: str,
    paper_ids: List[str],
    query: str,
    theme_chunks: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> None:
    """Store per-theme synthesis result."""
    qh = _hash(query.strip().lower())
    eh = _compute_evidence_hash(theme_chunks)
    key = _theme_cache_key(theme_name, paper_ids, qh, eh)
    _write(key, {
        "type": "theme_synthesis",
        "theme_name": theme_name,
        "paper_ids": paper_ids,
        "result": result,
    })
    logger.info("[query-cache] STORED theme | '%s' | %d papers | score=%.2f | key=%s",
                theme_name, len(paper_ids), result.get("anchoring_score", 0), key[:12])


def load_theme_synthesis(
    theme_name: str,
    paper_ids: List[str],
    query: str,
    theme_chunks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Retrieve cached theme synthesis, or None."""
    qh = _hash(query.strip().lower())
    eh = _compute_evidence_hash(theme_chunks)
    key = _theme_cache_key(theme_name, paper_ids, qh, eh)
    data = _read(key)
    if data is None:
        logger.info("[query-cache] MISS theme | '%s' | key=%s", theme_name, key[:12])
        return None
    result = data.get("result", {})
    age_h = (time.time() - data.get("_cached_at", 0)) / 3600
    logger.info("[query-cache] HIT theme | '%s' | score=%.2f | age=%.1fh | key=%s",
                theme_name, result.get("anchoring_score", 0), age_h, key[:12])
    return result


# ── Level 3: Cross-Theme Synthesis ────────────────────────────────────────────
def cache_cross_theme(
    query: str,
    theme_syntheses: Dict[str, Any],
    cross_synthesis: str,
    gap_analysis: str,
) -> None:
    """Store cross-theme synthesis and gap analysis."""
    themes_json = json.dumps(
        {k: v.get("anchoring_score", 0) for k, v in theme_syntheses.items()},
        sort_keys=True,
    )
    key = _hash(query.strip().lower(), themes_json)
    _write(key, {
        "type": "cross_theme",
        "query": query,
        "cross_theme_synthesis": cross_synthesis,
        "gap_analysis": gap_analysis,
    })
    logger.info("[query-cache] STORED cross-theme | query='%s...' | key=%s",
                query[:60], key[:12])


def load_cross_theme(
    query: str,
    theme_syntheses: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Retrieve cached cross-theme + gap, or None."""
    themes_json = json.dumps(
        {k: v.get("anchoring_score", 0) for k, v in theme_syntheses.items()},
        sort_keys=True,
    )
    key = _hash(query.strip().lower(), themes_json)
    data = _read(key)
    if data is None:
        logger.info("[query-cache] MISS cross-theme | query='%s...' | key=%s",
                    query[:60], key[:12])
        return None
    age_h = (time.time() - data.get("_cached_at", 0)) / 3600
    logger.info("[query-cache] HIT cross-theme | query='%s...' | age=%.1fh | key=%s",
                query[:60], age_h, key[:12])
    return {
        "cross_theme_synthesis": data.get("cross_theme_synthesis", ""),
        "gap_analysis": data.get("gap_analysis", ""),
    }
