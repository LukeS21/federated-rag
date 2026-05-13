"""
SQLite-backed cache store for all cache levels (L0–L4).

Replaces the file-per-entry JSON pattern with a single SQLite database,
eliminating filesystem churn at scale.  Same hash‑based key pattern,
same TTL‑based expiry.  Compatible with existing ``CACHE_VERSION``
invalidation.

Usage::

    from src.cache.cache_store import CacheStore

    store = CacheStore("projects/default/cache.db")
    store.set("my_key", {"result": "..."}, ttl_seconds=86400)
    value = store.get("my_key")  # → dict or None
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from src.cache import CACHE_VERSION

logger = logging.getLogger(__name__)

TABLE_NAME = "cache_entries"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    cache_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_expires ON {TABLE_NAME}(expires_at);
"""


class CacheStore:
    """SQLite-backed multi‑tier cache store.

    Thread‑safe (single writer lock), auto‑creates table on first use.
    Keys are deterministic: ``SHA-256(CACHE_VERSION | namespace | ...)``.
    """

    def __init__(self, db_path: str | Path = "projects/default/cache.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _make_key(self, *parts: str) -> str:
        raw = CACHE_VERSION + "|" + "|||".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, *key_parts: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value. Returns None if missing or expired."""
        cache_key = self._make_key(*key_parts)
        with self._lock:
            try:
                conn = self._get_conn()
                row = conn.execute(
                    f"SELECT value_json, expires_at FROM {TABLE_NAME} WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                conn.close()
                if row is None:
                    return None
                value_json, expires_at = row
                if time.time() > expires_at:
                    self._delete(cache_key)
                    return None
                return json.loads(value_json)
            except Exception as e:
                logger.debug("CacheStore.get failed: %s", e)
                return None

    def set(
        self,
        value: Any,
        *key_parts: str,
        ttl_seconds: int = 86400,
    ) -> None:
        """Store a value with TTL."""
        cache_key = self._make_key(*key_parts)
        now = time.time()
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    f"INSERT OR REPLACE INTO {TABLE_NAME} (cache_key, value_json, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (cache_key, json.dumps(value, ensure_ascii=False), now, now + ttl_seconds),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug("CacheStore.set failed: %s", e)

    def _delete(self, cache_key: str) -> None:
        try:
            conn = self._get_conn()
            conn.execute(f"DELETE FROM {TABLE_NAME} WHERE cache_key = ?", (cache_key,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def invalidate(self, *key_parts: str) -> None:
        """Delete a specific cache entry."""
        cache_key = self._make_key(*key_parts)
        self._delete(cache_key)

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.execute(
                    f"DELETE FROM {TABLE_NAME} WHERE expires_at < ?",
                    (time.time(),),
                )
                count = cursor.rowcount
                conn.commit()
                conn.close()
                return count
            except Exception:
                return 0

    def count(self) -> int:
        """Total entries in the cache."""
        try:
            conn = self._get_conn()
            row = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0


# Module-level singleton
_store: Optional[CacheStore] = None


def get_cache_store(db_path: str | Path = "projects/default/cache.db") -> CacheStore:
    """Return or create the singleton CacheStore."""
    global _store
    if _store is None:
        _store = CacheStore(db_path)
    return _store
