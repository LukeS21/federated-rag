"""Disk-based LLM response cache with TTL (time-to-live).

Hashes (system_prompt, user_prompt) → stores response. Eliminates redundant
API calls for identical or near-identical queries.

Used by: summarizer, category discovery, extraction (all temperature=0).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


class LLMCache:
    """Simple file‑based cache with 24‑hour TTL."""

    def __init__(self, cache_dir: str = "projects/default/cache", ttl_seconds: int = 86400) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    def _key(self, system_prompt: str, user_prompt: str) -> str:
        raw = f"{system_prompt}|||{user_prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        key = self._key(system_prompt, user_prompt)
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - data.get("ts", 0) > self._ttl:
                p.unlink(missing_ok=True)
                return None
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, system_prompt: str, user_prompt: str, response: str) -> None:
        key = self._key(system_prompt, user_prompt)
        self._path(key).write_text(
            json.dumps({"ts": time.time(), "response": response}, ensure_ascii=False),
            encoding="utf-8",
        )


# Module-level singleton for reuse across agents
_cache: Optional[LLMCache] = None


def get_cache() -> LLMCache:
    global _cache
    if _cache is None:
        _cache = LLMCache()
    return _cache
