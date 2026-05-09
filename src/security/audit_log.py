"""Security audit logging for Phase 5.

Timestamped log of all security-relevant events:
  - LLM calls (model, prompt/response token counts, latency)
  - Boundary crossings (secure→public data redacted)
  - Scope routing decisions
  - Access patterns
  - Anomaly detections
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_AUDIT_LOG_PATH = os.getenv("SECURITY_AUDIT_LOG", "./logs/security_audit.log")


class AuditLogger:
    """Thread-safe security event logger.

    Writes timestamped JSON lines to a configurable file path.
    """

    def __init__(self, log_path: str | None = None) -> None:
        self._path = Path(log_path or _AUDIT_LOG_PATH)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("security.audit")

    def _write_event(self, event: Dict[str, Any]) -> None:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str) + "\n")
            except OSError as e:
                self.logger.error("Failed to write audit event: %s", e)

    def log_llm_call(
        self,
        model: str,
        provider: str,
        operation: str,
        prompt_chars: int,
        response_chars: int,
        latency_ms: float,
        cached: bool = False,
    ) -> None:
        self._write_event({
            "type": "llm_call",
            "model": model,
            "provider": provider,
            "operation": operation,
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
            "latency_ms": round(latency_ms, 1),
            "cached": cached,
        })

    def log_boundary_crossing(
        self,
        direction: str,
        redaction_count: int,
        output_chars_before: int,
        output_chars_after: int,
    ) -> None:
        self._write_event({
            "type": "boundary_crossing",
            "direction": direction,
            "redaction_count": redaction_count,
            "output_chars_before": output_chars_before,
            "output_chars_after": output_chars_after,
        })

    def log_scope_routing(
        self,
        query_scope: str,
        mode: str,
        routing_decision: str,
        context_keys: List[str],
    ) -> None:
        self._write_event({
            "type": "scope_routing",
            "query_scope": query_scope,
            "mode": mode,
            "routing_decision": routing_decision,
            "context_keys": context_keys,
        })

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
    ) -> None:
        self._write_event({
            "type": "security_event",
            "event_type": event_type,
            "severity": severity,
            "details": details,
        })

    def log_access(
        self,
        operation: str,
        resource: str,
        query: str | None = None,
    ) -> None:
        self._write_event({
            "type": "access",
            "operation": operation,
            "resource": resource,
            "query_hash": _hash_query(query) if query else None,
        })

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not self._path.exists():
            return events
        with self._lock:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                pass
        return events[-limit:]


_audit_logger: AuditLogger | None = None
_audit_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        with _audit_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger


def _hash_query(query: str) -> str:
    import hashlib
    return hashlib.sha256(query.encode()).hexdigest()[:16]
