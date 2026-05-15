"""
Scheduler — cron/timer skeleton for the Phase 10 background orchestrator.
Provides a lightweight interval scheduler in a daemon thread.
No external dependencies (no cron, no APScheduler).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Scheduler:
    """Lightweight interval scheduler running a callback in a daemon thread."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    def schedule(
        self,
        interval_minutes: int,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run *callback* every *interval_minutes* in a background daemon thread.

        Only one schedule may be active at a time.
        """
        if self._running:
            logger.warning("Scheduler already running — ignoring duplicate schedule()")
            return

        def _loop() -> None:
            logger.info("Scheduler started (interval=%d min)", interval_minutes)
            self._running = True
            while not self._stop_event.is_set():
                try:
                    callback(*args, **kwargs)
                except Exception:
                    logger.exception("Scheduler callback failed")
                self._stop_event.wait(interval_minutes * 60)
            self._running = False
            logger.info("Scheduler stopped")

        self._stop_event.clear()
        self._thread = threading.Thread(target=_loop, daemon=True, name="scheduler")
        self._thread.start()

    def run_once(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run *callback* exactly once, blocking until complete."""
        return callback(*args, **kwargs)

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the scheduler to stop after the current cycle completes."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._running
