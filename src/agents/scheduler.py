"""
Scheduler — lightweight daemon loop for the Phase 10 background orchestrator.

Chain-based loop: runs callback → sleeps cooldown → runs callback → ...
No fixed-interval timer — the next cycle starts cooldown_seconds after the
previous cycle finishes, regardless of how long the previous cycle took.

No external dependencies (no cron, no APScheduler).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Scheduler:
    """Chain-based background loop with configurable cooldown.

    Unlike a cron-style timer (which fires at fixed wall-clock times),
    this runs the callback in a continuous chain: execute → cooldown →
    execute → cooldown → ...  A long-running callback doesn't cause
    missed ticks or overlapping executions — the next cycle simply
    starts later.

    Usage::

        s = Scheduler()
        s.schedule(callback, cooldown_seconds=600)   # 10 min between cycles
        # ... daemon runs ...
        s.stop()
    """

    DEFAULT_COOLDOWN_SECONDS = 60

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    def schedule(
        self,
        callback: Callable[..., Any],
        *args: Any,
        cooldown_seconds: int | float = 0,
        **kwargs: Any,
    ) -> None:
        """Run *callback* in a daemon thread. After each invocation, sleep
        *cooldown_seconds* before running again.  Only one loop may be
        active at a time.

        If *cooldown_seconds* is 0, there is no delay between cycles
        (the next cycle starts immediately after the previous finishes).
        """
        if self._running:
            logger.warning("Scheduler already running — ignoring duplicate schedule()")
            return

        def _loop() -> None:
            logger.info("Scheduler started (cooldown=%ss)", cooldown_seconds)
            self._running = True
            while not self._stop_event.is_set():
                try:
                    callback(*args, **kwargs)
                except Exception:
                    logger.exception("Scheduler callback failed")
                if cooldown_seconds > 0 and not self._stop_event.is_set():
                    if self._stop_event.wait(cooldown_seconds):
                        break
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
