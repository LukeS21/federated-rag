"""Tests for scheduler.py — thread lifecycle, callback execution, error handling.

These are real integration tests (no mocking). They verify that:
  - schedule() actually calls the callback on interval
  - run_once() blocks and returns the callback result
  - stop() terminates the loop cleanly
  - is_running reflects thread state
  - Callback crashes do NOT crash the scheduler
  - Duplicate schedule() calls are rejected
"""
import time
import pytest
from src.agents.scheduler import Scheduler


class TestSchedulerLifecycle:
    """Happy-path: schedule → run → stop.

    NOTE: Scheduler.schedule() takes *minutes*.  Tests use fractions
    like 1/6000 ≈ 0.01 s real time per "minute" passed to schedule().
    """

    def test_schedule_runs_callback_repeatedly(self):
        ticks = []
        s = Scheduler()
        s.schedule(0.005, lambda: ticks.append(time.monotonic()))
        # 0.005 min = 0.3 s per tick.  Sleep 1.0 s → ~3 ticks.
        time.sleep(1.0)
        s.stop()
        assert len(ticks) >= 2, f"Expected >= 2 ticks, got {len(ticks)}"

    def test_stop_terminates_loop(self):
        ticks = []
        s = Scheduler()
        s.schedule(0.005, lambda: ticks.append(1))
        time.sleep(1.0)
        s.stop()
        count_before = len(ticks)
        time.sleep(0.5)
        assert len(ticks) == count_before, (
            f"Ticks kept incrementing after stop: {count_before} → {len(ticks)}"
        )

    def test_run_once_blocks_and_returns(self):
        results = []
        s = Scheduler()
        ret = s.run_once(lambda: results.append(42) or "done")
        assert ret == "done"
        assert results == [42]

    def test_is_running(self):
        s = Scheduler()
        assert not s.is_running
        s.schedule(0.005, lambda: None)
        time.sleep(0.4)
        assert s.is_running
        s.stop()
        time.sleep(0.4)
        assert not s.is_running

    def test_duplicate_schedule_ignored(self):
        ticks = []
        s = Scheduler()
        s.schedule(0.005, lambda: ticks.append("A"))
        s.schedule(0.005, lambda: ticks.append("B"))  # should warn and skip
        time.sleep(1.0)
        s.stop()
        assert all(t == "A" for t in ticks), f"Got mixed ticks: {ticks}"


class TestSchedulerErrorHandling:
    """Callback crashes should not kill the scheduler."""

    def test_callback_crash_does_not_kill_scheduler(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise RuntimeError("simulated crash")

        s = Scheduler()
        s.schedule(0.005, flaky)  # 0.3 s interval
        time.sleep(1.5)
        s.stop()
        assert call_count[0] >= 2, f"Expected >= 2 calls after crash, got {call_count[0]}"


class TestSchedulerRunOnceArgs:
    """run_once forwards positional/keyword args."""

    def test_run_once_forwards_args(self):
        s = Scheduler()
        result = s.run_once(lambda a, b, x=0: a + b + x, 10, 32, x=8)
        assert result == 50
