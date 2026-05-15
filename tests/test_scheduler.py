"""Tests for scheduler.py — chain-based loop lifecycle, callback execution, error handling.

Tests verify that:
  - schedule() runs callback repeatedly with cooldown between cycles
  - run_once() blocks and returns the callback result
  - stop() terminates the loop cleanly
  - is_running reflects thread state
  - Callback crashes do NOT crash the scheduler
  - Duplicate schedule() calls are rejected
  - Zero cooldown runs back-to-back
"""
import time
import pytest
from src.agents.scheduler import Scheduler


class TestSchedulerLifecycle:
    """Happy-path: schedule → run → stop."""

    def test_schedule_runs_callback_repeatedly(self):
        ticks = []
        s = Scheduler()
        s.schedule(lambda: ticks.append(time.monotonic()), cooldown_seconds=0.1)
        time.sleep(0.7)
        s.stop()
        assert len(ticks) >= 3, f"Expected >= 3 ticks, got {len(ticks)}"

    def test_stop_terminates_loop(self):
        ticks = []
        s = Scheduler()
        s.schedule(lambda: ticks.append(1), cooldown_seconds=0.1)
        time.sleep(0.5)
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
        s.schedule(lambda: None, cooldown_seconds=0.1)
        time.sleep(0.3)
        assert s.is_running
        s.stop()
        time.sleep(0.3)
        assert not s.is_running

    def test_duplicate_schedule_ignored(self):
        ticks = []
        s = Scheduler()
        s.schedule(lambda: ticks.append("A"), cooldown_seconds=0.1)
        s.schedule(lambda: ticks.append("B"), cooldown_seconds=0.1)
        time.sleep(0.5)
        s.stop()
        assert all(t == "A" for t in ticks), f"Got mixed ticks: {ticks}"

    def test_zero_cooldown_runs_back_to_back(self):
        """With cooldown_seconds=0 there is no delay between cycles."""
        ticks = []
        s = Scheduler()
        s.schedule(lambda: ticks.append(time.monotonic()), cooldown_seconds=0)
        time.sleep(0.3)
        s.stop()
        assert len(ticks) >= 5, f"Expected >= 5 rapid ticks, got {len(ticks)}"

    def test_long_cooldown_single_tick(self):
        """A very long cooldown should produce only 1 tick in a short window."""
        ticks = []
        s = Scheduler()
        s.schedule(lambda: ticks.append(1), cooldown_seconds=60)
        time.sleep(0.2)
        s.stop()
        assert len(ticks) == 1, f"Expected exactly 1 tick, got {len(ticks)}"


class TestSchedulerErrorHandling:
    """Callback crashes should not kill the scheduler."""

    def test_callback_crash_does_not_kill_scheduler(self):
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise RuntimeError("simulated crash")

        s = Scheduler()
        s.schedule(flaky, cooldown_seconds=0.05)
        time.sleep(0.4)
        s.stop()
        assert call_count[0] >= 2, f"Expected >= 2 calls after crash, got {call_count[0]}"


class TestSchedulerRunOnceArgs:
    """run_once forwards positional/keyword args."""

    def test_run_once_forwards_args(self):
        s = Scheduler()
        result = s.run_once(lambda a, b, x=0: a + b + x, 10, 32, x=8)
        assert result == 50
