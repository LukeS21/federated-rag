"""Tests for subagents.py — parallel execution, error isolation, kwargs.

These tests exercise real ThreadPoolExecutor behavior with fast functions.
No mocking — we verify actual concurrency semantics:
  - All inputs are processed
  - Results are returned in the expected structure
  - One crash does NOT affect other tasks
  - kwarg forwarding works
  - Empty input returns empty list
  - max_workers is respected (doesn't create more threads than items)
"""
import pytest
from src.agents.subagents import run_parallel


def _double(x: int) -> int:
    return x * 2


def _crash_on_three(x: int) -> int:
    if x == 3:
        raise ValueError("three is not allowed")
    return x * 10


def _with_kwargs(x: int, *, multiplier: int = 2) -> int:
    return x * multiplier


class TestRunParallel:
    """Core run_parallel behavior."""

    def test_all_succeed(self):
        results = run_parallel(_double, [1, 2, 3, 4, 5])
        assert len(results) == 5
        for r in results:
            assert r["error"] is None
            assert r["result"] == r["item"] * 2

    def test_empty_input(self):
        assert run_parallel(_double, []) == []

    def test_error_isolation(self):
        """One task crashing should not prevent others from succeeding."""
        results = run_parallel(_crash_on_three, [1, 2, 3, 4, 5])
        assert len(results) == 5

        success_count = sum(1 for r in results if r["error"] is None)
        error_count = sum(1 for r in results if r["error"] is not None)
        assert success_count == 4
        assert error_count == 1

        # the failed item should be 3
        failed = [r for r in results if r["error"] is not None]
        assert failed[0]["item"] == 3
        assert "three is not allowed" in str(failed[0]["error"])

    def test_kwargs_forwarding(self):
        results = run_parallel(_with_kwargs, [1, 2, 3], multiplier=10)
        assert len(results) == 3
        for r in results:
            assert r["result"] == r["item"] * 10

    def test_result_structure(self):
        """Every result dict has the expected keys."""
        results = run_parallel(_double, [7])
        assert len(results) == 1
        r = results[0]
        assert "item" in r
        assert "result" in r
        assert "error" in r
        assert r["item"] == 7
        assert r["result"] == 14
        assert r["error"] is None

    def test_max_workers_limits_threads(self):
        """With max_workers=1, execution is effectively serial."""
        import time
        order = []

        def slow(x: int) -> int:
            time.sleep(0.02)
            order.append(x)
            return x

        results = run_parallel(slow, [1, 2, 3], max_workers=1)
        assert len(results) == 3
        # With 1 worker, completion order should match submission order
        # (each task fully completes before the next starts)
        assert order == [1, 2, 3]

    def test_single_item(self):
        results = run_parallel(_double, [42])
        assert len(results) == 1
        assert results[0]["result"] == 84
