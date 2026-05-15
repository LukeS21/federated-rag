"""
Subagents — parallel search/extract workers via ThreadPoolExecutor.
Enables concurrent EPMC search + XML fetch across multiple queries and
parallel pre-extraction across multiple papers.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_parallel(
    func: Callable[..., T],
    items: List[Any],
    *,
    max_workers: int = 4,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run *func(item, **kwargs)* for every item in *items* in parallel.

    Args:
        func: Callable receiving ``(item, **kwargs)`` as arguments.
        items: List of first-positional arguments, one per worker invocation.
        max_workers: Maximum number of ``ThreadPoolExecutor`` threads.
        **kwargs: Keyword arguments forwarded to every invocation of *func*.

    Returns:
        List of result dicts:
          ``{"item": item, "result": return_value, "error": None|str}``.
        Results are returned in completion order, not input order.
    """
    if not items:
        return []

    results: List[Dict[str, Any]] = []
    worker_count = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=max(worker_count, 1)) as executor:
        future_to_item = {
            executor.submit(func, item, **kwargs): item for item in items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                res = future.result()
                results.append({"item": item, "result": res, "error": None})
            except Exception as exc:
                logger.error("Subagent failed for %r: %s", item, exc)
                results.append({"item": item, "result": None, "error": str(exc)})
    return results
