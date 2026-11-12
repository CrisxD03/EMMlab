"""Process pools via multiprocessing.Pool (ProcessPoolExecutor's semaphore probe fails in sandboxes)."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from typing import Any


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def get_pool(workers: int | None = None):
    return mp.get_context("spawn").Pool(workers or default_workers())


def pmap(fn: Callable[[Any], Any], items: Iterable[Any], workers: int | None = None) -> list[Any]:
    items = list(items)
    if (workers or default_workers()) == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with get_pool(workers) as pool:
        return pool.map(fn, items, chunksize=1)
