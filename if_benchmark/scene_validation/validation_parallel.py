"""Bounded, ordered CPU validation; worker count never enters artifact identity."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import islice
import multiprocessing


_POOL = ContextVar("m15_validation_pool", default=None)


@contextmanager
def validation_pool(workers: int):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("validation workers must be an integer between 1 and 64")
    if workers == 1:
        token = _POOL.set(None)
        try:
            yield
        finally:
            _POOL.reset(token)
        return
    # Spawn never inherits a simulator, renderer context or a parent's threads.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        token = _POOL.set((pool, workers))
        try:
            yield
        finally:
            _POOL.reset(token)


def ordered_validation_map(function, items):
    """At most 2*workers in flight; preserve input order and propagate failures."""
    active = _POOL.get()
    if active is None:
        yield from map(function, items)
        return
    pool, workers = active
    items = iter(items)
    pending = deque()
    sentinel = object()
    try:
        pending.extend(pool.submit(function, item) for item in islice(items, 2 * workers))
        while pending:
            yield pending.popleft().result()
            item = next(items, sentinel)
            if item is not sentinel:
                pending.append(pool.submit(function, item))
    finally:
        for future in pending:
            future.cancel()
