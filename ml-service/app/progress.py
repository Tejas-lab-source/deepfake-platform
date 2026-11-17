"""Progress of running analyses, for GET /progress/{id}.

The server passes a progress id with an analysis request and polls this store while the
(blocking) request runs. Values are measured: faces or frames done out of the total. Entries
expire, so the store cannot grow without bound.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

# Progress callback: (stage, done, total, unit).
ProgressFn = Callable[[str, int, int, str], None]


def no_progress(_stage: str, _done: int, _total: int, _unit: str) -> None:
    return None


PROGRESS_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class ProgressStore:
    def __init__(self, ttl_s: float = 3600.0, max_entries: int = 1000) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, dict[str, Any]]] = {}
        self._ttl = ttl_s
        self._max = max_entries

    def set(self, progress_id: str, stage: str, done: int, total: int, unit: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._items[progress_id] = (
                now,
                {"stage": stage, "done": int(done), "total": int(total), "unit": unit},
            )
            if len(self._items) > self._max:
                self._prune(now)

    def get(self, progress_id: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(progress_id)
            if item is None or now - item[0] > self._ttl:
                return None
            return dict(item[1])

    def _prune(self, now: float) -> None:
        for key in [k for k, (t, _) in self._items.items() if now - t > self._ttl]:
            del self._items[key]
        # Still too many: drop the oldest.
        while len(self._items) > self._max:
            oldest = min(self._items, key=lambda k: self._items[k][0])
            del self._items[oldest]

    def reporter(self, progress_id: str | None) -> ProgressFn:
        """A progress callback for one request (a no-op without an id)."""
        if progress_id is None:
            return no_progress

        def report(stage: str, done: int, total: int, unit: str) -> None:
            self.set(progress_id, stage, done, total, unit)

        return report
