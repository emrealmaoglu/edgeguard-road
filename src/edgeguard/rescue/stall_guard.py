"""A best-effort guard against a single blocking read hanging indefinitely.

A mounted Google Drive FUSE volume can stall indefinitely on a single read
instead of raising an OSError, which naive retry loops only catch on genuine
errors. This is shared infrastructure for anything that streams from a
Drive-mounted path: `colab_data.py`'s archive copy/hash helpers and
`archive_inventory.py`'s raw-archive scanning both use it.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from typing import Any

DEFAULT_STALL_TIMEOUT_SECONDS = 120


class StallTimeout(TimeoutError):
    """Raised when a single read from a (likely Drive-mounted) source stalls."""


def stall_guard(seconds: int | None) -> contextlib.AbstractContextManager[None]:
    """Best-effort read-stall guard; a no-op where SIGALRM is unavailable."""
    if seconds is None or seconds <= 0:
        return contextlib.nullcontext()
    if not hasattr(signal, "alarm") or threading.current_thread() is not threading.main_thread():
        return contextlib.nullcontext()
    return _AlarmGuard(seconds)


class _AlarmGuard(contextlib.AbstractContextManager["None"]):
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._previous: Any = None

    def __enter__(self) -> None:
        def _raise_stall(signum: int, frame: Any) -> None:
            raise StallTimeout(f"no data received for {self.seconds}s; likely a Drive/FUSE hang")

        self._previous = signal.signal(signal.SIGALRM, _raise_stall)
        signal.alarm(self.seconds)

    def __exit__(self, *_exc: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._previous)
