"""Real and simulated clocks."""
from __future__ import annotations

import threading
import time
from typing import Callable, List

from .interfaces import Clock


class RealClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, dt: float) -> None:
        if dt > 0:
            time.sleep(dt)


class SimClock(Clock):
    """
    Virtual time. sleep() advances time instantly and runs tick listeners.

    Deterministic and much faster than real time, so a 300 s task round can be
    simulated in about a second of wall time.
    """

    def __init__(self, start: float = 1000.0):
        self._t = float(start)
        self._lock = threading.Lock()
        self._listeners: List[Callable[[float, float], None]] = []

    def now(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, dt: float) -> None:
        if dt <= 0:
            return
        with self._lock:
            t0 = self._t
            self._t += float(dt)
            t1 = self._t
        for fn in list(self._listeners):
            fn(t0, t1)

    def advance(self, dt: float) -> None:
        self.sleep(dt)

    def add_listener(self, fn: Callable[[float, float], None]) -> None:
        """Register fn(t_old, t_new), called after every advance."""
        self._listeners.append(fn)
