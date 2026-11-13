"""Per-provider circuit breaker: stop hammering an upstream that is down."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    CLOSED   --N consecutive failures-->  OPEN
    OPEN     --reset timeout elapses-->   HALF_OPEN (admits a single probe)
    HALF_OPEN --probe succeeds-->         CLOSED
    HALF_OPEN --probe fails-->            OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self.state = State.CLOSED
        self.consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_started_at: float | None = None

    def allow(self) -> bool:
        now = self._clock()
        if self.state is State.OPEN:
            if now - self._opened_at < self.reset_timeout_s:
                return False
            self.state = State.HALF_OPEN
            self._probe_started_at = None
        if self.state is State.HALF_OPEN:
            # A probe that never reported back (e.g. cancelled request) must not wedge the breaker.
            if self._probe_started_at is not None and now - self._probe_started_at < self.reset_timeout_s:
                return False
            self._probe_started_at = now
        return True

    def record_success(self) -> None:
        self.state = State.CLOSED
        self.consecutive_failures = 0
        self._probe_started_at = None

    def record_failure(self) -> None:
        self._probe_started_at = None
        if self.state is State.HALF_OPEN:
            self._trip()
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self.state = State.OPEN
        self._opened_at = self._clock()

    def snapshot(self) -> dict:
        return {"state": self.state.value, "consecutive_failures": self.consecutive_failures}
