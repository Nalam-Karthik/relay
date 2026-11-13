"""Per-tenant token-bucket rate limiting on both request count and LLM token volume."""

from __future__ import annotations

import time
from collections.abc import Callable

from .config import TenantConfig


class TokenBucket:
    def __init__(self, capacity: float, refill_per_s: float, clock: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.tokens = capacity
        self._clock = clock
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.refill_per_s)
        self._last = now

    def try_acquire(self, n: float = 1.0) -> float:
        """Take n tokens. Returns 0 on success, otherwise seconds until n tokens will be available."""
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return 0.0
        return (n - self.tokens) / self.refill_per_s

    def consume(self, n: float) -> None:
        """Take n tokens unconditionally; the bucket may go into debt."""
        self._refill()
        self.tokens -= n

    def wait_until_positive(self) -> float:
        self._refill()
        if self.tokens > 0:
            return 0.0
        return (1 - self.tokens) / self.refill_per_s


class RateLimiter:
    """
    Requests are charged up front. LLM tokens can't be known until the response
    finishes, so token usage is charged afterwards and a tenant in token debt is
    blocked until the bucket refills ("pay later" limiting).

    State is in-process; a multi-replica deployment would move this into Redis
    with an atomic Lua script.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[TokenBucket, TokenBucket]] = {}

    def _get(self, tenant: TenantConfig) -> tuple[TokenBucket, TokenBucket]:
        if tenant.name not in self._buckets:
            rpm, tpm = tenant.requests_per_minute, tenant.tokens_per_minute
            self._buckets[tenant.name] = (
                TokenBucket(rpm, rpm / 60, self._clock),
                TokenBucket(tpm, tpm / 60, self._clock),
            )
        return self._buckets[tenant.name]

    def check(self, tenant: TenantConfig) -> float:
        """Returns 0 if admitted, else a Retry-After in seconds."""
        requests, tokens = self._get(tenant)
        if wait := tokens.wait_until_positive():
            return wait
        return requests.try_acquire(1)

    def record_tokens(self, tenant: TenantConfig, n: int) -> None:
        self._get(tenant)[1].consume(n)
