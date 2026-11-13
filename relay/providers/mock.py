"""A deterministic fake upstream for local development, tests and load testing."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from .base import (
    Completion,
    Provider,
    ProviderError,
    ProviderRequest,
    StreamDelta,
    StreamEnd,
    StreamEvent,
    estimate_tokens,
)


class MockProvider(Provider):
    def __init__(self, name: str, latency_ms: int = 30, failure_rate: float = 0.0, seed: int | None = None):
        super().__init__(name)
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.calls = 0
        self._rng = random.Random(seed)

    async def _simulate_upstream(self) -> None:
        self.calls += 1
        await asyncio.sleep(self.latency_ms / 1000)
        if self._rng.random() < self.failure_rate:
            raise ProviderError(f"{self.name}: simulated upstream 503", retryable=True, status_code=503)

    def _reply(self, req: ProviderRequest) -> tuple[str, str]:
        last_user = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
        words = f"[{self.name}:{req.model}] You said: {last_user}".split(" ")
        finish = "length" if len(words) > req.max_tokens else "stop"
        return " ".join(words[: req.max_tokens]), finish

    def _input_tokens(self, req: ProviderRequest) -> int:
        return estimate_tokens(" ".join(m["content"] for m in req.messages))

    async def complete(self, req: ProviderRequest) -> Completion:
        await self._simulate_upstream()
        text, finish = self._reply(req)
        return Completion(text, self._input_tokens(req), estimate_tokens(text), finish, self.name, req.model)

    async def stream(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        await self._simulate_upstream()
        text, finish = self._reply(req)
        for i, word in enumerate(text.split(" ")):
            yield StreamDelta(word if i == 0 else " " + word)
            await asyncio.sleep(0.002)
        yield StreamEnd(self._input_tokens(req), estimate_tokens(text), finish)
