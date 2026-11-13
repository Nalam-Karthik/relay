"""Resolve a model name to a fallback chain and execute it with retries + circuit breaking."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace

from .circuit_breaker import CircuitBreaker
from .config import ResilienceConfig, Target
from .metrics import Metrics
from .providers import Completion, Provider, ProviderError, ProviderRequest, StreamEvent


@dataclass
class Attempt:
    provider: str
    model: str
    error: str


class NoRouteError(Exception):
    pass


class AllTargetsFailed(Exception):
    def __init__(self, attempts: list[Attempt]):
        self.attempts = attempts
        detail = "; ".join(f"{a.provider}/{a.model}: {a.error}" for a in attempts) or "no targets"
        super().__init__(f"all upstream targets failed ({detail})")


class Router:
    def __init__(
        self,
        routes: dict[str, list[Target]],
        providers: dict[str, Provider],
        resilience: ResilienceConfig,
        metrics: Metrics,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.routes = routes
        self.providers = providers
        self.cfg = resilience
        self.metrics = metrics
        self._sleep = sleep
        self.breakers = {
            name: CircuitBreaker(resilience.breaker_failure_threshold, resilience.breaker_reset_s)
            for name in providers
        }

    def resolve(self, model: str) -> list[Target]:
        """A route alias ("default") or a direct "provider/model" reference."""
        if model in self.routes:
            return self.routes[model]
        provider, sep, upstream_model = model.partition("/")
        if sep and provider in self.providers:
            return [Target(provider=provider, model=upstream_model)]
        raise NoRouteError(f"unknown model or route: {model!r}")

    def _backoff_s(self, attempt: int) -> float:
        # "Full jitter" exponential backoff: spreads retries out so clients don't stampede together.
        cap = min(self.cfg.backoff_max_ms, self.cfg.backoff_base_ms * 2**attempt)
        return random.uniform(0, cap) / 1000

    async def _run(
        self,
        model: str,
        req: ProviderRequest,
        call: Callable[[Provider, ProviderRequest], Awaitable],
    ) -> tuple[Target, object, list[Attempt]]:
        attempts: list[Attempt] = []
        for target in self.resolve(model):
            provider = self.providers[target.provider]
            if not provider.available:
                attempts.append(Attempt(target.provider, target.model, "provider not configured"))
                continue
            breaker = self.breakers[target.provider]
            target_req = replace(req, model=target.model)
            for attempt in range(self.cfg.max_retries + 1):
                if not breaker.allow():
                    attempts.append(Attempt(target.provider, target.model, "circuit open"))
                    self.metrics.upstream.inc(provider=target.provider, outcome="short_circuited")
                    break
                try:
                    result = await call(provider, target_req)
                except ProviderError as e:
                    attempts.append(Attempt(target.provider, target.model, str(e)))
                    self.metrics.upstream.inc(provider=target.provider, outcome="error")
                    if not e.retryable:
                        # The upstream answered (e.g. 400) so it is healthy; the request is the problem.
                        breaker.record_success()
                        break
                    breaker.record_failure()
                    if attempt < self.cfg.max_retries:
                        await self._sleep(self._backoff_s(attempt))
                    continue
                breaker.record_success()
                self.metrics.upstream.inc(provider=target.provider, outcome="ok")
                return target, result, attempts
        raise AllTargetsFailed(attempts)

    async def complete(self, model: str, req: ProviderRequest) -> tuple[Completion, list[Attempt]]:
        target, completion, attempts = await self._run(model, req, lambda p, r: p.complete(r))
        completion.provider, completion.model = target.provider, target.model
        return completion, attempts

    async def stream(
        self, model: str, req: ProviderRequest
    ) -> tuple[Target, AsyncIterator[StreamEvent], list[Attempt]]:
        """
        Fallback is only possible until the first event arrives; after that the
        client has already seen output, so a mid-stream failure is surfaced.
        """

        async def open_stream(provider: Provider, r: ProviderRequest):
            gen = provider.stream(r)
            try:
                first = await anext(gen)
            except StopAsyncIteration:
                raise ProviderError(f"{provider.name}: empty stream") from None
            except BaseException:
                await gen.aclose()
                raise
            return provider, first, gen

        target, (provider, first, gen), attempts = await self._run(model, req, open_stream)
        breaker = self.breakers[target.provider]

        async def events() -> AsyncIterator[StreamEvent]:
            yield first
            try:
                async for event in gen:
                    yield event
            except ProviderError:
                breaker.record_failure()
                raise
            finally:
                await gen.aclose()

        return target, events(), attempts
