"""The request pipeline, independent of the HTTP framework.

    authenticate -> authorize model -> budget -> rate limit -> PII redaction
        -> cache lookup -> route (retry / fallback / breaker) -> cache store
        -> usage ledger + metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .cache import CacheLookup, ResponseCache
from .config import Settings, TenantConfig
from .embeddings import Embedder, build_embedder
from .guardrails import redact
from .ledger import Ledger, UsageRecord, month_start
from .metrics import Metrics
from .providers import Completion, Provider, ProviderRequest, StreamDelta, StreamEnd
from .providers.base import estimate_tokens
from .ratelimit import RateLimiter
from .routing import AllTargetsFailed, Attempt, NoRouteError, Router
from .schemas import ChatCompletionRequest

log = logging.getLogger("relay")


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


@dataclass
class Outcome:
    completion: Completion
    cache: str  # "exact" | "semantic" | "miss" | "bypass"
    attempts: list[Attempt] = field(default_factory=list)


@dataclass
class StreamOutcome:
    provider: str
    model: str
    cache: str
    attempts: list[Attempt]
    events: AsyncIterator[StreamDelta | StreamEnd]


class Gateway:
    def __init__(self, settings: Settings, providers: dict[str, Provider], embedder: Embedder | None = None):
        self.settings = settings
        self.providers = providers
        self.metrics = Metrics()
        self.router = Router(
            {name: route.targets for name, route in settings.routes.items()},
            providers,
            settings.resilience,
            self.metrics,
        )
        self.cache = (
            ResponseCache(settings.cache, embedder or build_embedder(settings.cache.embedder))
            if settings.cache.enabled
            else None
        )
        self.limiter = RateLimiter()
        self.ledger = Ledger(settings.db_path)
        self._tenants = {t.api_key: t for t in settings.tenants}
        self.metrics.register_gauge(self._breaker_gauges)

    # ---- admission -------------------------------------------------------

    def authenticate(self, authorization: str | None) -> TenantConfig:
        scheme, _, token = (authorization or "").partition(" ")
        tenant = self._tenants.get(token.strip()) if scheme.lower() == "bearer" else None
        if tenant is None:
            raise GatewayError(401, "invalid_api_key", "missing or invalid API key")
        return tenant

    async def _admit(self, tenant: TenantConfig, body: ChatCompletionRequest) -> list[dict[str, str]]:
        if tenant.allowed_models is not None and body.model not in tenant.allowed_models:
            raise GatewayError(403, "model_not_allowed", f"tenant may not use {body.model!r}")
        try:
            self.router.resolve(body.model)
        except NoRouteError as e:
            raise GatewayError(404, "model_not_found", str(e)) from None

        if tenant.monthly_budget_usd is not None:
            spent = await asyncio.to_thread(self.ledger.spend_since, tenant.name, month_start())
            if spent >= tenant.monthly_budget_usd:
                raise GatewayError(
                    402, "budget_exceeded", f"monthly budget of ${tenant.monthly_budget_usd:.2f} exhausted"
                )

        if wait := self.limiter.check(tenant):
            self.metrics.rate_limited.inc(tenant=tenant.name)
            raise GatewayError(
                429, "rate_limited", "rate limit exceeded", {"Retry-After": str(math.ceil(wait))}
            )

        messages = [m.model_dump() for m in body.messages]
        if self.settings.guardrails.redact_pii:
            for m in messages:
                m["content"], counts = redact(m["content"])
                for kind, n in counts.items():
                    self.metrics.redactions.inc(n, kind=kind)
        return messages

    async def _cache_lookup(self, body: ChatCompletionRequest, messages: list[dict]) -> CacheLookup | None:
        # Sampling with temperature > 0 asks for variety; replaying a cached answer would defeat that.
        if self.cache is None or (body.temperature or 0) > 0:
            return None
        return await self.cache.lookup(body.model, messages, body.max_tokens)

    # ---- execution -------------------------------------------------------

    async def complete(self, tenant: TenantConfig, body: ChatCompletionRequest, request_id: str) -> Outcome:
        started = time.perf_counter()
        messages = await self._admit(tenant, body)
        lookup = await self._cache_lookup(body, messages)
        if lookup and lookup.hit:
            outcome = Outcome(lookup.hit, lookup.kind)
        else:
            req = ProviderRequest(body.model, messages, body.max_tokens, body.temperature)
            try:
                completion, attempts = await self.router.complete(body.model, req)
            except AllTargetsFailed as e:
                await self._account(tenant, body, None, "error", e.attempts, request_id, started)
                raise GatewayError(502, "upstream_unavailable", str(e)) from None
            if lookup:
                self.cache.store(lookup, completion)
            outcome = Outcome(completion, "miss" if lookup else "bypass", attempts)
        await self._account(tenant, body, outcome.completion, outcome.cache, outcome.attempts, request_id, started)
        return outcome

    async def stream(self, tenant: TenantConfig, body: ChatCompletionRequest, request_id: str) -> StreamOutcome:
        started = time.perf_counter()
        messages = await self._admit(tenant, body)
        lookup = await self._cache_lookup(body, messages)

        if lookup and lookup.hit:
            hit = lookup.hit

            async def replay() -> AsyncIterator[StreamDelta | StreamEnd]:
                try:
                    yield StreamDelta(hit.text)
                    yield StreamEnd(hit.input_tokens, hit.output_tokens, hit.finish_reason)
                finally:
                    await self._account(tenant, body, hit, lookup.kind, [], request_id, started)

            return StreamOutcome(hit.provider, hit.model, lookup.kind, [], replay())

        req = ProviderRequest(body.model, messages, body.max_tokens, body.temperature)
        try:
            target, upstream, attempts = await self.router.stream(body.model, req)
        except AllTargetsFailed as e:
            await self._account(tenant, body, None, "error", e.attempts, request_id, started)
            raise GatewayError(502, "upstream_unavailable", str(e)) from None

        cache_state = "miss" if lookup else "bypass"

        async def relay() -> AsyncIterator[StreamDelta | StreamEnd]:
            parts: list[str] = []
            end: StreamEnd | None = None
            try:
                async for event in upstream:
                    if isinstance(event, StreamDelta):
                        parts.append(event.text)
                    else:
                        end = event
                    yield event
            finally:
                # Runs on success, upstream error and client disconnect alike, so usage is never lost.
                text = "".join(parts)
                completion = Completion(
                    text,
                    end.input_tokens if end else estimate_tokens(" ".join(m["content"] for m in messages)),
                    end.output_tokens if end else estimate_tokens(text),
                    end.finish_reason if end else "error",
                    target.provider,
                    target.model,
                )
                if end and lookup:
                    self.cache.store(lookup, completion)
                await self._account(
                    tenant, body, completion, cache_state if end else "error", attempts, request_id, started
                )

        return StreamOutcome(target.provider, target.model, cache_state, attempts, relay())

    # ---- accounting ------------------------------------------------------

    def price(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.settings.pricing.get(f"{provider}/{model}") or self.settings.pricing.get(model)
        if p is None:
            return 0.0
        return (input_tokens * p.input_per_mtok + output_tokens * p.output_per_mtok) / 1_000_000

    async def _account(
        self,
        tenant: TenantConfig,
        body: ChatCompletionRequest,
        completion: Completion | None,
        cache: str,
        attempts: list[Attempt],
        request_id: str,
        started: float,
    ) -> None:
        latency_s = time.perf_counter() - started
        status = "error" if completion is None or cache == "error" else "ok"
        cached = cache in ("exact", "semantic")
        tin = completion.input_tokens if completion else 0
        tout = completion.output_tokens if completion else 0
        provider = completion.provider if completion else ""
        model = completion.model if completion else ""
        cost = 0.0 if cached or completion is None else self.price(provider, model, tin, tout)
        saved = self.price(provider, model, tin, tout) if cached else 0.0

        if not cached and completion is not None:
            self.limiter.record_tokens(tenant, tin + tout)
            self.metrics.tokens.inc(tin, tenant=tenant.name, direction="input")
            self.metrics.tokens.inc(tout, tenant=tenant.name, direction="output")
            self.metrics.cost.inc(cost, tenant=tenant.name)
        if saved:
            self.metrics.saved.inc(saved, tenant=tenant.name)
        if completion is not None and attempts:
            self.metrics.degraded.inc(route=body.model)
        self.metrics.requests.inc(route=body.model, status=status, cache=cache)
        self.metrics.latency.observe(latency_s, route=body.model, cache=cache)

        record = UsageRecord(
            time.time(), request_id, tenant.name, body.model, provider, model,
            tin, tout, cost, saved, latency_s * 1000, cache, status,
        )
        await asyncio.to_thread(self.ledger.record, record)
        log.info(json.dumps({"event": "request", **record.__dict__, "attempts": len(attempts) + 1}))

    # ---- introspection ---------------------------------------------------

    def provider_status(self) -> dict:
        return {
            name: {"available": p.available, **self.router.breakers[name].snapshot()}
            for name, p in self.providers.items()
        }

    def _breaker_gauges(self) -> list[str]:
        name = "relay_circuit_open"
        lines = [f"# HELP {name} 1 if the provider's circuit breaker is not closed", f"# TYPE {name} gauge"]
        for provider, breaker in self.router.breakers.items():
            lines.append(f'{name}{{provider="{provider}"}} {0 if breaker.state.value == "closed" else 1}')
        return lines

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
        self.ledger.close()

