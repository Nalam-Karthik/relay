"""Unit tests for the gateway's building blocks."""

import pytest

from relay.cache import ResponseCache
from relay.circuit_breaker import CircuitBreaker, State
from relay.config import CacheConfig, ResilienceConfig, Target, TenantConfig
from relay.embeddings import HashingEmbedder
from relay.guardrails import redact
from relay.metrics import Metrics
from relay.providers import Completion, MockProvider, ProviderError, ProviderRequest
from relay.ratelimit import RateLimiter, TokenBucket
from relay.routing import AllTargetsFailed, NoRouteError, Router


# ---- token bucket / rate limiter -------------------------------------------

def test_token_bucket_refills_over_time(clock):
    bucket = TokenBucket(capacity=2, refill_per_s=1, clock=clock)
    assert bucket.try_acquire() == 0
    assert bucket.try_acquire() == 0
    assert bucket.try_acquire() == pytest.approx(1.0)
    clock.advance(0.5)
    assert bucket.try_acquire() == pytest.approx(0.5)
    clock.advance(0.5)
    assert bucket.try_acquire() == 0


def test_token_bucket_never_exceeds_capacity(clock):
    bucket = TokenBucket(capacity=3, refill_per_s=10, clock=clock)
    clock.advance(100)
    bucket.try_acquire(0)
    assert bucket.tokens == 3


def test_rate_limiter_blocks_tenant_in_token_debt(clock):
    tenant = TenantConfig(name="t", api_key="k", requests_per_minute=100, tokens_per_minute=600)
    limiter = RateLimiter(clock)
    assert limiter.check(tenant) == 0
    limiter.record_tokens(tenant, 1200)  # one huge response puts the tenant 600 tokens into debt
    wait = limiter.check(tenant)
    assert wait == pytest.approx(601 / 10)  # 10 tokens/s refill
    clock.advance(wait)
    assert limiter.check(tenant) == 0


def test_rate_limiter_isolates_tenants(clock):
    a = TenantConfig(name="a", api_key="ka", requests_per_minute=1)
    b = TenantConfig(name="b", api_key="kb", requests_per_minute=1)
    limiter = RateLimiter(clock)
    assert limiter.check(a) == 0
    assert limiter.check(a) > 0
    assert limiter.check(b) == 0


# ---- circuit breaker -------------------------------------------------------

def test_breaker_lifecycle(clock):
    br = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=clock)
    br.record_failure()
    assert br.allow() and br.state is State.CLOSED
    br.record_failure()
    assert br.state is State.OPEN and not br.allow()

    clock.advance(10)
    assert br.allow() and br.state is State.HALF_OPEN
    assert not br.allow(), "only one probe at a time while half-open"
    br.record_failure()
    assert br.state is State.OPEN

    clock.advance(10)
    assert br.allow()
    br.record_success()
    assert br.state is State.CLOSED and br.allow()


def test_breaker_success_resets_failure_streak(clock):
    br = CircuitBreaker(failure_threshold=2, clock=clock)
    br.record_failure()
    br.record_success()
    br.record_failure()
    assert br.state is State.CLOSED


def test_breaker_recovers_from_abandoned_probe(clock):
    br = CircuitBreaker(failure_threshold=1, reset_timeout_s=5, clock=clock)
    br.record_failure()
    clock.advance(5)
    assert br.allow()  # probe starts but never reports (e.g. client hung up)
    assert not br.allow()
    clock.advance(5)
    assert br.allow()


# ---- router ----------------------------------------------------------------

class ScriptedProvider(MockProvider):
    """Fails with the given errors in order, then succeeds."""

    def __init__(self, name, errors=()):
        super().__init__(name, latency_ms=0)
        self.errors = list(errors)

    async def complete(self, req):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return Completion("ok", 1, 1, provider=self.name, model=req.model)


def make_router(providers, retries=1, threshold=3):
    async def no_sleep(_):
        pass

    routes = {"r": [Target(provider=name, model=f"{name}-model") for name in providers]}
    cfg = ResilienceConfig(max_retries=retries, breaker_failure_threshold=threshold)
    return Router(routes, providers, cfg, Metrics(), sleep=no_sleep)


REQ = ProviderRequest("r", [{"role": "user", "content": "hi"}], 16)


async def test_router_retries_then_succeeds():
    p = ScriptedProvider("a", [ProviderError("503")])
    completion, attempts = await make_router({"a": p}).complete("r", REQ)
    assert completion.provider == "a" and completion.model == "a-model"
    assert p.calls == 2 and len(attempts) == 1


async def test_router_falls_back_after_exhausting_retries():
    a = ScriptedProvider("a", [ProviderError("503")] * 5)
    b = ScriptedProvider("b")
    completion, attempts = await make_router({"a": a, "b": b}, retries=1).complete("r", REQ)
    assert completion.provider == "b"
    assert a.calls == 2 and len(attempts) == 2


async def test_router_does_not_retry_client_errors():
    a = ScriptedProvider("a", [ProviderError("400 bad request", retryable=False)])
    b = ScriptedProvider("b")
    router = make_router({"a": a, "b": b}, retries=3)
    completion, _ = await router.complete("r", REQ)
    assert a.calls == 1 and completion.provider == "b"
    assert router.breakers["a"].consecutive_failures == 0


async def test_open_circuit_skips_provider_without_calling_it():
    a = ScriptedProvider("a", [ProviderError("503")] * 100)
    b = ScriptedProvider("b")
    router = make_router({"a": a, "b": b}, retries=0, threshold=2)
    for _ in range(2):
        await router.complete("r", REQ)
    calls_before = a.calls
    _, attempts = await router.complete("r", REQ)
    assert a.calls == calls_before
    assert attempts[0].error == "circuit open"


async def test_router_raises_when_everything_fails():
    router = make_router({"a": ScriptedProvider("a", [ProviderError("503")] * 9)}, retries=1)
    with pytest.raises(AllTargetsFailed) as exc:
        await router.complete("r", REQ)
    assert len(exc.value.attempts) == 2


def test_router_direct_provider_addressing():
    router = make_router({"a": ScriptedProvider("a")})
    assert router.resolve("a/custom-model") == [Target(provider="a", model="custom-model")]
    with pytest.raises(NoRouteError):
        router.resolve("nope/model")


async def test_stream_falls_back_before_first_token():
    a = MockProvider("a", latency_ms=0, failure_rate=1.0)
    b = MockProvider("b", latency_ms=0)
    target, events, attempts = await make_router({"a": a, "b": b}, retries=0).stream("r", REQ)
    text = "".join([e.text async for e in events if hasattr(e, "text")])
    assert target.provider == "b" and "You said: hi" in text and len(attempts) == 1


# ---- cache -----------------------------------------------------------------

def _msgs(*contents):
    return [{"role": "user", "content": c} for c in contents]


async def test_cache_exact_then_semantic_then_miss(clock):
    cache = ResponseCache(CacheConfig(similarity_threshold=0.9), HashingEmbedder(), clock)
    lookup = await cache.lookup("m", _msgs("What is the capital of France?"), 100)
    assert lookup.hit is None
    cache.store(lookup, Completion("Paris", 5, 1))

    exact = await cache.lookup("m", _msgs("What is the capital of France?"), 100)
    assert exact.kind == "exact" and exact.hit.text == "Paris"

    near = await cache.lookup("m", _msgs("what is the capital of france"), 100)
    assert near.kind == "semantic" and near.score > 0.99

    other = await cache.lookup("m", _msgs("What is the capital of Germany?"), 100)
    assert other.hit is None


async def test_semantic_cache_never_matches_prompts_with_different_numbers(clock):
    cache = ResponseCache(CacheConfig(similarity_threshold=0.9), HashingEmbedder(), clock)
    q = "Summarize the top 10 risks in this contract for a startup founder"
    cache.store(await cache.lookup("m", _msgs(q), 100), Completion("ten risks", 5, 1))
    swapped = await cache.lookup("m", _msgs(q.replace("10", "5")), 100)
    assert swapped.hit is None, "0.92 cosine similarity, but a different question"
    recased = await cache.lookup("m", _msgs(q.upper()), 100)
    assert recased.kind == "semantic"


async def test_cache_partitions_by_model_and_context(clock):
    cache = ResponseCache(CacheConfig(), HashingEmbedder(), clock)
    lookup = await cache.lookup("m", _msgs("hello"), 100)
    cache.store(lookup, Completion("hi", 1, 1))
    assert (await cache.lookup("other-model", _msgs("hello"), 100)).hit is None
    assert (await cache.lookup("m", _msgs("hello"), 50)).hit is None
    assert (await cache.lookup("m", _msgs("earlier turn", "hello"), 100)).hit is None


async def test_cache_ttl_and_lru_eviction(clock):
    cache = ResponseCache(CacheConfig(ttl_s=10, max_entries=2), HashingEmbedder(), clock)
    for q in ("one", "two", "three"):
        cache.store(await cache.lookup("m", _msgs(q), 10), Completion(q, 1, 1))
    assert len(cache) == 2
    assert (await cache.lookup("m", _msgs("one"), 10)).hit is None
    clock.advance(11)
    assert (await cache.lookup("m", _msgs("three"), 10)).hit is None


async def test_cache_does_not_store_failed_completions(clock):
    cache = ResponseCache(CacheConfig(), HashingEmbedder(), clock)
    cache.store(await cache.lookup("m", _msgs("q"), 10), Completion("partial", 1, 1, finish_reason="error"))
    assert len(cache) == 0


# ---- guardrails ------------------------------------------------------------

def test_redacts_common_pii():
    text = ("Mail jane.doe@example.com or call (415) 555-0132. Card 4111 1111 1111 1111, "
            "SSN 123-45-6789, host 10.0.0.12.")
    out, counts = redact(text)
    assert "example.com" not in out and "4111" not in out and "123-45-6789" not in out
    assert counts == {"EMAIL": 1, "PHONE": 1, "CREDIT_CARD": 1, "SSN": 1, "IPV4": 1}


def test_does_not_redact_non_luhn_numbers():
    out, counts = redact("order 1234 5678 9012 3456 shipped")
    assert "CREDIT_CARD" not in counts
