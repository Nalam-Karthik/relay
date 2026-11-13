import pytest
from fastapi.testclient import TestClient

from relay.app import create_app
from relay.config import Settings
from relay.providers import MockProvider


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_settings(**overrides) -> Settings:
    data = {
        "providers": {
            "primary": {"type": "mock", "latency_ms": 0},
            "backup": {"type": "mock", "latency_ms": 0},
        },
        "routes": {
            "default": {"targets": [
                {"provider": "primary", "model": "big"},
                {"provider": "backup", "model": "small"},
            ]},
        },
        "pricing": {"big": {"input_per_mtok": 10, "output_per_mtok": 30},
                    "small": {"input_per_mtok": 1, "output_per_mtok": 2}},
        "tenants": [
            {"name": "acme", "api_key": "k-acme", "requests_per_minute": 1000, "tokens_per_minute": 10**7,
             "admin": True},
            {"name": "tiny", "api_key": "k-tiny", "requests_per_minute": 2, "tokens_per_minute": 10**6,
             "allowed_models": ["default"]},
            {"name": "broke", "api_key": "k-broke", "monthly_budget_usd": 0.0},
        ],
        "resilience": {"max_retries": 1, "backoff_base_ms": 0, "backoff_max_ms": 0,
                       "breaker_failure_threshold": 3, "breaker_reset_s": 30},
        "db_path": ":memory:",
    }
    data.update(overrides)
    return Settings.model_validate(data)


@pytest.fixture
def providers() -> dict[str, MockProvider]:
    return {"primary": MockProvider("primary", latency_ms=0), "backup": MockProvider("backup", latency_ms=0)}


@pytest.fixture
def client(providers):
    with TestClient(create_app(make_settings(), providers)) as c:
        yield c


def chat(client, content="hello there", key="k-acme", **body):
    payload = {"model": "default", "messages": [{"role": "user", "content": content}], **body}
    return client.post("/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {key}"})
