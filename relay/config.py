"""Typed gateway configuration, loaded from YAML."""

from __future__ import annotations

import os
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    type: Literal["mock", "anthropic", "openai"]
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    # Mock-only knobs, used to simulate latency and flaky upstreams.
    latency_ms: int = 30
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int | None = None


class Target(BaseModel):
    provider: str
    model: str


class Route(BaseModel):
    """An ordered fallback chain: the first healthy target serves the request."""

    targets: list[Target] = Field(min_length=1)


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class TenantConfig(BaseModel):
    name: str
    api_key: str
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    monthly_budget_usd: float | None = None
    allowed_models: list[str] | None = None
    admin: bool = False


class EmbedderConfig(BaseModel):
    type: Literal["hashing", "openai"] = "hashing"
    dim: int = 1024
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: float = 3600.0
    max_entries: int = 5000
    semantic_enabled: bool = True
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    embedder: EmbedderConfig = EmbedderConfig()


class ResilienceConfig(BaseModel):
    max_retries: int = 2
    backoff_base_ms: float = 100.0
    backoff_max_ms: float = 2000.0
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0


class GuardrailsConfig(BaseModel):
    redact_pii: bool = True


class Settings(BaseModel):
    providers: dict[str, ProviderConfig]
    routes: dict[str, Route]
    pricing: dict[str, ModelPrice] = {}
    tenants: list[TenantConfig]
    cache: CacheConfig = CacheConfig()
    resilience: ResilienceConfig = ResilienceConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    db_path: str = "relay.db"


def load_settings(path: str | None = None) -> Settings:
    path = path or os.environ.get("RELAY_CONFIG", "config.yaml")
    with open(path) as f:
        return Settings.model_validate(yaml.safe_load(f))
