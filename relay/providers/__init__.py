import os

from ..config import Settings
from .anthropic import AnthropicProvider
from .base import Completion, Provider, ProviderError, ProviderRequest, StreamDelta, StreamEnd, StreamEvent
from .mock import MockProvider
from .openai import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider",
    "Completion",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "ProviderRequest",
    "StreamDelta",
    "StreamEnd",
    "StreamEvent",
    "build_providers",
]


def build_providers(settings: Settings) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    for name, cfg in settings.providers.items():
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        match cfg.type:
            case "mock":
                providers[name] = MockProvider(name, cfg.latency_ms, cfg.failure_rate, cfg.seed)
            case "anthropic":
                providers[name] = AnthropicProvider(name, api_key, cfg.base_url, cfg.timeout_s)
            case "openai":
                providers[name] = OpenAICompatibleProvider(
                    name,
                    cfg.base_url or "https://api.openai.com/v1",
                    api_key,
                    requires_key=cfg.api_key_env is not None,
                    timeout_s=cfg.timeout_s,
                )
    return providers
