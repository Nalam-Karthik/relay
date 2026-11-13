"""Provider abstraction: every upstream speaks this small, normalized interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float | None = None


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class StreamDelta:
    text: str


@dataclass(frozen=True)
class StreamEnd:
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"


StreamEvent = StreamDelta | StreamEnd


class ProviderError(Exception):
    """An upstream failure. `retryable` separates outages from bad requests."""

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def is_retryable_status(status: int) -> bool:
    return status in (408, 409, 425, 429) or status >= 500


def estimate_tokens(text: str) -> int:
    # ~4 chars/token is the usual heuristic for English; only used when upstream omits usage.
    return max(1, len(text) // 4)


class Provider(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    def available(self) -> bool:
        """False when the provider is not configured (e.g. missing API key)."""
        return True

    @abstractmethod
    async def complete(self, req: ProviderRequest) -> Completion: ...

    @abstractmethod
    def stream(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None:
        pass
