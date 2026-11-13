"""Anthropic Messages API adapter (translates from the OpenAI chat shape)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from .base import (
    Completion,
    Provider,
    ProviderError,
    ProviderRequest,
    StreamDelta,
    StreamEnd,
    StreamEvent,
    is_retryable_status,
)

_FINISH_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


class AnthropicProvider(Provider):
    API_VERSION = "2023-06-01"

    def __init__(self, name: str, api_key: str | None, base_url: str | None = None, timeout_s: float = 60.0):
        super().__init__(name)
        self._client = (
            httpx.AsyncClient(
                base_url=base_url or "https://api.anthropic.com",
                timeout=timeout_s,
                headers={"x-api-key": api_key, "anthropic-version": self.API_VERSION},
            )
            if api_key
            else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    def _payload(self, req: ProviderRequest, stream: bool = False) -> dict:
        # Anthropic takes system prompts as a top-level field rather than a message role.
        system = "\n\n".join(m["content"] for m in req.messages if m["role"] == "system")
        body: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [m for m in req.messages if m["role"] != "system"],
        }
        if system:
            body["system"] = system
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if stream:
            body["stream"] = True
        return body

    def _http_error(self, status: int, body: str) -> ProviderError:
        return ProviderError(
            f"{self.name}: HTTP {status}: {body[:300]}", retryable=is_retryable_status(status), status_code=status
        )

    async def complete(self, req: ProviderRequest) -> Completion:
        try:
            r = await self._client.post("/v1/messages", json=self._payload(req))
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: transport error: {e!r}") from e
        if r.status_code != 200:
            raise self._http_error(r.status_code, r.text)
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            finish_reason=_FINISH_REASONS.get(data.get("stop_reason"), "stop"),
            provider=self.name,
            model=req.model,
        )

    async def stream(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        input_tokens = output_tokens = 0
        finish = "stop"
        try:
            async with self._client.stream("POST", "/v1/messages", json=self._payload(req, stream=True)) as r:
                if r.status_code != 200:
                    raise self._http_error(r.status_code, (await r.aread()).decode(errors="replace"))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    kind = event.get("type")
                    if kind == "message_start":
                        input_tokens = event["message"].get("usage", {}).get("input_tokens", 0)
                    elif kind == "content_block_delta" and event["delta"].get("type") == "text_delta":
                        yield StreamDelta(event["delta"]["text"])
                    elif kind == "message_delta":
                        output_tokens = event.get("usage", {}).get("output_tokens", output_tokens)
                        finish = _FINISH_REASONS.get(event.get("delta", {}).get("stop_reason"), finish)
                    elif kind == "error":
                        raise ProviderError(f"{self.name}: stream error: {event.get('error')}")
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: transport error: {e!r}") from e
        yield StreamEnd(input_tokens, output_tokens, finish)

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
