"""Adapter for any OpenAI-compatible server: OpenAI itself, vLLM, Ollama, LM Studio, etc."""

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
    estimate_tokens,
    is_retryable_status,
)


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        requires_key: bool = False,
        timeout_s: float = 60.0,
    ):
        super().__init__(name)
        # Local servers (Ollama, vLLM) usually need no key; hosted ones do.
        self._configured = bool(api_key) or not requires_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, headers=headers)

    @property
    def available(self) -> bool:
        return self._configured

    def _payload(self, req: ProviderRequest, stream: bool = False) -> dict:
        body: dict = {"model": req.model, "messages": req.messages, "max_tokens": req.max_tokens}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _http_error(self, status: int, body: str) -> ProviderError:
        return ProviderError(
            f"{self.name}: HTTP {status}: {body[:300]}", retryable=is_retryable_status(status), status_code=status
        )

    async def complete(self, req: ProviderRequest) -> Completion:
        try:
            r = await self._client.post("/chat/completions", json=self._payload(req))
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: transport error: {e!r}") from e
        if r.status_code != 200:
            raise self._http_error(r.status_code, r.text)
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        text = choice["message"].get("content") or ""
        return Completion(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", estimate_tokens(text)),
            finish_reason=choice.get("finish_reason") or "stop",
            provider=self.name,
            model=req.model,
        )

    async def stream(self, req: ProviderRequest) -> AsyncIterator[StreamEvent]:
        parts: list[str] = []
        usage: dict = {}
        finish = "stop"
        try:
            async with self._client.stream("POST", "/chat/completions", json=self._payload(req, stream=True)) as r:
                if r.status_code != 200:
                    raise self._http_error(r.status_code, (await r.aread()).decode(errors="replace"))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        if text := choice.get("delta", {}).get("content"):
                            parts.append(text)
                            yield StreamDelta(text)
                        finish = choice.get("finish_reason") or finish
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: transport error: {e!r}") from e
        prompt = " ".join(m["content"] for m in req.messages)
        yield StreamEnd(
            usage.get("prompt_tokens", estimate_tokens(prompt)),
            usage.get("completion_tokens", estimate_tokens("".join(parts))),
            finish,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
