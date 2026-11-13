"""OpenAI-compatible request shapes, so existing SDKs can point at the gateway unchanged."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=32_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False
    user: str | None = None
