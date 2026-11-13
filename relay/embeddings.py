"""Embedders used by the semantic cache."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

import httpx

from .config import EmbedderConfig

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """
    Zero-dependency embedder using signed feature hashing over word unigrams,
    word bigrams and character trigrams, L2-normalised.

    This captures *lexical* similarity (rephrasings, casing, contractions,
    typos), not meaning. Swap in a model-backed embedder for true semantic
    matching; the cache does not care which one it gets.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = _WORD_RE.findall(text.lower())
        feats = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 1.0) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.5) for i in range(len(padded) - 2)]
        return feats

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat, weight in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAICompatibleEmbedder:
    """Calls a /embeddings endpoint (OpenAI, Ollama, vLLM, TEI...)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=10.0)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        r = await self._client.post("/embeddings", json={"model": self.model, "input": text})
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.type == "openai":
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        return OpenAICompatibleEmbedder(cfg.base_url or "https://api.openai.com/v1", cfg.model or "", api_key)
    return HashingEmbedder(cfg.dim)
