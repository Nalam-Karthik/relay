"""Two-tier response cache: exact match, then semantic (embedding) match."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from .config import CacheConfig
from .embeddings import Embedder
from .providers import Completion


@dataclass
class _Entry:
    key: str
    partition: str
    vector: list[float] | None
    numbers: tuple[str, ...]
    completion: Completion
    expires_at: float


@dataclass
class CacheLookup:
    key: str
    partition: str
    vector: list[float] | None
    numbers: tuple[str, ...] = ()
    hit: Completion | None = None
    kind: str = "miss"  # "miss" | "exact" | "semantic"
    score: float = 0.0


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(sorted(_NUMBER_RE.findall(text)))


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ResponseCache:
    """
    Exact tier: sha256 over (model, max_tokens, full conversation).

    Semantic tier: entries are partitioned by (model, max_tokens, every message
    except the last). Inside a partition we compare only the final user turn by
    cosine similarity. Partitioning means a near-duplicate question is only
    reused when the surrounding context is identical, which is what keeps the
    semantic tier from answering a question asked in a different conversation.

    Similarity alone is not enough: "top 10 risks" and "top 5 risks" embed
    almost identically but need different answers. A semantic hit therefore
    also requires the prompts to contain exactly the same numbers.

    Both tiers share one LRU + TTL budget.
    """

    def __init__(self, cfg: CacheConfig, embedder: Embedder, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.embedder = embedder
        self._clock = clock
        self._lru: OrderedDict[str, _Entry] = OrderedDict()
        self._partitions: dict[str, dict[str, _Entry]] = {}

    def __len__(self) -> int:
        return len(self._lru)

    def _remove(self, entry: _Entry) -> None:
        self._lru.pop(entry.key, None)
        part = self._partitions.get(entry.partition)
        if part is not None:
            part.pop(entry.key, None)
            if not part:
                del self._partitions[entry.partition]

    async def lookup(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> CacheLookup:
        now = self._clock()
        key = _digest([model, max_tokens, messages])
        partition = _digest([model, max_tokens, messages[:-1]])
        result = CacheLookup(key, partition, None)

        entry = self._lru.get(key)
        if entry is not None:
            if entry.expires_at > now:
                self._lru.move_to_end(key)
                result.hit, result.kind, result.score = entry.completion, "exact", 1.0
                return result
            self._remove(entry)

        last = messages[-1]
        if not self.cfg.semantic_enabled or last["role"] != "user":
            return result

        result.vector = await self.embedder.embed(last["content"])
        result.numbers = _numbers(last["content"])
        best: _Entry | None = None
        best_score = -1.0
        for candidate in list(self._partitions.get(partition, {}).values()):
            if candidate.expires_at <= now:
                self._remove(candidate)
                continue
            if candidate.numbers != result.numbers:
                continue
            score = sum(a * b for a, b in zip(result.vector, candidate.vector or ()))
            if score > best_score:
                best, best_score = candidate, score
        if best is not None and best_score >= self.cfg.similarity_threshold:
            self._lru.move_to_end(best.key)
            result.hit, result.kind, result.score = best.completion, "semantic", best_score
        return result

    def store(self, lookup: CacheLookup, completion: Completion) -> None:
        if completion.finish_reason not in ("stop", "length") or not completion.text:
            return
        if lookup.key in self._lru:
            self._remove(self._lru[lookup.key])
        entry = _Entry(
            lookup.key, lookup.partition, lookup.vector, lookup.numbers, completion, self._clock() + self.cfg.ttl_s
        )
        self._lru[lookup.key] = entry
        if lookup.vector is not None:
            self._partitions.setdefault(lookup.partition, {})[lookup.key] = entry
        while len(self._lru) > self.cfg.max_entries:
            self._remove(next(iter(self._lru.values())))
