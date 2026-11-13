"""Minimal Prometheus text-format metrics (no client library needed)."""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Callable, Iterable


def _fmt_labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{n}="{v}"' for n, v in zip(names, values)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class Counter:
    def __init__(self, name: str, help: str, labels: Iterable[str] = ()):
        self.name, self.help, self.labelnames = name, help, tuple(labels)
        self._values: dict[tuple[str, ...], float] = defaultdict(float)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        self._values[tuple(str(labels.get(n, "")) for n in self.labelnames)] += amount

    def value(self, **labels: str) -> float:
        return self._values.get(tuple(str(labels.get(n, "")) for n in self.labelnames), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        lines += [f"{self.name}{_fmt_labels(self.labelnames, k)} {v}" for k, v in sorted(self._values.items())]
        return lines


class Histogram:
    DEFAULT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

    def __init__(self, name: str, help: str, labels: Iterable[str] = (), buckets: tuple[float, ...] = DEFAULT_BUCKETS):
        self.name, self.help, self.labelnames, self.buckets = name, help, tuple(labels), buckets
        self._counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = defaultdict(float)

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(str(labels.get(n, "")) for n in self.labelnames)
        counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
        counts[bisect.bisect_left(self.buckets, value)] += 1
        self._sums[key] += value

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self._counts.items()):
            cumulative = 0
            for bound, c in zip((*self.buckets, float("inf")), counts):
                cumulative += c
                le = 'le="+Inf"' if bound == float("inf") else f'le="{bound}"'
                lines.append(f"{self.name}_bucket{_fmt_labels(self.labelnames, key, le)} {cumulative}")
            lines.append(f"{self.name}_sum{_fmt_labels(self.labelnames, key)} {self._sums[key]}")
            lines.append(f"{self.name}_count{_fmt_labels(self.labelnames, key)} {cumulative}")
        return lines


class Metrics:
    def __init__(self) -> None:
        self.requests = Counter("relay_requests_total", "Requests handled", ("route", "status", "cache"))
        self.latency = Histogram("relay_request_duration_seconds", "End-to-end request latency", ("route", "cache"))
        self.upstream = Counter("relay_upstream_calls_total", "Calls to upstream providers", ("provider", "outcome"))
        self.degraded = Counter(
            "relay_degraded_requests_total", "Requests that succeeded only after a retry or fallback", ("route",)
        )
        self.tokens = Counter("relay_tokens_total", "LLM tokens processed", ("tenant", "direction"))
        self.cost = Counter("relay_cost_usd_total", "Upstream spend in USD", ("tenant",))
        self.saved = Counter("relay_cache_saved_usd_total", "Spend avoided by cache hits", ("tenant",))
        self.rate_limited = Counter("relay_rate_limited_total", "Requests rejected by rate limits", ("tenant",))
        self.redactions = Counter("relay_pii_redactions_total", "PII spans redacted from prompts", ("kind",))
        self._gauges: list[Callable[[], list[str]]] = []

    def register_gauge(self, fn: Callable[[], list[str]]) -> None:
        self._gauges.append(fn)

    def render(self) -> str:
        lines: list[str] = []
        for m in (self.requests, self.latency, self.upstream, self.degraded, self.tokens, self.cost, self.saved,
                  self.rate_limited, self.redactions):
            lines += m.render()
        for fn in self._gauges:
            lines += fn()
        return "\n".join(lines) + "\n"
