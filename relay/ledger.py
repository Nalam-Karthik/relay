"""Durable usage ledger (SQLite) backing cost reports and budget enforcement."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import astuple, dataclass
from datetime import datetime, timezone


@dataclass
class UsageRecord:
    ts: float
    request_id: str
    tenant: str
    route: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    saved_usd: float
    latency_ms: float
    cache: str
    status: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    ts            REAL NOT NULL,
    request_id    TEXT NOT NULL,
    tenant        TEXT NOT NULL,
    route         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    saved_usd     REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    cache         TEXT NOT NULL,
    status        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_tenant_ts ON usage (tenant, ts);
"""


def month_start(now: float | None = None) -> float:
    dt = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


class Ledger:
    """Synchronous and thread-safe; the gateway calls it via asyncio.to_thread."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def record(self, rec: UsageRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", astuple(rec))

    def spend_since(self, tenant: str, since: float) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE tenant = ? AND ts >= ?", (tenant, since)
            ).fetchone()
        return row[0]

    def summary(self, tenant: str, since: float) -> dict:
        with self._lock:
            totals = self._conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(saved_usd),0),
                          COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                          COALESCE(SUM(cache != 'miss' AND cache != 'bypass'),0), COALESCE(AVG(latency_ms),0)
                   FROM usage WHERE tenant = ? AND ts >= ?""",
                (tenant, since),
            ).fetchone()
            by_model = self._conn.execute(
                """SELECT provider, model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)
                   FROM usage WHERE tenant = ? AND ts >= ? AND provider != ''
                   GROUP BY provider, model ORDER BY SUM(cost_usd) DESC""",
                (tenant, since),
            ).fetchall()
        requests, cost, saved, tin, tout, hits, avg_latency = totals
        return {
            "requests": requests,
            "cost_usd": round(cost, 6),
            "cache_saved_usd": round(saved, 6),
            "cache_hit_rate": round(hits / requests, 4) if requests else 0.0,
            "input_tokens": tin,
            "output_tokens": tout,
            "avg_latency_ms": round(avg_latency, 2),
            "by_model": [
                {"provider": p, "model": m, "requests": n, "input_tokens": i, "output_tokens": o,
                 "cost_usd": round(c, 6)}
                for p, m, n, i, o, c in by_model
            ],
        }

    def close(self) -> None:
        self._conn.close()
