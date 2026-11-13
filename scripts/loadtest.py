"""Fire concurrent traffic at a running gateway and report latency, cache and failure stats.

    python scripts/loadtest.py --requests 2000 --concurrency 50 --unique 200
"""

import argparse
import asyncio
import random
import statistics
import time
from collections import Counter

import httpx

QUESTIONS = [
    "How do I reverse a linked list?", "Explain the CAP theorem", "What is a bloom filter?",
    "Compare Raft and Paxos", "When should I use a B-tree index?", "What does idempotency mean in APIs?",
]


def make_prompt(i: int, unique: int, rng: random.Random) -> str:
    base = f"{QUESTIONS[i % len(QUESTIONS)]} (variant {i % unique})"
    # A share of traffic re-asks with different casing/punctuation, which the semantic tier catches.
    return base.upper() + "!" if rng.random() < 0.2 else base


async def main(args: argparse.Namespace) -> None:
    rng = random.Random(7)
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    cache: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, timeout=30, headers={"Authorization": f"Bearer {args.key}"}) as c:
        async def one(i: int) -> None:
            prompt = make_prompt(rng.randrange(args.unique * 3), args.unique, rng)
            async with sem:
                t0 = time.perf_counter()
                r = await c.post("/v1/chat/completions", json={
                    "model": args.model, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}],
                })
                latencies.append((time.perf_counter() - t0) * 1000)
            statuses[str(r.status_code)] += 1
            if r.status_code == 200:
                cache[r.headers["x-relay-cache"]] += 1
                providers[r.headers["x-relay-provider"]] += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        elapsed = time.perf_counter() - t0
        usage = (await c.get("/v1/usage")).json()

    q = statistics.quantiles(latencies, n=100)
    print(f"requests     {args.requests} in {elapsed:.2f}s  ({args.requests / elapsed:.0f} req/s)")
    print(f"latency ms   p50={q[49]:.1f}  p95={q[94]:.1f}  p99={q[98]:.1f}")
    print(f"status       {dict(statuses)}")
    print(f"cache        {dict(cache)}")
    print(f"served by    {dict(providers)}")
    print(f"spend        ${usage['cost_usd']:.4f}  saved by cache ${usage['cache_saved_usd']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--key", default="relay-dev-bench")
    p.add_argument("--model", default="default")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--unique", type=int, default=100)
    asyncio.run(main(p.parse_args()))
