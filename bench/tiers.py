"""Mixed-tier open-loop load for the priority scheduling experiments.

Runs two independent open-loop streams -- paid at one rate, free at another --
against the same server for a fixed duration, then reports each tier
separately. Open loop for the same reason as bench/burst.py: the point is to
offer more than the system can drain and see who gets served, who gets
refused, and who waits.

    # tiered admission: does overload shed free first?
    python -m bench.tiers --paid 14 --free 4 --duration 20

    # starvation: with aging off, do free requests in the queue ever get out?
    AGING_MS=0    ...   python -m bench.tiers --paid 10 --free 1 --duration 30
    AGING_MS=2000 ...   python -m bench.tiers --paid 10 --free 1 --duration 30
"""

import argparse
import asyncio
import csv
import os
import statistics
import time
from typing import Dict, List

import httpx

from bench.burst import percentile

DEFAULT_PROMPT = "Explain what priority scheduling is in one short paragraph."


class Rec:
    __slots__ = ("tier", "send_s", "latency_ms", "status", "queue_wait_ms",
                 "aged", "depth")

    def __init__(self, tier, send_s, latency_ms, status, queue_wait_ms, aged, depth):
        self.tier = tier
        self.send_s = send_s
        self.latency_ms = latency_ms
        self.status = status
        self.queue_wait_ms = queue_wait_ms
        self.aged = aged
        self.depth = depth


async def fire(client, url, payload, tier, t0, out: List[Rec]) -> None:
    send_s = time.perf_counter() - t0
    started = time.perf_counter()
    qwait = depth = float("nan")
    aged = False
    try:
        r = await client.post(url, json={**payload, "tier": tier})
        latency = (time.perf_counter() - started) * 1000.0
        status = r.status_code
        if status == 200:
            b = r.json()
            qwait = b.get("queue_wait_ms", float("nan"))
            aged = bool(b.get("aged", False))
            depth = b.get("queue_depth_at_enqueue", float("nan"))
        elif status == 503:
            try:
                d = r.json().get("detail", {})
                if isinstance(d, dict):
                    depth = d.get("queue_depth", float("nan"))
            except Exception:
                pass
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000.0
        status = "error:" + type(exc).__name__
    out.append(Rec(tier, send_s, latency, status, qwait, aged, depth))


async def stream(client, url, payload, tier, rate, duration, t0, out, tasks):
    """One open-loop arrival process for one tier."""
    if rate <= 0:
        return
    total = int(round(rate * duration))
    interval = 1.0 / rate
    for i in range(total):
        target = t0 + i * interval
        slack = target - time.perf_counter()
        if slack > 0:
            await asyncio.sleep(slack)
        tasks.append(asyncio.ensure_future(fire(client, url, payload, tier, t0, out)))


async def run(url, payload, paid_rate, free_rate, duration, timeout_s) -> List[Rec]:
    out: List[Rec] = []
    tasks: list = []
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(
            stream(client, url, payload, "paid", paid_rate, duration, t0, out, tasks),
            stream(client, url, payload, "free", free_rate, duration, t0, out, tasks),
        )
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    print(f"  offered {len(out)} requests over {duration:.0f}s, all responses in by {wall:.1f}s")
    return out


def tier_summary(recs: List[Rec], tier: str) -> Dict:
    mine = [r for r in recs if r.tier == tier]
    ok = [r for r in mine if r.status == 200]
    rej = [r for r in mine if r.status == 503]
    lat = [r.latency_ms for r in ok]
    wait = [r.queue_wait_ms for r in ok if r.queue_wait_ms == r.queue_wait_ms]
    return {
        "offered": len(mine),
        "accepted": len(ok),
        "rejected": len(rej),
        "errors": len(mine) - len(ok) - len(rej),
        "accept_pct": 100.0 * len(ok) / len(mine) if mine else float("nan"),
        "p50": percentile(lat, .5), "p95": percentile(lat, .95), "p99": percentile(lat, .99),
        "max": max(lat) if lat else float("nan"),
        "wait_p50": percentile(wait, .5), "wait_p99": percentile(wait, .99),
        "wait_max": max(wait) if wait else float("nan"),
        "aged": sum(1 for r in ok if r.aged),
    }


def report(recs: List[Rec], label: str) -> None:
    print()
    print(f"=== {label} ===")
    hdr = (f"  {'tier':<6} {'offered':>8} {'accepted':>9} {'rejected':>9} {'acc%':>6} "
           f"{'p50':>8} {'p99':>8} {'max':>8} {'wait p50':>9} {'wait p99':>9} {'wait max':>9} {'aged':>5}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for tier in ("paid", "free"):
        s = tier_summary(recs, tier)
        if s["offered"] == 0:
            continue
        print(f"  {tier:<6} {s['offered']:>8} {s['accepted']:>9} {s['rejected']:>9} {s['accept_pct']:>5.0f}% "
              f"{s['p50']:>8.0f} {s['p99']:>8.0f} {s['max']:>8.0f} "
              f"{s['wait_p50']:>9.0f} {s['wait_p99']:>9.0f} {s['wait_max']:>9.0f} {s['aged']:>5}")
    print()
    print("  latency/wait in ms. 'aged' = free requests promoted by aging before being served.")

    # Free-tier wait over time is the starvation evidence: growing = starving.
    free_ok = sorted((r for r in recs if r.tier == "free" and r.status == 200), key=lambda r: r.send_s)
    if free_ok:
        print()
        print("  free-tier queue wait by offer time (5s windows):")
        buckets: Dict[int, List[float]] = {}
        for r in free_ok:
            buckets.setdefault(int(r.send_s // 5), []).append(r.queue_wait_ms)
        for b in sorted(buckets):
            v = buckets[b]
            print(f"    {b*5:>3}-{b*5+5:<3}s  n={len(v):<3} p50={percentile(v,.5):>7.0f}  max={max(v):>7.0f}")


def write_csv(path: str, recs: List[Rec]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "send_s", "status", "latency_ms", "queue_wait_ms", "aged", "queue_depth_at_enqueue"])
        for r in sorted(recs, key=lambda x: x.send_s):
            w.writerow([r.tier, f"{r.send_s:.4f}", r.status, f"{r.latency_ms:.2f}",
                        f"{r.queue_wait_ms:.2f}", int(r.aged), f"{r.depth:.1f}"])
    print(f"\n  wrote {path}")


async def main_async(a) -> None:
    url = a.url.rstrip("/") + "/generate"
    payload = {"prompt": a.prompt, "max_tokens": a.max_tokens}
    async with httpx.AsyncClient(timeout=a.timeout) as c:
        try:
            h = (await c.get(a.url.rstrip("/") + "/healthz")).json()
        except Exception as exc:
            raise SystemExit(f"cannot reach {a.url}: {exc}")
        print(f"server: max_queue_depth={h.get('max_queue_depth')} "
              f"free_limit={h.get('max_queue_depth_free')} aging_ms={h.get('aging_ms')}")
        await c.post(url, json={**payload, "tier": "paid"})
    print(f"offering paid={a.paid} rps, free={a.free} rps for {a.duration}s ...")
    recs = await run(url, payload, a.paid, a.free, a.duration, a.timeout)
    report(recs, a.label)
    if a.out:
        write_csv(a.out, recs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--paid", type=float, default=10.0, help="Paid-tier offered rps.")
    p.add_argument("--free", type=float, default=2.0, help="Free-tier offered rps.")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--label", default="tiers")
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
