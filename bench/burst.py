"""Open-loop burst generator for overload testing.

Open loop means requests are sent on a fixed schedule — one every 1/rate
seconds — **regardless of whether earlier ones have come back**. That is the
whole point, and it is why bench/loadtest.py cannot be used for this: a
closed-loop generator's workers each wait for a response before sending again,
so the offered load throttles itself to whatever the server can serve. It is
structurally incapable of overloading anything.

To test backpressure you have to be able to offer more load than the system can
drain and keep offering it. That is what this does.

    # overload with admission control OFF (control condition)
    MAX_QUEUE_DEPTH=0 uvicorn app.main:app ...
    python -m bench.burst --rate 20 --duration 20 --label "no admission control"

    # same load, admission control ON
    MAX_QUEUE_DEPTH=16 uvicorn app.main:app ...
    python -m bench.burst --rate 20 --duration 20 --label "admission control"

The headline evidence is the per-bucket latency table: without admission
control, accepted latency climbs without bound for as long as the overload
lasts. With it, accepted latency is flat.
"""

import argparse
import asyncio
import csv
import os
import statistics
import time
from typing import Dict, List, Optional

import httpx

DEFAULT_PROMPT = "Explain what backpressure means in one short paragraph."


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


class Rec:
    __slots__ = ("idx", "send_s", "latency_ms", "status", "queue_wait_ms",
                 "batch_size", "depth")

    def __init__(self, idx, send_s, latency_ms, status, queue_wait_ms,
                 batch_size, depth):
        self.idx = idx
        self.send_s = send_s
        self.latency_ms = latency_ms
        self.status = status
        self.queue_wait_ms = queue_wait_ms
        self.batch_size = batch_size
        self.depth = depth


async def fire(client, url, payload, idx, t0, out: List[Rec]) -> None:
    """Send one request. Never waits for anything but its own response."""
    send_s = time.perf_counter() - t0
    started = time.perf_counter()
    qwait = bsize = depth = float("nan")
    try:
        r = await client.post(url, json=payload)
        latency = (time.perf_counter() - started) * 1000.0
        status = r.status_code
        if status == 200:
            body = r.json()
            qwait = body.get("queue_wait_ms", float("nan"))
            bsize = body.get("batch_size", float("nan"))
            depth = body.get("queue_depth_at_enqueue", float("nan"))
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
    out.append(Rec(idx, send_s, latency, status, qwait, bsize, depth))


async def run_burst(url, payload, rate, duration, timeout_s) -> List[Rec]:
    total = int(round(rate * duration))
    interval = 1.0 / rate
    out: List[Rec] = []

    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        t0 = time.perf_counter()
        tasks = []
        for i in range(total):
            # Sleep until this request's scheduled send time, then fire and
            # move on. We do not await the response here — that would make the
            # generator closed-loop and defeat the entire test.
            target = t0 + i * interval
            slack = target - time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)
            tasks.append(asyncio.ensure_future(
                fire(client, url, payload, i, t0, out)))
        offered_s = time.perf_counter() - t0
        # Now wait for the backlog to drain. With admission control off this
        # is where the run gets long — that IS the finding.
        await asyncio.gather(*tasks)
        total_s = time.perf_counter() - t0

    print(f"  offered {total} requests over {offered_s:.1f}s "
          f"({total / offered_s:.1f} rps), drained at {total_s:.1f}s")
    return out


def summarise(recs: List[Rec], rate: float, duration: float,
              bucket_s: float) -> Dict:
    ok = [r for r in recs if r.status == 200]
    rejected = [r for r in recs if r.status == 503]
    errors = [r for r in recs if not isinstance(r.status, int)]
    other = [r for r in recs
             if isinstance(r.status, int) and r.status not in (200, 503)]

    ok_lat = [r.latency_ms for r in ok]
    rej_lat = [r.latency_ms for r in rejected]

    # Bucket by SEND time, not completion time — we want latency as a function
    # of when load was offered, which is what shows unbounded growth.
    buckets: Dict[int, List[float]] = {}
    for r in ok:
        b = int(r.send_s // bucket_s)
        buckets.setdefault(b, []).append(r.latency_ms)

    offer_span = max((r.send_s for r in recs), default=0.0)
    # Wall clock must run to the LAST RESPONSE, not the last send. Dividing by
    # the send window would credit the unprotected server with the full offered
    # rate while it was still 20 seconds behind — measuring what we asked for
    # rather than what it delivered.
    wall = max((r.send_s + r.latency_ms / 1000.0 for r in recs), default=0.0)
    return {
        "offered": len(recs),
        "accepted": len(ok),
        "rejected": len(rejected),
        "errors": len(errors) + len(other),
        "accept_pct": 100.0 * len(ok) / len(recs) if recs else 0.0,
        "offered_rps": len(recs) / offer_span if offer_span > 0 else float("nan"),
        "wall_s": wall,
        "goodput_rps": len(ok) / wall if wall > 0 else float("nan"),
        "ok_p50": percentile(ok_lat, 0.50),
        "ok_p95": percentile(ok_lat, 0.95),
        "ok_p99": percentile(ok_lat, 0.99),
        "ok_max": max(ok_lat) if ok_lat else float("nan"),
        "ok_mean": statistics.fmean(ok_lat) if ok_lat else float("nan"),
        "rej_p50": percentile(rej_lat, 0.50),
        "rej_p99": percentile(rej_lat, 0.99),
        "peak_depth": max((r.depth for r in recs if r.depth == r.depth),
                          default=float("nan")),
        "buckets": buckets,
        "bucket_s": bucket_s,
        "_recs": recs,
    }


def report(s: Dict, label: str) -> None:
    print()
    print(f"=== {label} ===")
    print(f"  offered   {s['offered']:>5}")
    print(f"  accepted  {s['accepted']:>5}  ({s['accept_pct']:.1f}%)")
    print(f"  rejected  {s['rejected']:>5}  (HTTP 503)")
    if s["errors"]:
        print(f"  errors    {s['errors']:>5}  <-- timeouts / connection failures")
    print()
    print(f"  offered rate        {s['offered_rps']:>8.2f} rps")
    print(f"  wall clock          {s['wall_s']:>8.1f} s   "
          f"(until the last response landed)")
    print(f"  goodput             {s['goodput_rps']:>8.2f} rps  "
          f"(accepted / wall clock)")
    print(f"  accepted p50        {s['ok_p50']:>8.0f} ms")
    print(f"  accepted p95        {s['ok_p95']:>8.0f} ms")
    print(f"  accepted p99        {s['ok_p99']:>8.0f} ms")
    print(f"  accepted max        {s['ok_max']:>8.0f} ms")
    if s["rejected"]:
        print(f"  rejection p50       {s['rej_p50']:>8.1f} ms  <-- cost of a 'no'")
        print(f"  rejection p99       {s['rej_p99']:>8.1f} ms")
    print(f"  peak queue depth    {s['peak_depth']:>8.0f}")

    print()
    print(f"  accepted latency by offer time ({s['bucket_s']:.0f}s buckets):")
    print(f"    {'window':>12} {'n':>5} {'p50':>9} {'p95':>9} {'max':>9}")
    for b in sorted(s["buckets"]):
        vals = s["buckets"][b]
        lo = b * s["bucket_s"]
        print(f"    {lo:>5.0f}-{lo + s['bucket_s']:<6.0f} {len(vals):>5} "
              f"{percentile(vals, .5):>9.0f} {percentile(vals, .95):>9.0f} "
              f"{max(vals):>9.0f}")


def write_csv(path: str, s: Dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "send_s", "status", "latency_ms", "queue_wait_ms",
                    "batch_size", "queue_depth_at_enqueue"])
        for r in sorted(s["_recs"], key=lambda x: x.idx):
            w.writerow([r.idx, f"{r.send_s:.4f}", r.status,
                        f"{r.latency_ms:.2f}", f"{r.queue_wait_ms:.2f}",
                        f"{r.batch_size:.1f}", f"{r.depth:.1f}"])
    print(f"\n  wrote {path}")


async def main_async(a) -> None:
    url = a.url.rstrip("/") + "/generate"
    payload = {"prompt": a.prompt, "max_tokens": a.max_tokens}

    async with httpx.AsyncClient(timeout=a.timeout) as c:
        try:
            h = (await c.get(a.url.rstrip("/") + "/healthz")).json()
        except Exception as exc:
            raise SystemExit(f"cannot reach {a.url}: {exc}")
        limit = h.get("max_queue_depth", 0)
        print(f"server: max_batch_size={h.get('max_batch_size')} "
              f"max_queue_depth={limit if limit else 'UNBOUNDED (off)'}")
        # Warm up so the first bucket does not absorb connection setup.
        await c.post(url, json=payload)

    print(f"offering {a.rate} rps for {a.duration}s ...")
    recs = await run_burst(url, payload, a.rate, a.duration, a.timeout)
    s = summarise(recs, a.rate, a.duration, a.bucket)
    report(s, a.label)
    if a.out:
        write_csv(a.out, s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--rate", type=float, default=20.0,
                   help="Requests per second to offer, open loop.")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--bucket", type=float, default=4.0,
                   help="Seconds per latency-vs-time bucket.")
    p.add_argument("--label", default="burst")
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
