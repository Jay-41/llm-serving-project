"""Closed-loop load generator for the serving layer.

Closed loop means: `--concurrency` workers each hold one request in flight at a
time and send the next as soon as the previous returns. That measures how fast
the server *can* go (saturation throughput), which is what the Phase 1 vs
Phase 2 comparison needs. An open-loop generator with a fixed arrival rate
answers a different question -- "does it keep up with X rps" -- and is the right
tool for Phase 4's burst testing, not here.

Usage:
    python -m bench.loadtest --concurrency 1,2,4,8,16 --requests 40
    python -m bench.loadtest --concurrency 1 --requests 20 --out bench/results/phase1.csv
"""

import argparse
import asyncio
import csv
import os
import statistics
import time
from typing import Dict, List, Optional

import httpx

DEFAULT_PROMPT = "Explain what request batching is in one short paragraph."


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0, 1]."""
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


class Result:
    __slots__ = ("concurrency", "worker", "index", "status", "latency_ms",
                 "wait_ms", "inference_ms", "batch_size", "ttft_client_ms",
                 "ttft_server_ms")

    def __init__(self, concurrency, worker, index, status, latency_ms,
                 wait_ms, inference_ms, batch_size,
                 ttft_client_ms=float("nan"), ttft_server_ms=float("nan")):
        self.concurrency = concurrency
        self.worker = worker
        self.index = index
        self.status = status
        self.latency_ms = latency_ms
        self.wait_ms = wait_ms
        self.inference_ms = inference_ms
        self.batch_size = batch_size
        # Client-side: first token event observed on the wire. Server-side:
        # the ttft_ms the server reports. The gap between them is network.
        self.ttft_client_ms = ttft_client_ms
        self.ttft_server_ms = ttft_server_ms


async def worker_loop(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict,
    concurrency: int,
    worker_id: int,
    budget: "asyncio.Queue",
    results: List[Result],
) -> None:
    """Pull one token off the shared budget per request, until it is empty."""
    index = 0
    while True:
        try:
            budget.get_nowait()
        except asyncio.QueueEmpty:
            return

        started = time.perf_counter()
        wait_ms = inference_ms = batch_size = float("nan")
        ttft_c = ttft_s = float("nan")
        try:
            if payload.get("stream"):
                status, body, ttft_c = await _stream_once(client, url, payload, started)
            else:
                response = await client.post(url, json=payload)
                status = response.status_code
                body = response.json() if status == 200 else None
            latency_ms = (time.perf_counter() - started) * 1000.0
            if status == 200 and body:
                wait_ms = body.get("queue_wait_ms", float("nan"))
                inference_ms = body.get("inference_ms", float("nan"))
                batch_size = body.get("batch_size", float("nan"))
                t = body.get("ttft_ms")
                ttft_s = float(t) if t is not None else float("nan")
        except Exception as exc:  # network error, timeout, refused connection
            latency_ms = (time.perf_counter() - started) * 1000.0
            status = f"error:{type(exc).__name__}"

        results.append(
            Result(concurrency, worker_id, index, status, latency_ms,
                   wait_ms, inference_ms, batch_size, ttft_c, ttft_s)
        )
        index += 1


async def _stream_once(client, url, payload, started):
    """Consume one SSE response. Returns (status, done_payload, ttft_client_ms).

    TTFT is stamped when the first `token` event's data line arrives on the
    wire -- what a user would perceive -- not when the connection opens.
    """
    import json as _json

    ttft = float("nan")
    done = None
    async with client.stream("POST", url, json=payload) as r:
        if r.status_code != 200:
            await r.aread()
            return r.status_code, None, ttft
        event = None
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                if event == "token" and ttft != ttft:  # first token, ttft is NaN
                    ttft = (time.perf_counter() - started) * 1000.0
                elif event == "done":
                    done = _json.loads(line[5:])
    return 200, done, ttft


async def run_level(
    url: str,
    payload: Dict,
    concurrency: int,
    total_requests: int,
    timeout_s: float,
) -> Dict:
    """Run one concurrency level to completion and summarise it."""
    budget: "asyncio.Queue" = asyncio.Queue()
    for _ in range(total_requests):
        budget.put_nowait(1)

    results: List[Result] = []
    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 8,
    )

    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        wall_start = time.perf_counter()
        await asyncio.gather(
            *[
                worker_loop(client, url, payload, concurrency, wid, budget, results)
                for wid in range(concurrency)
            ]
        )
        wall_s = time.perf_counter() - wall_start

    ok = [r for r in results if r.status == 200]
    latencies = [r.latency_ms for r in ok]
    waits = [r.wait_ms for r in ok if r.wait_ms == r.wait_ms]  # drop NaN
    infers = [r.inference_ms for r in ok if r.inference_ms == r.inference_ms]
    batches = [r.batch_size for r in ok if r.batch_size == r.batch_size]
    ttft_c = [r.ttft_client_ms for r in ok if r.ttft_client_ms == r.ttft_client_ms]
    ttft_s = [r.ttft_server_ms for r in ok if r.ttft_server_ms == r.ttft_server_ms]

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": wall_s,
        "throughput_rps": (len(ok) / wall_s) if wall_s > 0 else float("nan"),
        "mean_ms": statistics.fmean(latencies) if latencies else float("nan"),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "max_ms": max(latencies) if latencies else float("nan"),
        "mean_wait_ms": statistics.fmean(waits) if waits else float("nan"),
        "mean_inference_ms": statistics.fmean(infers) if infers else float("nan"),
        "mean_batch_size": statistics.fmean(batches) if batches else float("nan"),
        "ttft_client_p50": percentile(ttft_c, 0.50),
        "ttft_client_p95": percentile(ttft_c, 0.95),
        "ttft_server_p50": percentile(ttft_s, 0.50),
        "ttft_server_p95": percentile(ttft_s, 0.95),
        "_rows": results,
    }


def print_summary(summaries: List[Dict], label: str) -> None:
    print()
    print(f"=== {label} ===")
    header = (
        f"{'conc':>5} {'ok':>5} {'fail':>5} {'rps':>8} {'mean':>9} "
        f"{'p50':>9} {'p95':>9} {'p99':>9} {'wait':>9} {'infer':>9} {'batch':>7} "
        f"{'ttft50':>8} {'ttft95':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['concurrency']:>5} {s['ok']:>5} {s['failed']:>5} "
            f"{s['throughput_rps']:>8.2f} {s['mean_ms']:>9.1f} "
            f"{s['p50_ms']:>9.1f} {s['p95_ms']:>9.1f} {s['p99_ms']:>9.1f} "
            f"{s['mean_wait_ms']:>9.1f} {s['mean_inference_ms']:>9.1f} "
            f"{s['mean_batch_size']:>7.2f} "
            f"{s['ttft_client_p50']:>8.1f} {s['ttft_client_p95']:>8.1f}"
        )
    print()
    print("rps = completed requests / wall clock. Latency columns are ms, "
          "measured client-side.")
    print("wait/infer are server-reported: queueing delay vs. time in the model.")
    print("batch = mean number of requests sharing a forward pass.")
    print("ttft = time to first token, client-side; only measured with --stream.")


def write_csv(path: str, summaries: List[Dict]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["concurrency", "worker", "index", "status", "latency_ms",
             "wait_ms", "inference_ms", "batch_size",
             "ttft_client_ms", "ttft_server_ms"]
        )
        for summary in summaries:
            for row in summary["_rows"]:
                writer.writerow(
                    [row.concurrency, row.worker, row.index, row.status,
                     f"{row.latency_ms:.3f}", f"{row.wait_ms:.3f}",
                     f"{row.inference_ms:.3f}", f"{row.batch_size:.2f}",
                     f"{row.ttft_client_ms:.3f}", f"{row.ttft_server_ms:.3f}"]
                )

    summary_path = path.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        keys = ["concurrency", "requests", "ok", "failed", "wall_s",
                "throughput_rps", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                "max_ms", "mean_wait_ms", "mean_inference_ms",
                "mean_batch_size", "ttft_client_p50", "ttft_client_p95",
                "ttft_server_p50", "ttft_server_p95"]
        writer.writerow(keys)
        for summary in summaries:
            writer.writerow([summary[k] for k in keys])

    print(f"wrote {path}")
    print(f"wrote {summary_path}")


async def main_async(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/") + "/generate"
    payload = {"prompt": args.prompt, "max_tokens": args.max_tokens}
    if args.stream:
        payload["stream"] = True
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]

    # Warm up so the first level does not absorb connection setup and any
    # first-call cost in the backend.
    if args.warmup > 0:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            for _ in range(args.warmup):
                try:
                    await client.post(url, json=payload)
                except Exception as exc:
                    raise SystemExit(
                        f"warmup request to {url} failed: {exc}\n"
                        "Is the server running? "
                        "uvicorn app.main:app --port 8000"
                    )
        print(f"warmup: {args.warmup} request(s) done")

    summaries = []
    for concurrency in levels:
        requests = args.requests if args.requests else concurrency * 10
        print(f"running concurrency={concurrency} requests={requests} ...")
        summary = await run_level(url, payload, concurrency, requests, args.timeout)
        summaries.append(summary)

    print_summary(summaries, args.label)
    if args.out:
        write_csv(args.out, summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels to sweep.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=40,
        help="Requests per concurrency level. 0 means concurrency * 10.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stream", action="store_true",
                        help="Request SSE streaming and measure time to first token.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="load test")
    parser.add_argument("--out", default=None, help="Path for per-request CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
