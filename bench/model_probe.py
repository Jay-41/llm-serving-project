"""Measure the real model's cost curve and compare it to the mock's.

Loads the backend directly -- no HTTP, no scheduler -- and times one batched
generation at each batch size, splitting it into prefill (time to the first
decode step) and per-step decode cost. That is exactly the two-part shape the
mock assumes:

    total(N, T) = base + per_token * T * (1 + alpha * (N - 1))

so the output is a direct check on MOCK_BASE_MS, MOCK_PER_TOKEN_MS and above
all MOCK_BATCH_ALPHA -- the one number every Phase 2-5 result is downstream of.

    BACKEND=qwen python -m bench.model_probe --sizes 1,2,4,8 --tokens 64

Run once on the Mac (MPS) to shake out the backend code for free, then on the
rented GPU for the numbers that count. The two will not agree, and that is
fine: the question here is whether the SHAPE holds, and the GPU run supplies
the absolute values.
"""

import argparse
import os
import statistics
import time
from typing import Dict, List

from app.backends import build_backend
from app.config import get_settings

PROMPT = "Explain, in one short paragraph, why batching requests improves GPU throughput."


def time_one(backend, batch_size: int, max_tokens: int) -> Dict[str, float]:
    prompts = [PROMPT] * batch_size
    t0 = time.perf_counter()
    first = None
    steps = 0
    for _ in backend.generate_batch_stream(prompts, max_tokens):
        now = time.perf_counter()
        if first is None:
            first = now
        steps += 1
    end = time.perf_counter()
    assert first is not None
    prefill_ms = (first - t0) * 1000.0
    decode_ms = (end - first) * 1000.0
    return {
        "steps": steps,
        "prefill_ms": prefill_ms,
        "step_ms": decode_ms / max(steps - 1, 1),
        "total_ms": (end - t0) * 1000.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="1,2,4,8")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()

    settings = get_settings()
    if settings.backend != "qwen":
        raise SystemExit("set BACKEND=qwen -- this script measures the real model")

    backend = build_backend(settings)
    t0 = time.perf_counter()
    backend.load()
    print(f"loaded {settings.model_name} on {backend._device} in {time.perf_counter() - t0:.1f}s")

    # Warm-up: first call pays for kernel compilation / cache allocation and
    # would otherwise poison the batch-1 measurement.
    time_one(backend, 1, 8)

    sizes = [int(x) for x in a.sizes.split(",")]
    rows: List[Dict[str, float]] = []
    for n in sizes:
        runs = [time_one(backend, n, a.tokens) for _ in range(a.repeats)]
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        med["batch"] = n
        rows.append(med)
        print(f"  batch={n:<2} prefill={med['prefill_ms']:7.1f}ms  "
              f"step={med['step_ms']:6.2f}ms  total={med['total_ms']:7.0f}ms  "
              f"({n * a.tokens / (med['total_ms'] / 1000):6.0f} tok/s)")

    base = rows[0]
    print()
    print(f"{'batch':>5} {'step ms':>8} {'vs batch1':>10} {'implied α':>10} {'tok/s':>7}")
    for r in rows:
        n = int(r["batch"])
        factor = r["step_ms"] / base["step_ms"]
        alpha = (factor - 1.0) / (n - 1) if n > 1 else float("nan")
        tps = n * a.tokens / (r["total_ms"] / 1000.0)
        print(f"{n:>5} {r['step_ms']:>8.2f} {factor:>9.2f}x {alpha:>10.3f} {tps:>7.0f}")

    big = rows[-1]
    n = int(big["batch"])
    if n < 2:
        print("\n(pass --sizes with a batch size > 1 to estimate alpha)")
        return
    alpha_at_max = (big["step_ms"] / base["step_ms"] - 1.0) / (n - 1)
    print()
    print("measured model, to compare with the mock's defaults:")
    print(f"  MOCK_BASE_MS       {base['prefill_ms']:7.1f}   (mock default 40)")
    print(f"  MOCK_PER_TOKEN_MS  {base['step_ms']:7.2f}   (mock default 8)")
    print(f"  MOCK_BATCH_ALPHA   {alpha_at_max:7.3f}   (mock default 0.08, from batch {n})")
    print()
    print(f"  batch-{n} throughput gain over batch-1: "
          f"{(n * a.tokens / big['total_ms']) / (a.tokens / base['total_ms']):.2f}x")

    # One greppable line so bench/phase6_sweep.sh --auto can derive the
    # admission threshold from the measured batch-8 time without a human
    # reading the table.
    b8 = next((r for r in rows if int(r["batch"]) == 8), big)
    print(f"PROBE base_ms={base['prefill_ms']:.1f} per_token_ms={base['step_ms']:.2f} "
          f"alpha={alpha_at_max:.4f} batch8_total_ms={b8['total_ms']:.0f} device={backend._device}")


if __name__ == "__main__":
    main()
