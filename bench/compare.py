"""Compare two loadtest summary CSVs and report the delta per concurrency level.

    python -m bench.compare \
        --before bench/results/phase1_baseline_summary.csv \
        --after  bench/results/phase2_batching_summary.csv

Exists so the headline number ("Nx throughput") is computed from the recorded
runs rather than eyeballed off two tables.
"""

import argparse
import csv
from typing import Dict, List


def load_summary(path: str) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed = {}
            for key, value in raw.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = float("nan")
            rows[int(parsed["concurrency"])] = parsed
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    args = parser.parse_args()

    before = load_summary(args.before)
    after = load_summary(args.after)
    levels: List[int] = sorted(set(before) & set(after))

    if not levels:
        raise SystemExit("No concurrency levels in common between the two files.")

    print()
    print(f"=== {args.before_label} -> {args.after_label} ===")
    header = (
        f"{'conc':>5} | {'rps b':>8} {'rps a':>8} {'gain':>7} | "
        f"{'p50 b':>9} {'p50 a':>9} {'delta':>8} | "
        f"{'p95 b':>9} {'p95 a':>9} {'delta':>8} | {'batch':>6}"
    )
    print(header)
    print("-" * len(header))

    gains = []
    for level in levels:
        b, a = before[level], after[level]
        gain = a["throughput_rps"] / b["throughput_rps"]
        gains.append(gain)
        print(
            f"{level:>5} | {b['throughput_rps']:>8.2f} {a['throughput_rps']:>8.2f} "
            f"{gain:>6.2f}x | "
            f"{b['p50_ms']:>9.1f} {a['p50_ms']:>9.1f} "
            f"{a['p50_ms'] - b['p50_ms']:>+8.1f} | "
            f"{b['p95_ms']:>9.1f} {a['p95_ms']:>9.1f} "
            f"{a['p95_ms'] - b['p95_ms']:>+8.1f} | "
            f"{a.get('mean_batch_size', float('nan')):>6.2f}"
        )

    print()
    print(f"peak throughput gain: {max(gains):.2f}x "
          f"(at concurrency {levels[gains.index(max(gains))]})")
    print("Latency deltas are ms; negative means faster.")


if __name__ == "__main__":
    main()
