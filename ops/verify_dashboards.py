"""Verify the Grafana dashboard against Prometheus, panel by panel.

Reads every PromQL expression out of ops/grafana/dashboards/llm-serving.json and
runs it against the Prometheus HTTP API over a chosen window, reporting
min/max/last for each series. Because the queries are read from the dashboard
file rather than retyped here, what gets verified is exactly what the panels
render — a passing run means the dashboards work, not that some parallel set of
queries works.

    # after a benchmark run, check the last 60 seconds
    python ops/verify_dashboards.py --window 60

    # only the panels that matter for a specific claim
    python ops/verify_dashboards.py --window 60 --panel "Queue depth"
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(HERE, "grafana", "dashboards", "llm-serving.json")


def query_range(base: str, expr: str, start: float, end: float,
                step: float) -> List[Dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}",
         "step": f"{step:g}"}
    )
    url = f"{base.rstrip('/')}/api/v1/query_range?{params}"
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.load(r)
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("error", "query failed"))
    return payload["data"]["result"]


def series_label(s: Dict[str, Any], legend: str) -> str:
    metric = s.get("metric", {})
    if legend and "{{" not in legend:
        return legend
    if not metric:
        return "(scalar)"
    return ", ".join(f"{k}={v}" for k, v in sorted(metric.items())
                     if k != "__name__") or metric.get("__name__", "(value)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prom", default="http://localhost:9090")
    p.add_argument("--window", type=float, default=60.0,
                   help="Seconds back from --end to evaluate.")
    p.add_argument("--end", type=float, default=None,
                   help="Unix time for the window end (default: now).")
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--panel", default=None,
                   help="Only panels whose title contains this substring.")
    a = p.parse_args()

    import time as _time
    end = a.end if a.end else _time.time()
    start = end - a.window

    with open(DASHBOARD) as fh:
        dash = json.load(fh)

    panels = [pa for pa in dash["panels"]
              if not a.panel or a.panel.lower() in pa["title"].lower()]
    if not panels:
        print(f"no panel matching {a.panel!r}", file=sys.stderr)
        return 2

    print(f"dashboard: {dash['title']}  ({len(panels)} panels)")
    print(f"window:    {a.window:.0f}s ending {_time.strftime('%H:%M:%S', _time.localtime(end))}")
    print()

    failures = 0
    for panel in panels:
        print(f"── [{panel['id']:>2}] {panel['title']}")
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            legend = target.get("legendFormat", "")
            try:
                result = query_range(a.prom, expr, start, end, a.step)
            except Exception as exc:
                print(f"     {legend or expr[:40]:<24} ERROR: {exc}")
                failures += 1
                continue

            if not result:
                print(f"     {legend or expr[:40]:<24} (no data)")
                failures += 1
                continue

            for s in result:
                vals = [float(v[1]) for v in s["values"]
                        if v[1] not in ("NaN", "+Inf", "-Inf")]
                if not vals:
                    print(f"     {series_label(s, legend):<24} (all NaN)")
                    failures += 1
                    continue
                print(f"     {series_label(s, legend):<24} "
                      f"min={min(vals):>10.3f}  max={max(vals):>10.3f}  "
                      f"last={vals[-1]:>10.3f}  n={len(vals)}")
        print()

    if failures:
        print(f"{failures} target(s) returned no usable data.")
    else:
        print("every panel target returned data.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
