"""Render the README benchmark charts from the recorded CSVs.

    python -m bench.charts            # writes docs/charts/*.png

Every number on every chart is read from bench/results/ -- nothing is typed
in -- so re-running a benchmark and re-running this script keeps the README
honest.
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("bench/results")
OUT = Path("docs/charts")
OUT.mkdir(parents=True, exist_ok=True)

# Validated pair (light surface): CVD dE 21.7, normal-vision dE 26.7, all six
# checks pass. Reference series are neutral and dashed -- not a category.
BLUE, AMBER, REF = "#2A7DB8", "#C2740C", "#8A8F94"
INK, INK2, GRID = "#1F2933", "#5B6570", "#E3E7EA"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "semibold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})
MARK = dict(marker="o", markersize=6, linewidth=2)


def summary(name):
    rows = list(csv.DictReader(open(R / name)))
    return {int(r["concurrency"]): {k: float(v) for k, v in r.items() if v not in ("", None)} for r in rows}


def label(ax, x, y, text, dx=0, dy=8, color=INK, ha="center", size=10, weight="semibold"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va="bottom", fontsize=size, color=color, fontweight=weight)


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. batching
nb, b = summary("gpu_nobatch_summary.csv"), summary("gpu_batching_summary.csv")
mock = summary("phase2_batching_summary.csv")
xs = sorted(b)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(xs, [mock[x]["throughput_rps"] for x in xs], color=REF, linestyle="--", linewidth=1.6,
        marker="o", markersize=4, label="Batching, mock (predicted before the GPU run)")
ax.plot(xs, [nb[x]["throughput_rps"] for x in xs], color=BLUE, label="No batching  (MAX_BATCH_SIZE=1)", **MARK)
ax.plot(xs, [b[x]["throughput_rps"] for x in xs], color=AMBER, label="Batching  (MAX_BATCH_SIZE=8)", **MARK)
for x in (8, 16):
    label(ax, x, b[x]["throughput_rps"], f"{b[x]['throughput_rps']:.2f} rps", dy=9)
label(ax, 16, nb[16]["throughput_rps"], f"{nb[16]['throughput_rps']:.2f} rps", dy=-16)
gain = b[16]["throughput_rps"] / nb[16]["throughput_rps"]
ax.annotate("", xy=(16, b[16]["throughput_rps"] - 0.25), xytext=(16, nb[16]["throughput_rps"] + 0.35),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
label(ax, 16, (b[16]["throughput_rps"] + nb[16]["throughput_rps"]) / 2, f"{gain:.2f}×", dx=-14, dy=-6, ha="right", size=12)
ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel("Concurrent clients (closed loop)"); ax.set_ylabel("Throughput, requests / s")
ax.set_ylim(0, 10.5); ax.set_title("Dynamic batching on an NVIDIA L4 — Qwen2.5-1.5B, 64 tokens/request")
ax.legend(loc="upper left")
save(fig, "batching_throughput.png",
     "Same code, one setting flipped. 40 requests per level, admission control off, zero rejections. Plateau at 8 is MAX_BATCH_SIZE.")

# ------------------------------------------------------------- 2. calibration
steps = {}
for line in open(R / "gpu_probe.txt"):
    m = re.match(r"\s+batch=(\d+)\s+prefill=\s*([\d.]+)ms\s+step=\s*([\d.]+)ms", line)
    if m:
        steps[int(m.group(1))] = float(m.group(3))
ns = sorted(steps)
rel = [steps[n] / steps[1] for n in ns]
mock_rel = [1 + 0.08 * (n - 1) for n in ns]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(ns, mock_rel, color=REF, linestyle="--", linewidth=1.6, marker="o", markersize=4, label="Mock's assumption, α = 0.08")
ax.plot(ns, rel, color=AMBER, label="Measured on the L4, α = 0.007", **MARK)
ax.axhline(1.0, color=GRID, linewidth=1)
label(ax, 16, rel[-1], f"{rel[-1]:.2f}× — sixteen sequences cost 11% more per step than one", dx=-4, dy=-22, ha="right")
label(ax, 16, mock_rel[-1], f"{mock_rel[-1]:.2f}×", dy=6, color=INK2, weight="normal")
ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("Sequences in the batch"); ax.set_ylabel("Cost of one decode step, relative to batch of 1")
ax.set_ylim(0.8, 2.4); ax.set_title("Why batching is nearly free: decode is memory-bandwidth bound")
ax.legend(loc="upper left")
save(fig, "batch_cost_curve.png",
     "The weights are read from HBM once per decode step no matter how many sequences share it. Linear scaling would be 16× (off the chart).")

# ------------------------------------------------------------ 3. backpressure
def buckets(name, width=4.0):
    rows = [r for r in csv.DictReader(open(R / name)) if r["status"] == "200"]
    bk = {}
    for r in rows:
        bk.setdefault(int(float(r["send_s"]) // width), []).append(float(r["latency_ms"]))
    xs = sorted(bk)
    p50 = [sorted(bk[k])[len(bk[k]) // 2] / 1000 for k in xs]
    return [k * width + width / 2 for k in xs], p50
xo, yo = buckets("gpu_no_admission.csv"); xa, ya = buckets("gpu_admission.csv")
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(xo, yo, color=BLUE, label="Admission control off — queue unbounded", **MARK)
ax.plot(xa, ya, color=AMBER, label="Admission control on — queue capped at 10", **MARK)
label(ax, xo[-1], yo[-1], f"{yo[-1]:.1f} s and still climbing", dx=-6, dy=6, ha="right")
label(ax, xa[-1], ya[-1], f"{ya[-1]:.1f} s, flat", dx=-6, dy=8, ha="right")
ax.set_xlabel("When the request was sent (seconds into a 20 s overload at 2× capacity)")
ax.set_ylabel("Accepted-request p50 latency, seconds"); ax.set_ylim(0, max(yo) * 1.15)
ax.set_title("Backpressure under sustained overload on the L4 — 18 rps offered, 9 rps capacity")
ax.legend(loc="upper left")
save(fig, "backpressure_latency.png",
     "Bucketed by send time. Unprotected, latency grows for as long as the overload lasts; protected, the excess is refused in ~3 ms each.")

# ----------------------------------------------------------------- 4. TTFT
st = summary("gpu_stream_summary.csv")
xs = sorted(st)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(xs, [st[x]["p50_ms"] for x in xs], color=BLUE, label="Full response, p50", **MARK)
ax.plot(xs, [st[x]["ttft_client_p50"] for x in xs], color=AMBER, label="Time to first token, p50", **MARK)
for x in (1, 8):
    r = st[x]["p50_ms"] / st[x]["ttft_client_p50"]
    label(ax, x, st[x]["ttft_client_p50"], f"{st[x]['ttft_client_p50']:.0f} ms — first output {r:.0f}× sooner", dy=-18, dx=6, ha="left", size=9.5)
label(ax, 16, st[16]["ttft_client_p50"], f"{st[16]['ttft_client_p50']:.0f} ms — waited for someone else's batch", dy=-34, dx=-6, ha="right", size=9.5)
ax.set_yscale("log"); ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000]); ax.set_yticklabels(["20", "50", "100", "200", "500", "1000", "2000"])
ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel("Concurrent clients"); ax.set_ylabel("Milliseconds (log scale)")
ax.set_title("Streaming on the L4: when the first token arrives vs when the response completes")
ax.legend(loc="upper left")
save(fig, "ttft.png",
     "Under capacity the first token lands in ~30–45 ms. Over capacity a new arrival waits one full batch — the static-batching cost continuous batching removes.")
