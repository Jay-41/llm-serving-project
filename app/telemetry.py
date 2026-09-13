"""Prometheus metrics — the aggregate observability layer.

This sits alongside `app/metrics.py`, which writes one JSON record per request.
The two are not redundant; they answer different questions:

    metrics.py   "what happened to request #4182?"          per-request truth
    telemetry.py "what is p99 latency doing right now?"      aggregate trend

Prometheus cannot reconstruct the first from the second — histograms discard
identity — and the JSONL cannot cheaply answer the second over a live window.
Working rule #2 depends on the JSONL; the dashboards depend on this.

Cardinality note: the only labels used anywhere here are `outcome` (two
values) and `tier` (two values). Nothing is labelled per-request, per-prompt
or per-client — that is how a metrics endpoint turns into an outage.

Single-process assumption: uvicorn runs one worker, so the default global
registry is correct. Running multiple workers would need
prometheus_client's multiprocess mode, or each worker would report only its
own slice.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- Counters ------------------------------------------------------------
# rate(llm_requests_total{outcome="served"}[30s]) is goodput.
# rate(llm_requests_total{outcome="rejected"}[30s]) is shed load.
REQUESTS = Counter(
    "llm_requests_total",
    "Requests leaving the admission decision, by outcome and tier.",
    ["outcome", "tier"],
)

# Free requests promoted to paid priority by aging. If this is zero under
# mixed load, either aging is off or free requests are never waiting long
# enough to need it. If it is high, the aging threshold may be doing all the
# scheduling and the tiers are not really distinct.
AGED = Counter(
    "llm_aged_promotions_total",
    "Free-tier requests promoted to paid priority after waiting aging_ms.",
)

BATCHES = Counter(
    "llm_batches_total",
    "Forward passes dispatched to the model.",
)

# --- Gauges --------------------------------------------------------------
# queue_depth is wired to a callback so a scrape reads the live value rather
# than whatever it happened to be at the last update.
QUEUE_DEPTH = Gauge(
    "llm_queue_depth",
    "Requests currently waiting for a batch.",
)

QUEUE_DEPTH_BY_TIER = Gauge(
    "llm_queue_depth_by_tier",
    "Requests currently waiting for a batch, by tier.",
    ["tier"],
)

QUEUE_DEPTH_LIMIT_FREE = Gauge(
    "llm_queue_depth_limit_free",
    "Admission threshold for free-tier requests. 0 means same as paid.",
)

# High-water mark. A 1s scrape interval will miss transient spikes; this will
# not, which is what makes it the evidence that the admission limit held.
QUEUE_DEPTH_PEAK = Gauge(
    "llm_queue_depth_peak",
    "Highest queue depth observed since startup.",
)

# Configuration exported as metrics so dashboards can draw the limit as a line
# on the same axis as the measurement, instead of hardcoding it in the panel.
QUEUE_DEPTH_LIMIT = Gauge(
    "llm_queue_depth_limit",
    "Admission control threshold. 0 means admission control is disabled.",
)

MAX_BATCH_SIZE = Gauge(
    "llm_max_batch_size",
    "Configured maximum batch size.",
)

# --- Histograms ----------------------------------------------------------
# Buckets are chosen against measured Phase 2 / Phase 4 ranges. Defaults would
# put almost every observation in one bucket and make the quantiles useless.
# Re-tune these in Phase 6 when real-model timings move.

# Phase 4 saw queue wait span 3ms (protected) to 22s (unprotected), so this
# has to be wide and log-ish to resolve both ends.
QUEUE_WAIT = Histogram(
    "llm_queue_wait_seconds",
    "Time from enqueue until a batch picked the request up.",
    ["tier"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
             1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
)

# Inference is tightly clustered: ~0.55s for a batch of 1, ~0.84s for 8.
# Fine buckets across that band, then a coarse tail.
INFERENCE = Histogram(
    "llm_inference_seconds",
    "Duration of one batched forward pass.",
    buckets=(0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0,
             1.25, 1.5, 2.0, 5.0),
)

# Extra resolution between 2.0 and 3.0: MAX_QUEUE_DEPTH is derived from a 2.5s
# p99 target, so that is precisely where the quantile must be trustworthy. With
# edges at only 2.5 and 3.0, a measured p99 of 2.53s interpolated to 2.93s —
# histogram_quantile assumes observations are spread uniformly inside a bucket,
# so a coarse bucket at the interesting point produces a confidently wrong
# number. Buckets belong where the decisions are.
E2E = Histogram(
    "llm_request_duration_seconds",
    "Enqueue until the response was ready.",
    ["tier"],
    buckets=(0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0,
             10.0, 20.0, 30.0),
)

# Time-to-first-token: enqueue until the first token EXISTED (the decode step
# that produced it completed), measured server-side. Under light load this is
# queue wait + prefill, roughly 50ms. Under heavy load it is dominated by
# waiting for someone else's batch to finish -- the static-batching cost that
# continuous batching removes. Fine buckets at the low end so that cost is
# legible; the streaming/non-streaming difference lives between 10ms and 1s.
TTFT = Histogram(
    "llm_ttft_seconds",
    "Enqueue until the request's first token was produced.",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
             1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

# rate(llm_tokens_generated_total[30s]) is tokens/sec -- the throughput unit
# that actually matters for a text generator, since requests vary in length.
TOKENS = Counter(
    "llm_tokens_generated_total",
    "Tokens produced across all served requests.",
)

# Observed once per batch, not per request. Mean batch size is therefore
# rate(llm_batch_size_sum[30s]) / rate(llm_batch_size_count[30s]).
BATCH_SIZE = Histogram(
    "llm_batch_size",
    "Number of requests sharing a forward pass.",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32),
)


def export_config(max_batch_size: int, max_queue_depth: int,
                  max_queue_depth_free: int = 0) -> None:
    """Publish the settings the dashboards draw limit lines from, and
    materialise every label value at zero.

    Touching both `outcome` children matters more than it looks. A Prometheus
    client only creates a labelled series the first time it is used, so a run
    with no rejections exports no `outcome="rejected"` series at all — and a
    panel querying it renders an empty legend entry rather than a flat zero
    line. Worse, `rate()` over a series that springs into existence mid-window
    has nothing to compare against, so the first rejection is undercounted.
    Initialising both up front means "zero rejections" is reported as zero
    rather than as silence.
    """
    MAX_BATCH_SIZE.set(max_batch_size)
    QUEUE_DEPTH_LIMIT.set(max_queue_depth)
    QUEUE_DEPTH_LIMIT_FREE.set(max_queue_depth_free)
    for tier in ("paid", "free"):
        REQUESTS.labels(outcome="served", tier=tier)
        REQUESTS.labels(outcome="rejected", tier=tier)
        QUEUE_WAIT.labels(tier=tier)
        E2E.labels(tier=tier)
        QUEUE_DEPTH_BY_TIER.labels(tier=tier)
