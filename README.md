# LLM Inference Serving Layer

Request queueing, dynamic batching, and backpressure for a single-node LLM
server. See `llm_serving_project_spec.md` for the full spec and phase plan.

> The architecture writeup, design-tradeoff discussion, and benchmark graphs
> are Phase 7 deliverables. This file is setup and run instructions only.

## Quick start

The whole stack — serving layer, Prometheus, Grafana — in one command:

```bash
docker compose up -d --build
```

| | |
| --- | --- |
| App | http://localhost:8000 |
| Grafana | http://localhost:3000 (opens on the dashboard, no login) |
| Prometheus | http://localhost:9090 |

Cold start from wiped volumes takes about 6 seconds. Every setting is
overridable from the shell, which is how the Phase 4 control comparison runs
without editing anything:

```bash
MAX_QUEUE_DEPTH=0 docker compose up -d app   # admission control off
docker compose up -d app                     # back to the default 16
```

```bash
docker compose down      # stop, keep metrics history
docker compose down -v   # stop and wipe volumes
```

Per-request logs live in a named volume (the container runs as uid 10001, so a
bind mount would arrive owned by the host user and the app could not write to
it):

```bash
docker compose exec app tail -f /app/logs/requests.jsonl
```

## Local setup without Docker

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --port 8000
```

`requirements-model.txt` (torch/transformers) is **not** needed until Phase 6.
Everything through Phase 5 runs on the CPU mock backend.

Running Prometheus in a container against an app on the host needs the scrape
target changed to `host.docker.internal:8000` and
`--add-host=host.docker.internal:host-gateway`; see `ops/prometheus/prometheus.yml`.

```bash
curl -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "hello", "max_tokens": 64}'
```

## Configuration

All via environment variables (see `app/config.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `BACKEND` | `mock` | `mock` or `qwen` |
| `MOCK_BASE_MS` | `40` | Modelled prefill cost |
| `MOCK_PER_TOKEN_MS` | `8` | Modelled per-token decode cost |
| `MOCK_JITTER_MS` | `5` | +/- jitter on mock service time |
| `MOCK_BATCH_ALPHA` | `0.08` | Marginal cost of each extra request in a batch |
| `MAX_BATCH_SIZE` | `8` | Batch dispatches when this full. `1` disables batching |
| `MAX_WAIT_MS` | `10` | Batch dispatches when this elapses, whichever comes first |
| `MAX_QUEUE_DEPTH` | `16` | Reject with 503 once this many are queued. `0` disables admission control |
| `RETRY_AFTER_S` | `2` | Value of the `Retry-After` header on rejections |
| `METRICS_PATH` | `logs/requests.jsonl` | Per-request JSONL sink |
| `DEFAULT_MAX_TOKENS` | `64` | Used when a request omits `max_tokens` |
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Phase 6 only |

Mock service time for a batch of N:

```
base_ms + per_token_ms * max_tokens * (1 + batch_alpha * (N - 1))
```

At the defaults a single 64-token request takes ~552ms and a batch of 8 takes
~839ms — 1.52x the cost for 8x the work. The shape comes from decode being
memory-bandwidth bound at small batch sizes: the weights are read from HBM once
per decode step regardless of how many sequences share that step, so batching is
much cheaper than linear but not free. `MOCK_BATCH_ALPHA` is the one number to
recalibrate against measured Qwen numbers in Phase 6.

## Load testing

```bash
.venv/bin/python -m bench.loadtest \
  --concurrency 1,2,4,8,16 --requests 40 \
  --out bench/results/phase1_baseline.csv
```

Closed-loop: each of N workers keeps one request in flight and sends the next
as soon as the previous returns, so the numbers measure saturation throughput.

For **overload** testing use the open-loop generator instead — the closed-loop
one cannot overload the server, because it waits for responses and therefore
self-throttles to the drain rate:

```bash
.venv/bin/python -m bench.burst --rate 20 --duration 20 --bucket 4 \
  --out bench/results/phase4_admission.csv
```

Compare two runs:

```bash
.venv/bin/python -m bench.compare \
  --before bench/results/phase2_nobatch_summary.csv \
  --after  bench/results/phase2_batching_summary.csv
```

## Phase 1 baseline results

Mock backend, 64 tokens/request, `max_workers=1` (no batching), 40 requests per
level. Full data in `bench/results/`.

| Concurrency | Throughput (rps) | p50 (ms) | p95 (ms) | p99 (ms) | Mean queue wait (ms) | Mean inference (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.79 | 559 | 564 | 565 | 0.1 | 556 |
| 2 | 1.80 | 1110 | 1119 | 1122 | 538 | 555 |
| 4 | 1.80 | 2221 | 2230 | 2234 | 1579 | 555 |
| 8 | 1.80 | 4440 | 4455 | 4457 | 3497 | 555 |
| 16 | 1.80 | 8878 | 8896 | 8901 | 6658 | 555 |

**The "before" number: throughput is flat at ~1.8 rps no matter how much
concurrency you add, and latency grows linearly with it.**

That is the signature of a serialised server. One model instance cannot run two
forward passes at once, so requests queue: inference time stays pinned at ~555ms
while queue wait absorbs all of the growth (at 16 concurrent, 75% of a request's
latency is spent waiting, not generating). Adding load adds latency and buys no
extra work done.

Phase 2 attacks exactly this by batching queued requests into a single forward
pass, so a deeper queue produces bigger batches instead of longer waits.

## Phase 2 results: queue + dynamic batching

Same mock, same 64 tokens/request, same single-worker executor. The only change
is that a scheduler now groups queued requests into one forward pass
(`MAX_BATCH_SIZE=8`, `MAX_WAIT_MS=10`).

| Concurrency | Throughput (rps) | p50 (ms) | p95 (ms) | Mean queue wait (ms) | Mean batch size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.75 | 570 | 576 | 10.5 | 1.00 |
| 2 | 3.28 | 609 | 618 | 10.1 | 2.00 |
| 4 | 5.71 | 699 | 706 | 8.9 | 4.00 |
| 8 | 9.22 | 867 | 874 | 3.4 | 8.00 |
| 16 | 9.44 | 1680 | 1713 | 661.6 | 8.00 |

### Before / after

The "before" column is a **control run on this same code** with
`MAX_BATCH_SIZE=1`, not the Phase 1 binary. It reproduces the Phase 1 numbers to
within noise (1.80 rps, p50 8892ms vs 8878ms), which is what rules out the
improvement being an artifact of unrelated changes between the two phases.

| Concurrency | rps before | rps after | Gain | p50 before | p50 after | p50 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.79 | 1.75 | **0.98x** | 559.8 | 569.5 | **+9.7** |
| 2 | 1.80 | 3.28 | **1.82x** | 1110.9 | 609.3 | −501.6 |
| 4 | 1.80 | 5.71 | **3.18x** | 2222.2 | 699.3 | −1522.8 |
| 8 | 1.80 | 9.22 | **5.13x** | 4445.0 | 867.2 | −3577.8 |
| 16 | 1.80 | 9.44 | **5.25x** | 8892.5 | 1680.0 | −7212.6 |

**5.25x peak throughput, and p50 latency 5.3x lower at 16 concurrent.**

Batching improving latency looks paradoxical — batching *adds* work to each
forward pass. It happens because at any concurrency above 1, latency was
dominated by queue wait, not by inference. Serving 8 requests in one 844ms pass
instead of eight serial 555ms passes drains the queue ~5x faster, and the queue
wait that disappears is far larger than the ~290ms the wider batch adds.

### What it costs

At **concurrency 1, batching is a 2% throughput regression and +10ms latency.**
That is `MAX_WAIT_MS` doing exactly what it is configured to do: a lone request
with no traffic to batch with waits out the full 10ms deadline before being
dispatched alone. This is the real tradeoff, and it is the right shape — the
cost is paid only when the system has spare capacity, and it buys 5x when the
system does not.

`MAX_WAIT_MS` is the knob: raise it to catch bigger batches under bursty load at
the cost of more idle-case latency; set it to 0 and batches only ever contain
what is already queued.

### Where the 16-concurrency number comes from

Throughput plateaus at 9.4 rps between concurrency 8 and 16 because
`MAX_BATCH_SIZE=8` caps it — mean batch size stays at 8.00 and the surplus 8
requests wait for the next pass, which is why queue wait jumps back to 662ms.
The system is saturated at that point; more concurrency buys latency, not
throughput. Raising `MAX_BATCH_SIZE` moves the plateau, and finding where it
stops moving on real hardware is a Phase 6 question.

### Verifying it from the logs

`logs/phase2_batching.jsonl` has one JSON record per request with the fields the
spec's measurement plan requires: `enqueued_at`, `dequeued_at`, `queue_wait_ms`,
`batch_size`, `inference_ms`, `e2e_ms`, `queue_depth_at_enqueue`. Across the
202-request sweep, 82 batches were dispatched and batch size tracked offered
concurrency exactly:

| Batch size | Requests served in it |
| ---: | ---: |
| 1 | 42 |
| 2 | 40 |
| 4 | 40 |
| 8 | 80 |

## Phase 4 results: backpressure / admission control

Phase 2 ended with the system saturated: throughput pinned at ~9.4 rps past
concurrency 8, queue wait climbing again. The queue was unbounded, so offered
load beyond the drain rate had nowhere to go but into everyone's latency.

Admission control rejects a new request with **503 Service Unavailable** once
`MAX_QUEUE_DEPTH` requests are already waiting. `MAX_QUEUE_DEPTH=0` disables it,
which is the control condition below.

### Choosing the threshold

Queue depth alone is meaningless — what matters is how long it takes to *drain*.
Depth converts to promised latency at a fixed rate:

```
wait per queued request = batch_time / max_batch_size = 840ms / 8 ≈ 105ms
```

So the threshold is a latency budget in disguise. Working backwards from a
target:

| Target p99 end-to-end | Wait budget (minus own 840ms pass) | Implied depth |
| ---: | ---: | ---: |
| 1.5 s | 660 ms | ~6 |
| **2.5 s** | **1,660 ms** | **~16** |
| 5.0 s | 4,160 ms | ~40 |

**16 was chosen** — a tidy `2 × MAX_BATCH_SIZE`, i.e. *"a request may sit behind
at most two batches of backlog."* Bigger is not safer: a deep queue is a latency
bomb, because work accepted but not drained in time means the client times out
anyway and the GPU burns a pass on a response nobody is waiting for. Smaller is
not safer either: arrivals are jittery, and a shallow queue turns every
momentary clump into a rejection while capacity sits idle.

The prediction was checked against the run. Predicted worst-case queue wait
1,660ms; **measured maximum 1,693ms — within 2%.**

### Why 503 and not 429

The deciding question is whose fault the rejection is. `429 Too Many Requests`
means *you, the client, sent too much* — the code for a per-client rate limit or
quota. Nothing here is measured per client: the check is global queue depth, so
a client's very first request is refused if it arrives at a bad moment. That is
server capacity, which is what `503` means. A `Retry-After` header carries the
back-off signal.

The honest counterargument: many load balancers treat 503 as "this backend is
unhealthy, eject it," which is wrong here — the server is fine, just full.
HuggingFace's TGI returns 429 for a full queue for that reason. Behind such a
load balancer, switch to 429.

### The test: sustained overload, 20 rps offered against a ~9.4 rps system

This needs an **open-loop** generator (`bench/burst.py`). The Phase 2 closed-loop
tester structurally cannot overload anything — its workers wait for a response
before sending again, so offered load throttles itself to whatever the server can
serve. Overload testing requires sending on a fixed schedule regardless of
whether the server is keeping up.

```bash
MAX_QUEUE_DEPTH=0  uvicorn app.main:app --port 8000   # control
MAX_QUEUE_DEPTH=16 uvicorn app.main:app --port 8000   # protected
.venv/bin/python -m bench.burst --rate 20 --duration 20 --bucket 4
```

| | Admission control OFF | Admission control ON |
| --- | ---: | ---: |
| Offered | 400 @ 20 rps | 400 @ 20 rps |
| Accepted | 400 (100%) | 201 (50.2%) |
| Rejected | 0 | 199 |
| **Accepted p50** | **11,996 ms** | **2,312 ms** |
| **Accepted p99** | **22,831 ms** | **2,535 ms** |
| Accepted max | 23,021 ms | 2,543 ms |
| Rejection p50 | — | **3.9 ms** |
| Peak queue depth | **214** | **16** |
| Wall clock to drain | 42.7 s | 21.7 s |
| **Goodput** | **9.37 rps** | **9.27 rps** |

### The headline: latency over time

Bucketed by when load was offered — this is the actual proof, not the 503s:

| Offer window | OFF: p50 | OFF: max | ON: p50 | ON: max |
| ---: | ---: | ---: | ---: | ---: |
| 0–4 s | 3,114 ms | 5,363 ms | 2,182 ms | 2,536 ms |
| 4–8 s | 7,566 ms | 9,804 ms | 2,322 ms | 2,515 ms |
| 8–12 s | 11,996 ms | 14,239 ms | 2,347 ms | 2,543 ms |
| 12–16 s | 16,428 ms | 18,657 ms | 2,353 ms | 2,516 ms |
| 16–20 s | 20,841 ms | 23,021 ms | 2,325 ms | 2,529 ms |

**Unprotected, latency grows without bound for as long as the overload lasts** —
p50 climbs linearly 3.1s → 20.8s and would keep going. **Protected, it is flat**:
2.18s → 2.33s across the entire run, never exceeding 2.55s.

### The finding that actually matters

**Goodput is 9.37 rps unprotected vs 9.27 rps protected — statistically
identical.** Admission control gave up about 1% of useful work.

That is the whole argument. The unprotected server was never doing *more* work;
it was doing the same work while also holding 214 requests hostage and quoting
everyone a 20-second latency. Rejecting the excess in 3.9ms costs essentially
nothing in throughput and buys a latency bound — and a client told "no" in 4ms
can retry, shed load, or fail over, while a client waiting 23 seconds has already
timed out and burned a GPU pass on an answer nobody will read.

`logs/phase4_*.jsonl` carries the audit trail: `queue_depth_at_enqueue` reaches
214 unprotected and never exceeds 15 protected, with every rejection recorded as
its own `"event": "rejected"` record.

## Phase 4.1: observability — Prometheus + Grafana

Everything measured so far was reconstructed after the fact from CSVs and JSONL.
Phase 4.1 makes the same results readable **live, from scrapes**, before the GPU
work — so Phase 6 produces real dashboard graphs instead of static charts
rebuilt afterward.

### Two layers, not one

`app/metrics.py` (JSONL) and `app/telemetry.py` (Prometheus) are not redundant:

| | Answers | Keeps |
| --- | --- | --- |
| **JSONL** | "what happened to request #4182?" | per-request identity |
| **Prometheus** | "what is p99 doing right now?" | aggregate trend |

Prometheus cannot reconstruct the first — histograms discard identity. The JSONL
cannot cheaply answer the second over a live window. Working rule #2 depends on
the former; the dashboards depend on the latter.

### What is exported

Nine metrics, at `/metrics`. The only label anywhere is `outcome`
(`served`/`rejected`) with two values — nothing is labelled per-request,
per-prompt or per-client, which is how a metrics endpoint becomes an outage.

```
llm_requests_total{outcome}      counter    goodput and shed load
llm_batches_total                counter    forward passes
llm_queue_depth                  gauge      live, via scrape-time callback
llm_queue_depth_peak             gauge      high-water mark (1s scrapes miss spikes)
llm_queue_depth_limit            gauge      config, so panels draw the limit line
llm_max_batch_size               gauge      config
llm_queue_wait_seconds           histogram  1ms … 30s
llm_inference_seconds            histogram  fine across 0.4–1.0s
llm_request_duration_seconds     histogram  fine across 2.0–3.0s
llm_batch_size                   histogram  observed per batch
```

Bucket edges are chosen against measured ranges, not left at defaults — with
defaults, nearly every observation lands in one bucket and the quantiles are
worthless.

### Running it

`docker compose up -d` (see Quick start). Datasource and dashboard are
auto-provisioned from `ops/grafana/`, so they are version-controlled config
rather than hand-built UI state that dies with the container.

### Verification

`ops/verify_dashboards.py` **reads the PromQL out of the dashboard JSON** and
runs each expression against Prometheus. Because the queries are extracted
rather than retyped, a passing run proves the panels work — not that some
parallel set of queries works.

```bash
python ops/verify_dashboards.py --window 60
```

Both Phase 4 conditions were re-run under live scraping. The dashboards
reproduce both:

| Panel query | Protected (limit 16) | Unprotected (limit 0) |
| --- | ---: | ---: |
| `llm_queue_depth` max | **16** | **207** |
| `llm_queue_depth_peak` | **16** | **215** |
| e2e p50 | 2.30 s | **16.81 s** |
| e2e p95 | 2.58 s | **28.12 s** |
| served / rejected totals | 202 / 199 | 401 / 0 |
| mean batch size | 7.59 | 8.00 |

The Phase 2 result is visible in the same dashboard: `queue wait p50` 1.45s
against `inference p50` 0.824s — the split that shows the bottleneck is the line,
not the model.

The `admission limit` series is `llm_queue_depth_limit > 0`, so it *disappears*
when admission control is off rather than drawing a threshold line that isn't
being enforced.

### One defect the verification caught

The dashboard initially reported **p99 = 2.93s** where the burst test measured
**2.53s** — a 16% overstatement. Cause: `histogram_quantile` assumes
observations spread uniformly within a bucket, and the `le` edges jumped
2.5 → 3.0, straddling exactly where p99 landed. Since `MAX_QUEUE_DEPTH` is
*derived* from a 2.5s p99 target, that was the worst possible place to be coarse.

Adding edges at 2.25 and 2.75 cut the error to 7% (2.71s vs 2.54s), with p50
within 0.4%.

**Buckets belong where the decisions are.** A histogram quantile is an estimate,
and it is only as good as the resolution near the number you actually act on.
The tail above 20s stays coarse on purpose — the difference between a 23s and a
29s p99 changes no decision, since both mean the same thing.

## Phase 4.2: containerization

One `Dockerfile` for the app, one `docker-compose.yml` for the whole stack.
See **Quick start** above for usage.

**Cold start from nothing** — `docker compose down -v`, `build --no-cache`,
`up -d` — brings all three services healthy in **6.2 seconds**, with the Grafana
dashboard provisioned onto a wiped volume and Prometheus already scraping. No
manual steps.

### Decisions worth defending

**Python 3.12 in the image, though development was on 3.9.** The code is
3.9-compatible, but there is no reason to ship an interpreter three years older
than necessary, and Phase 6 adds torch.

**Dependencies copied before source.** Editing a `.py` file then rebuilds in
~2s instead of ~40s, because the `pip install` layer stays cached.

**Non-root (uid 10001).** Nothing here needs root, and Phase 4.3 puts this image
on the public internet. This is also why logs go to a *named volume* rather than
a bind mount — a bind mount arrives owned by the host user, and the container
user cannot write to it.

**Health check uses the interpreter, not curl.** `python:3.12-slim` ships
neither curl nor wget, and installing one purely for a health check is a wasted
layer plus extra attack surface when Python is already present.

**Prometheus waits on `service_healthy`, not just container start.** Otherwise
it records a stretch of failed scrapes during app startup and every panel opens
with a gap.

**Grafana opens directly on the dashboard**
(`GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH`). Grafana's default home page
lists only dashboards *you have already visited*, so a freshly provisioned
dashboard is invisible on a new browser — it looks like provisioning failed
when it did not.

**Image is 270MB**, essentially all base image and dependencies.

### Containerizing changed nothing measurable

The Phase 4 comparison re-run against the containerized app:

| | Host process | Container |
| --- | ---: | ---: |
| Accepted / rejected | 201 / 199 | 204 / 196 |
| Accepted p50 | 2,312 ms | 2,290 ms |
| Accepted p99 | 2,534 ms | 2,519 ms |
| Goodput | 9.28 rps | 9.21 rps |
| Peak queue depth | 16 | 16 |

Unprotected, via `MAX_QUEUE_DEPTH=0 docker compose up -d app`: p50 **11,788 ms**,
p99 **22,480 ms**, peak depth **210** — reproducing the host result. The
`.dockerignore` keeps `.venv/`, `logs/` and `bench/results/` out of the build
context; the venv alone is hundreds of MB and its binaries are built for the
host, so copying it into a Linux image would be both slow and wrong.

### A note on making the mock honest

Both backends are blocking functions dispatched to a `ThreadPoolExecutor` with
`max_workers=1`. If the mock instead `await asyncio.sleep()`-ed, concurrent
requests would all "infer" in parallel, throughput would appear to scale
linearly, and Phase 2's batching would look like a pointless regression against
an impossible baseline. Serialising through one worker reproduces the real
constraint a single model instance imposes.
