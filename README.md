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

## Live demo

**https://llm-serving-demo.onrender.com**

```bash
curl https://llm-serving-demo.onrender.com/healthz
curl -X POST https://llm-serving-demo.onrender.com/generate \
  -H 'content-type: application/json' -d '{"prompt": "hello", "max_tokens": 32}'
curl https://llm-serving-demo.onrender.com/metrics
```

The deployed instance runs the **mock backend on CPU** — it demonstrates the
queueing, batching and admission-control behaviour, not model quality. It is
also **not** where benchmark numbers come from: those are local runs, and Phase
6 on a GPU.

Two things to expect from Render's free tier:

- The instance **sleeps after ~15 minutes idle**, so the first request after a
  quiet spell takes 30–60s to cold-start. Subsequent ones are normal.
- **0.1 CPU / 512MB.** The mock spends its time in `sleep()` rather than
  burning CPU, so latency tracks the local figures; sustained throughput under
  heavy concurrency will not.

## Deploying

`render.yaml` is a Render Blueprint, so the service configuration is
version-controlled rather than click-ops nobody can reproduce.

1. Sign in at [dashboard.render.com](https://dashboard.render.com) and
   authorize the Render GitHub app for this repository (it is private, so
   Render needs explicit access — granting it to just this repo is enough).
2. **New → Blueprint**, select `llm-serving-project`, branch `main`.
3. Render reads `render.yaml` and proposes one free web service,
   `llm-serving-demo`. Apply.
4. First build takes a few minutes. `healthCheckPath: /healthz` means Render
   waits for a 200 before routing traffic, so a broken build never replaces a
   working one.

Only the **app** is deployed. Prometheus and Grafana stay local — the compose
stack runs Grafana with anonymous admin access, which is fine on localhost and
reckless on a public URL. The public surface is `/generate`, `/healthz` and
`/metrics`.

`MAX_ALLOWED_TOKENS` is 128 in `render.yaml` versus 512 locally: the endpoint is
public and unauthenticated, and mock cost scales linearly with `max_tokens`, so
this bounds what one caller can make the instance sleep for.

### Verified against the live instance

The Phase 4 overload test, run from a laptop against the public URL — 300
requests at 20 rps for 15s:

| | Local container | Live on Render |
| --- | ---: | ---: |
| Accepted / rejected | 204 / 196 | 153 / 147 |
| Accepted p50 | 2,290 ms | 2,409 ms |
| Accepted p99 | 2,519 ms | 2,717 ms |
| Goodput | 9.21 rps | 9.13 rps |
| Peak queue depth | 16 | 16 |
| Rejection p50 | 3.9 ms | 94.6 ms |

Accepted latency stayed flat across the run (2,350 → 2,428 → 2,425 ms by 5s
window), so admission control holds from across the internet, not just on
localhost. The ~100 ms gap in accepted latency and the 94 ms rejection cost are
the same number: network round-trip to Oregon. The server still refuses in a
few milliseconds; the client just has to cross the country to hear it.

The mock cost model also survives the move to 0.1 CPU — a 32-token request
reported `inference_ms: 301` against a predicted 40 + 8 × 32 = 296 — because
`sleep()` does not care how much CPU it has. Data:
`bench/results/phase43_render*.csv`.

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
| `MAX_QUEUE_DEPTH_FREE` | `8` | Free-tier admission limit. Equal to `MAX_QUEUE_DEPTH` (or `0`) disables tiering |
| `AGING_MS` | `2000` | Free request promoted to paid priority after waiting this long. `0` = strict priority |
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

## Phase 3: token streaming (SSE)

`POST /generate` with `"stream": true` returns `text/event-stream`: one `token`
event per generated token, then a single `done` event carrying the same timing
fields a non-streaming response would have returned.

```bash
curl -N -X POST https://llm-serving-demo.onrender.com/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "hello", "max_tokens": 32, "stream": true}'
```

```
event: token
data: {"token": "the "}

event: token
data: {"token": "quick "}
...
event: done
data: {"request_id": 7, "batch_size": 1, "tokens": 32, "ttft_ms": 61.2, "e2e_ms": 310.4, ...}
```

### Streaming and batching are not in tension

The worry: batching runs 8 requests together and returns when all are done;
streaming wants each token the instant it exists. How can you stream from a
batch?

Because that is not how a transformer generates. It runs a **decode loop** —
one forward pass per token — and every pass advances *every* sequence in the
batch by one token, in lockstep. Batching was always producing tokens
incrementally; Phase 2 just collected them in a bucket and returned the bucket.
Phase 3 hands each sequence its token after each step instead.

So the backend became a generator yielding one step at a time. A non-streaming
response is just every step concatenated. One code path through the model, two
delivery modes, and both can share a batch.

**Throughput is unchanged**, which is the first thing to verify:

| Concurrency | Phase 2 rps | Phase 3, non-streaming | Phase 3, streaming |
| ---: | ---: | ---: | ---: |
| 1 | 1.75 | 1.76 | 1.77 |
| 8 | 9.22 | 9.37 | 9.39 |
| 16 | 9.44 | 9.46 | 9.51 |

Streaming changes *when* bytes arrive, not how much work the model does.

### What streaming delivers: time to first token

| Concurrency | TTFT p50 | Full response p50 | First output arrives |
| ---: | ---: | ---: | ---: |
| 1 | 61 ms | 566 ms | **9.4× sooner** |
| 4 | 64 ms | 693 ms | **10.8× sooner** |
| 8 | 61 ms | 851 ms | **14.0× sooner** |
| 16 | 888 ms | 1,680 ms | 1.9× sooner |

Under capacity, the first token lands in ~60ms — queue wait plus prefill —
while the complete response takes 550–850ms. Measured client-side, tokens then
arrive every **8.0ms**, exactly the modelled per-token cost. Without streaming
the user stares at nothing for the full duration and gets everything at once.

`ttft_ms` is reported on **every** response, streaming or not, because it is
what streaming *would* have delivered. The gap between `ttft_ms` and `e2e_ms`
is precisely the wait streaming removes from the user's experience.

### The static-batching cost, measured

At concurrency 16, TTFT jumps to 888ms. The distribution is bimodal:

```
    0-100 ms  ████████                          8 requests   mean  57ms
  800-900 ms  ████████████████████████████████ 32 requests   mean 890ms
```

The first 8 got seats in the first batch. Every subsequent request had to wait
for that batch to finish — **832ms, against a modelled batch duration of
839ms** — before its own prefill could start. Batches here are *static*: once
one starts, nobody joins until it ends.

That is precisely what **continuous batching** (vLLM, TGI) removes: at every
decode step, finished sequences leave and waiting ones join, so a new arrival
waits ~one step (~13ms) rather than ~one batch (~830ms). It is out of scope
here, but the cost is measured rather than hidden: it would turn the
concurrency-16 TTFT from 890ms into roughly 70ms.

### Implementation notes worth defending

**SSE, not WebSockets.** The traffic is one-directional, plain HTTP works with
`curl` and every proxy, browsers have `EventSource` built in, and there is no
upgrade handshake. WebSockets would buy nothing and cost a protocol.

**Admission control fires before the stream opens.** An overloaded server
answers a streaming request with a plain 503 — never a 200 that opens a stream
and then dies.

**One thread hop per step, not per token per client.** The backend generator
runs on the single inference thread; each step crosses to the event loop once
via `call_soon_threadsafe`, and the scheduler fans it out to every job in the
batch. Tokens are stamped with the time they *existed*, not the time the loop
got around to them, so server-side TTFT is honest.

**Members that finish early are delivered early.** A job asking for 16 tokens in
a batch running 64 gets its `done` event at step 16 — its `e2e_ms` ends there,
not when the batch does. Its slot then idles for 48 steps, which is the waste
continuous batching reclaims.

**The mock sleeps to absolute deadlines.** Splitting one 840ms sleep into 64
small ones exposed `time.sleep()` overshoot: a couple of ms per call, compounding
to **+20%** across a batch — enough to fail the regression check for a reason
that has nothing to do with the design. Anchoring each step to the batch start
time lets one step's overshoot be absorbed by the next; total lands within 1%.

**Client disconnect does not free the slot.** The handler's generator is
cancelled; the job keeps running in its batch (a static batch cannot evict a
member) and its remaining tokens land on a queue nobody reads, bounded by
`max_tokens`. Reclaiming that slot mid-batch is, again, continuous batching.

Data: `bench/results/phase3_*`. Dashboard gains a *Time to first token* panel
(TTFT p50/p95 against full-response p50) and *Token throughput* (tokens/s).

## Phase 5: priority tiers

`POST /generate` accepts `"tier": "paid" | "free"` (default free). Two things
change: paid requests are **served first**, and under overload free requests are
**refused at a shallower queue depth** than paid. The tier is trusted as given —
this demo has no auth; a real system would derive it from the caller's identity.

```bash
curl -X POST https://llm-serving-demo.onrender.com/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "hello", "max_tokens": 32, "tier": "paid"}'
```

### Two knobs, deliberately separate

| | What it decides | Setting |
| --- | --- | --- |
| **Tiered admission** | who gets *refused* under overload | `MAX_QUEUE_DEPTH_FREE=8`, `MAX_QUEUE_DEPTH=16` |
| **Priority + aging** | who goes *first* among those admitted | `AGING_MS=2000` |

Priority buys a better seat in line, not a faster oven. A paid request that lands
in a batch with seven free ones still waits for that batch's full pass. The
latency benefit to paid is real but bounded; the *admission* benefit — being the
one who gets in when the queue is nearly full — is the larger effect.

### Tiered admission: shed the right load

Paid at 14 rps plus free at 4 rps against a ~9.4 rps system, 20 seconds:

| | Flat limits (16/16) | Tiered (free 8 / paid 16) |
| --- | ---: | ---: |
| Paid accepted | 154 / 280 (**55%**) | 197 / 280 (**70%**) |
| Free accepted | 48 / 80 (**60%**) | 5 / 80 (**6%**) |
| **Total accepted** | **202** | **202** |

Same goodput to the request — 202 either way — but the tiered version hands
those slots to paid. Under overload the queue sits between the two limits, so
free arrivals bounce while paid ones still get in: the depth from 8 to 16 is
effectively reserved for paid. The flat control even shows free doing slightly
*better* than paid (aging was promoting them), which is exactly the blind
shedding the tiers exist to fix.

### Starvation, and aging as the fix

Strict priority has a textbook failure mode. If paid traffic never lets up, a
free request already in the queue never reaches the front — every new paid
arrival jumps ahead of it. This is the OS-scheduling starvation problem, and the
standard fix is **aging**: after waiting `AGING_MS`, a free request is treated
as paid priority.

The experiment isolates queue ordering by giving both tiers the same admission
limit (16), then offers paid at 10 rps (just over capacity) with free at 1 rps,
for 30 seconds:

| | Aging off (strict priority) | Aging on (2000 ms) |
| --- | ---: | ---: |
| Free queue wait p50 | **8,330 ms** | **2,349 ms** |
| Free queue wait max | **9,741 ms** | **2,838 ms** |
| Free p99 latency | 10,578 ms | 3,677 ms |
| Free requests promoted | 0 | 25 of 28 |
| Paid queue wait p50 | 500 ms | 1,116 ms |
| Paid p99 latency | 1,879 ms | 2,429 ms |

Free-tier wait by offer time, aging off:

```
   0-5 s   p50  8,330 ms
   5-10 s  p50  9,369 ms
  10-15 s  p50  9,503 ms      ← not coming down
  15-20 s  p50  8,476 ms
  20-25 s  p50  8,157 ms
  25-30 s  p50  3,836 ms      ← only because paid load stopped
```

Aging on: flat at 2.2–2.5 s across every window.

**The bound was predicted before it was measured.** A promoted request waits at
most `AGING_MS` to be promoted, then at most one batch to be picked:
2000 + 839 = **2,839 ms**. Measured free-tier max wait: **2,838 ms.**

**The cost is real and reported.** Paid p50 wait doubled (500 → 1,116 ms). Each
aged free request takes a batch slot from paid, and paid was already overloaded,
so a starvation guarantee for free is paid for in paid-tier latency. That is the
tradeoff, and the knob is `AGING_MS`: higher means less interference with paid
and a longer worst case for free.

### Implementation notes

**A scanned list, not a heap.** `TieredQueue` keeps a plain list and finds the
best job on every take. Admission control bounds it at 16, so the scan is
trivial — and a heap could not do aging anyway. A heap orders by a key fixed at
insertion; aging means a job's priority changes just by sitting there.
Computing effective priority at take time is both simpler and the only correct
option. Within a tier it is still FIFO.

**Both label values materialised at startup**, as with `outcome` in Phase 4.1,
so a tier with no traffic reports zero rather than absence.

**Cardinality stays flat.** `tier` has exactly two values. Nothing per-client.

Data: `bench/results/phase5_*`. Dashboard gains *Queue wait by tier* and
*Admission by tier* (served/rejected per tier plus aged promotions).

### A note on making the mock honest

Both backends are blocking functions dispatched to a `ThreadPoolExecutor` with
`max_workers=1`. If the mock instead `await asyncio.sleep()`-ed, concurrent
requests would all "infer" in parallel, throughput would appear to scale
linearly, and Phase 2's batching would look like a pointless regression against
an impossible baseline. Serialising through one worker reproduces the real
constraint a single model instance imposes.
