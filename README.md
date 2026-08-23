# LLM Inference Serving Layer

Request queueing, dynamic batching, and backpressure for a single-node LLM
server. See `llm_serving_project_spec.md` for the full spec and phase plan.

> The architecture writeup, design-tradeoff discussion, and benchmark graphs
> are Phase 7 deliverables. This file is setup and run instructions only.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements-model.txt` (torch/transformers) is **not** needed until Phase 6.
Phases 1-5 run entirely on the CPU mock backend.

## Run the server

```bash
.venv/bin/python -m uvicorn app.main:app --port 8000
```

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

### A note on making the mock honest

Both backends are blocking functions dispatched to a `ThreadPoolExecutor` with
`max_workers=1`. If the mock instead `await asyncio.sleep()`-ed, concurrent
requests would all "infer" in parallel, throughput would appear to scale
linearly, and Phase 2's batching would look like a pointless regression against
an impossible baseline. Serialising through one worker reproduces the real
constraint a single model instance imposes.
