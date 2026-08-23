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
| `DEFAULT_MAX_TOKENS` | `64` | Used when a request omits `max_tokens` |
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Phase 6 only |

Mock service time is `MOCK_BASE_MS + MOCK_PER_TOKEN_MS * max_tokens`, so at the
defaults a 64-token request takes ~552ms.

## Load testing

```bash
.venv/bin/python -m bench.loadtest \
  --concurrency 1,2,4,8,16 --requests 40 \
  --out bench/results/phase1_baseline.csv
```

Closed-loop: each of N workers keeps one request in flight and sends the next
as soon as the previous returns, so the numbers measure saturation throughput.

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

### A note on making the mock honest

Both backends are blocking functions dispatched to a `ThreadPoolExecutor` with
`max_workers=1`. If the mock instead `await asyncio.sleep()`-ed, concurrent
requests would all "infer" in parallel, throughput would appear to scale
linearly, and Phase 2's batching would look like a pointless regression against
an impossible baseline. Serialising through one worker reproduces the real
constraint a single model instance imposes.
