# LLM Inference Serving Layer — Project Spec

## Overview

Build a scoped, production-flavored serving layer for a small open-weight LLM: request queueing, dynamic batching, backpressure, token streaming, and priority scheduling — the same core problems solved by systems like vLLM, TensorRT-LLM, and Ray Serve, at a scale one person can build and fully understand in \~7 weeks. The system is instrumented with Prometheus/Grafana, containerized, and deployed live as a CPU-only public demo.

**Learning goal:** genuinely understand the systems tradeoffs behind LLM serving — batching vs. latency, queueing under load, backpressure, admission control — well enough to defend design decisions in a system design interview, not just claim the buzzwords.

**Resume goal:** produce a project with real, self-measured benchmark numbers (not estimates) that reads as applied infra work, not another RAG demo.

**Target companies:** Databricks (Mosaic AI / Model Serving is literally this problem), AI infra teams at Amazon (Bedrock), Microsoft (Azure AI / Copilot infra), Google (Vertex AI), and any AI-forward startup. Secondary relevance to Stripe and similar infra-heavy companies via the underlying queueing/backpressure/observability concepts, which are domain-agnostic systems skills.

---

## Architecture

Client

  │  HTTP request

  ▼

FastAPI endpoint  ──────────────►  Request Queue (asyncio.Queue)

                                          │

                                          ▼

                              Scheduler Loop (background task)

                              \- pulls requests off queue

                              \- groups into a batch when:

                                  batch reaches max\_batch\_size, OR

                                  max\_wait\_ms elapses

                              \- checks queue depth → admission control

                                          │

                                          ▼

                                  Inference call

                          (CPU mock in dev → real model on GPU later)

                                          │

                                          ▼

                              Response returned / streamed to client

Logging/metrics collected at every stage: queue depth, wait time,

batch size, inference time, end-to-end latency.

---

## Tech Stack

- **Model:** Qwen2.5-1.5B-Instruct or Llama-3.2-1B (small enough to run on a single consumer GPU or a cheap cloud rental)  
- **Serving framework:** Python, FastAPI, `asyncio`  
- **Model runtime:** Hugging Face `transformers`  
- **Queue:** in-process `asyncio.Queue` (no need for Redis/external broker at this scope — keep it simple, single-node)  
- **Load testing:** Locust, or a custom `asyncio`\-based request generator  
- **Metrics:** two layers, both core. Per-request structured logging (JSON lines) from Phase 2 as the raw audit trail, plus **Prometheus \+ Grafana from Phase 4.1** — a `/metrics` endpoint exposing queue depth, batch size, latency and throughput, with dashboards on top. Prometheus is no longer an if-time-allows item: it lands *before* the GPU work so Phase 6 produces real dashboard graphs directly.  
- **Containerization:** Docker \+ docker-compose (Phase 4.2) — one Dockerfile for the app, one compose file bringing up app \+ Prometheus \+ Grafana together  
- **Deployment:** a PaaS free/hobby tier (Fly.io, Railway, or Render) hosting the CPU mock as a live public demo (Phase 4.3). Single replica — this is for visibility, not scale.  
- **GPU access:** local GPU if available, otherwise Colab or a cheap rental (RunPod / Lambda Labs, \~$0.20–0.50/hr) — only needed for Phase 6 real benchmarking

---

## Development Strategy: Mock First, GPU Last

Validate all scheduling logic (queueing, batching, backpressure) against a **mocked model** — a fake inference function that just `sleep()`s for N milliseconds to simulate compute time. This lets you build and debug the entire system on CPU, for free, before spending any GPU rental time. Swap in the real model only once the logic is correct, for Phase 6 benchmarking.

---

## Phase Plan (\~7-week realistic timeline)

Listed in **execution order**, which is deliberately not numeric order. Phases 4.1–4.3 were inserted after Phase 4; Phases 3 and 5 now follow the deploy, so their features ship as a new container image and arrive as new Grafana panels rather than a separate instrumentation effort. Both still land before Phase 6, which matters — time-to-first-token is a Phase 6 metric, so streaming has to exist before the GPU session to be measured there.

| Phase | What | Time | Priority | Status |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Baseline: single-request FastAPI endpoint, no batching | 1 week | **Core** | ✅ done |
| 2 | Request queue \+ dynamic batching scheduler \+ basic logging (queue depth, batch size, latency) from day one, using the CPU mock | 1.5 weeks | **Core** | ✅ done |
| 4 | Backpressure / admission control under simulated overload | 3–5 days | **Core** | ← next |
| 4.1 | Prometheus instrumentation \+ Grafana dashboards — `/metrics` exposing queue depth, batch size, latency, throughput | 1 day | **Core** |  |
| 4.2 | Dockerize — one Dockerfile, one docker-compose covering app \+ Prometheus \+ Grafana | 0.5 day | **Core** |  |
| 4.3 | Deploy the CPU mock to a PaaS (Fly.io / Railway / Render) as a live public demo | 0.5 day | **Core** |  |
| 3 | Token streaming (SSE) to the client | 3–5 days | **Core** |  |
| 5 | Priority tiers (premium vs. free request scheduling) | 3–5 days | **Core** |  |
| 6 | Swap in real model on GPU, rerun load tests, capture final benchmark numbers | 1 week | **Core** |  |
| 7 | Writeup: README, design decisions, benchmark graphs, live demo link, dashboard screenshots | 2–3 days | **Core** |  |

**Total: \~6.5–7.5 weeks** part-time (30.5–37.5 working days), accounting for class workload eating into available hours. Up from the original \~6–6.5 weeks: \+2 days for the three new items, and the range now assumes **nothing is cut**.

### Why 4.1–4.3 come before Phase 6

Both orderings are deliberate, and both are about making the GPU session productive rather than about polish:

- **Dockerizing before Phase 6** means the GPU rental step deploys a known, working image instead of hand-configuring a fresh box under time pressure while the meter runs.  
- **Building the Grafana dashboards before Phase 6** means the benchmark session produces real dashboard graphs directly, instead of finishing the run and then rebuilding static charts from CSVs afterward.

### Priority

**Everything in the table is core. The plan is to ship all of it, including both formerly-stretch phases (3 and 5).**

The three additions (4.1–4.3) earn more externally-visible signal per day invested than either stretch phase would: a live deployed service and real observability dashboards are things an interviewer can *look at*, and they cost two days combined.

The story at completion: "I built a system that batches concurrent LLM requests, protects itself under overload, streams tokens, schedules by priority, and exports real metrics — here is the live demo, here are the dashboards, and here are the measured throughput and latency numbers proving it works."

*Last-resort fallback only, not the plan of record:* if the timeline genuinely collapses, drop Phase 5 first and Phase 3 second. Phases 1, 2, 4, 4.1, 4.2, 4.3, 6 and 7 are never cut.

---

## Measurement Plan (don't defer this to the end)

Start logging from **Phase 2**, not Phase 6\. Every request should record, at minimum:

- Timestamp enqueued / timestamp dequeued (→ queue wait time)  
- Batch size it was served in  
- Inference time for that batch  
- End-to-end latency (enqueue → response returned)  
- Queue depth at time of enqueue

This is what lets you actually verify the batching improvement as you build it — e.g., confirming throughput increased after Phase 2 — instead of reconstructing evidence retroactively in Phase 6\. Crude JSONL logging is enough for Phases 2–4; the dashboards arrive in Phase 4.1 and are core, not a nice-to-have. The two layers then coexist permanently and do different jobs: Prometheus aggregates into time series for dashboards, while the per-request JSONL keeps the granularity that aggregates cannot reconstruct.

**Metrics to capture at each milestone:**

- **After Phase 1 (baseline):** requests/sec at 1 concurrent request; latency at low vs. high concurrency with no batching (this is your "before" number)  
- **After Phase 2 (batching):** requests/sec and latency at the same concurrency levels — compute the throughput delta vs. Phase 1  
- **After Phase 4 (backpressure):** behavior under a deliberate burst load — confirm the system degrades gracefully (rejects excess requests) instead of latency collapsing for everyone  
- **After Phase 4.1 (observability):** the Phase 2 batching result and the Phase 4 rejection behaviour both reproduced live on a Grafana dashboard, driven by scrapes rather than CSVs — if the dashboards can't retell the story the CSVs already told, the instrumentation is wrong  
- **After Phase 4.3 (deploy):** the public URL serving `/generate` and `/healthz`, with a load test run against it  
- **After Phase 6 (real GPU benchmark):** final p50/p95/p99 latency, max sustained throughput, time-to-first-token (streaming ships in Phase 3, before this), and a recalibrated `MOCK_BATCH_ALPHA` measured against the real model's batch curve

---

## Deliverables

1. GitHub repo with clean commit history reflecting the phase progression  
2. README covering: problem statement, architecture, design tradeoffs made and why, benchmark results with graphs, **a link to the live demo, and Grafana dashboard screenshots** — not just CSV-derived charts  
3. A load-testing script/report showing before/after batching numbers  
4. **A live, publicly reachable mock deployment** (CPU-only, single replica) that an interviewer can hit themselves  
5. **A one-command local stack** — `docker compose up` brings up the app plus Prometheus and Grafana with dashboards preloaded  
6. Resume bullet(s), to be filled in once real numbers exist, e.g.:  
   - "Built an LLM inference serving layer with dynamic request batching and backpressure, improving throughput by **\[X\]x** over naive single-request serving while maintaining p95 latency under \*\*\[Y\]\*\*ms at **\[Z\]** concurrent requests"  
   - "Implemented admission control to reject excess load under simulated traffic bursts, preventing latency collapse observed in the unprotected baseline"

---

## Setup Notes

- Estimated GPU cost for the full project: low single-digit dollars if using a cheap on-demand rental only for Phase 6 (a few hours of testing at $0.20–0.50/hr).  
- Everything before Phase 6 — Phases 1, 2, 4, 4.1, 4.2, 4.3, 3 and 5, including the live deployment — runs on CPU with the mocked inference function. No GPU spend is required until Phase 6\.  
- The Phase 4.3 deploy targets a **free or hobby tier** (Fly.io, Railway, Render). The mock needs no GPU and very little memory, so hosting stays free or near-free and the project's total spend stays in low single-digit dollars.  
- Keep the scope to a single node / single model instance. Multi-replica autoscaling is an interesting extension but out of scope for the 6-week core plan — mention it as a "future work" item in the README if you want to signal awareness of it without committing to building it.

