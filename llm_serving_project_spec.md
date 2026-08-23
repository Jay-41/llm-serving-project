# LLM Inference Serving Layer — Project Spec

## Overview

Build a scoped, production-flavored serving layer for a small open-weight LLM: request queueing, dynamic batching, backpressure, and (optionally) streaming and priority scheduling — the same core problems solved by systems like vLLM, TensorRT-LLM, and Ray Serve, at a scale one person can build and fully understand in \~6 weeks.

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
- **Metrics:** simple structured logging to start; optionally Prometheus \+ a Grafana dashboard if time allows in Phase 6  
- **GPU access:** local GPU if available, otherwise Colab or a cheap rental (RunPod / Lambda Labs, \~$0.20–0.50/hr) — only needed for Phase 6 real benchmarking

---

## Development Strategy: Mock First, GPU Last

Validate all scheduling logic (queueing, batching, backpressure) against a **mocked model** — a fake inference function that just `sleep()`s for N milliseconds to simulate compute time. This lets you build and debug the entire system on CPU, for free, before spending any GPU rental time. Swap in the real model only once the logic is correct, for Phase 6 benchmarking.

---

## Phase Plan (6-week realistic timeline)

| Phase | What | Time | Priority |
| :---- | :---- | :---- | :---- |
| 1 | Baseline: single-request FastAPI endpoint, no batching | 1 week | **Core** |
| 2 | Request queue \+ dynamic batching scheduler \+ basic logging (queue depth, batch size, latency) from day one, using the CPU mock | 1.5 weeks | **Core** |
| 3 | Token streaming (SSE) to the client | 3–5 days | Stretch |
| 4 | Backpressure / admission control under simulated overload | 3–5 days | **Core** |
| 5 | Priority tiers (premium vs. free request scheduling) | 3–5 days | Stretch |
| 6 | Swap in real model on GPU, rerun load tests, capture final benchmark numbers | 1 week | **Core** |
| 7 | Writeup: README, design decisions, benchmark graphs | 2–3 days | **Core** |

**Total: \~6–6.5 weeks** part-time, accounting for class workload eating into available hours.

### Cut line

If time runs short, the **non-negotiable core path is Phases 1, 2, 4, 6, and 7** — baseline, batching, backpressure, real benchmarks, and a documented writeup. That alone is a complete, defensible story: "I built a system that batches concurrent LLM requests, protects itself under overload, and I can show you the measured throughput and latency numbers proving it works."

Phase 5 (priority scheduling) is the first thing to cut. Phase 3 (streaming) is the second — nice for UX polish and worth doing if time allows, but not load-bearing for the core narrative.

---

## Measurement Plan (don't defer this to the end)

Start logging from **Phase 2**, not Phase 6\. Every request should record, at minimum:

- Timestamp enqueued / timestamp dequeued (→ queue wait time)  
- Batch size it was served in  
- Inference time for that batch  
- End-to-end latency (enqueue → response returned)  
- Queue depth at time of enqueue

This is what lets you actually verify the batching improvement as you build it — e.g., confirming throughput increased after Phase 2 — instead of reconstructing evidence retroactively in Phase 6\. Even crude `print`/CSV logging is enough at this stage; a real dashboard is a Phase 6 nice-to-have, not a Phase 2 requirement.

**Metrics to capture at each milestone:**

- **After Phase 1 (baseline):** requests/sec at 1 concurrent request; latency at low vs. high concurrency with no batching (this is your "before" number)  
- **After Phase 2 (batching):** requests/sec and latency at the same concurrency levels — compute the throughput delta vs. Phase 1  
- **After Phase 4 (backpressure):** behavior under a deliberate burst load — confirm the system degrades gracefully (rejects excess requests) instead of latency collapsing for everyone  
- **After Phase 6 (real GPU benchmark):** final p50/p95/p99 latency, max sustained throughput, time-to-first-token if streaming was built

---

## Deliverables

1. GitHub repo with clean commit history reflecting the phase progression  
2. README covering: problem statement, architecture, design tradeoffs made and why, and benchmark results with graphs  
3. A load-testing script/report showing before/after batching numbers  
4. Resume bullet(s), to be filled in once real numbers exist, e.g.:  
   - "Built an LLM inference serving layer with dynamic request batching and backpressure, improving throughput by **\[X\]x** over naive single-request serving while maintaining p95 latency under \*\*\[Y\]\*\*ms at **\[Z\]** concurrent requests"  
   - "Implemented admission control to reject excess load under simulated traffic bursts, preventing latency collapse observed in the unprotected baseline"

---

## Setup Notes

- Estimated GPU cost for the full project: low single-digit dollars if using a cheap on-demand rental only for Phase 6 (a few hours of testing at $0.20–0.50/hr).  
- Phases 1–5 can be built and fully validated on CPU with the mocked inference function — no GPU spend required until Phase 6\.  
- Keep the scope to a single node / single model instance. Multi-replica autoscaling is an interesting extension but out of scope for the 6-week core plan — mention it as a "future work" item in the README if you want to signal awareness of it without committing to building it.

