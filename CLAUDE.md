# Project: LLM Inference Serving Layer

Full spec: see `llm_serving_project_spec.md` in this repo. Read it before starting any phase.

## What this is

A scoped, production-flavored LLM serving system: request queueing, dynamic batching, backpressure, and (optionally) streaming and priority scheduling. Goal is to genuinely understand these systems tradeoffs well enough to defend them in an interview, and to produce real, self-measured benchmark numbers for a resume bullet — not estimates.

## Tech stack

- Python, FastAPI, `asyncio`  
- Hugging Face `transformers` for the model (Qwen2.5-1.5B-Instruct or Llama-3.2-1B)  
- In-process `asyncio.Queue` — no external broker, keep it single-node  
- Locust or a custom async script for load testing  
- Structured logging (CSV or JSON lines is fine) for metrics — do not skip this in early phases

## Working rules

1. **Build against a mocked model first.** Use a fake inference function that `sleep()`s for N milliseconds instead of the real model. Validate all queueing/batching/backpressure logic on CPU with the mock before touching a real model or any GPU. Only swap in the real model for Phase 6 benchmarking.  
2. **Log from Phase 2 onward, not just at the end.** Every request should record: enqueue time, dequeue time, batch size it was served in, inference time, end-to-end latency, and queue depth at enqueue. This is required for verifying the batching improvement as it's built, not just at the finish line.  
3. **One phase at a time.** Finish and verify a phase (run it, check the logs/metrics prove the claim) before starting the next. Don't let scope creep across phase boundaries.  
4. **Core path if time is short:** Phases 1, 2, 4, 6, 7 are non-negotiable. Phase 5 (priority tiers) is the first cut, Phase 3 (streaming) is the second. Don't build 3 or 5 before 1, 2, 4, 6, and 7 are solid.  
5. **Commit after each phase**, not mid-phase, with a message referencing the phase and what was verified.

## Phase status tracker

Update this section as work progresses. Mark each phase `todo` / `in progress` / `done`, and note the key verified metric once done.

- [x] Phase 1 — `done`. Baseline single-request FastAPI endpoint (no batching). **Verified:** throughput flat at **1.80 rps** across concurrency 1→16 (mock backend, 64 tok/req, ~555ms service time); p50 latency grows linearly 559ms → 8878ms; inference time constant at ~555ms while queue wait absorbs all growth (6658ms mean wait at conc 16). Data: `bench/results/phase1_baseline*.csv`.  
- [ ] Phase 2 — Request queue \+ dynamic batching scheduler \+ basic logging, on the CPU mock. Metric: throughput/latency delta vs. Phase 1 baseline at the same concurrency levels.  
- [ ] Phase 3 (stretch) — Token streaming via SSE.  
- [ ] Phase 4 — Backpressure / admission control under simulated burst load. Metric: confirm graceful rejection instead of latency collapse under overload.  
- [ ] Phase 5 (stretch) — Priority tiers (premium vs. free scheduling).  
- [ ] Phase 6 — Swap in real model on GPU, rerun load tests at scale, capture final p50/p95/p99 latency and max sustained throughput.  
- [ ] Phase 7 — README with architecture, design tradeoffs, and benchmark graphs.

## Conventions

- Keep the scope to a single node / single model instance. Multi-replica autoscaling is explicitly out of scope — note it as future work in the README rather than building it.  
- Prefer simple, explicit code over cleverness — this project is meant to be fully understood and defensible in an interview, not just functional.

