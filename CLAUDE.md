# Project: LLM Inference Serving Layer

Full spec: see `llm_serving_project_spec.md` in this repo. Read it before starting any phase.

## What this is

A scoped, production-flavored LLM serving system: request queueing, dynamic batching, backpressure, token streaming, and priority scheduling — instrumented with Prometheus/Grafana, containerized, and deployed live. Goal is to genuinely understand these systems tradeoffs well enough to defend them in an interview, and to produce real, self-measured benchmark numbers for a resume bullet — not estimates.

## Tech stack

- Python, FastAPI, `asyncio`  
- Hugging Face `transformers` for the model (Qwen2.5-1.5B-Instruct or Llama-3.2-1B)  
- In-process `asyncio.Queue` — no external broker, keep it single-node  
- Locust or a custom async script for load testing  
- **Per-request structured logging (JSON lines)** — the raw audit trail, one record per request. Do not skip this in early phases; it is what proves each phase's claim.  
- **Prometheus + Grafana — core, not optional.** A `/metrics` endpoint exposing queue depth, batch size, latency and throughput, with Grafana dashboards on top. Built in Phase 4.1, before the GPU work, so Phase 6 produces real dashboard graphs instead of static charts rebuilt from CSVs afterward.  
- **Docker + docker-compose** — one Dockerfile for the app, one compose file covering app + Prometheus + Grafana (Phase 4.2)  
- **A PaaS target for the live mock demo** — Fly.io, Railway, or Render, CPU-only free/hobby tier (Phase 4.3)

## Working rules

1. **Build against a mocked model first.** Use a fake inference function that `sleep()`s for N milliseconds instead of the real model. Validate all queueing/batching/backpressure logic on CPU with the mock before touching a real model or any GPU. Only swap in the real model for Phase 6 benchmarking.  
2. **Log from Phase 2 onward, not just at the end.** Every request should record: enqueue time, dequeue time, batch size it was served in, inference time, end-to-end latency, and queue depth at enqueue. This is required for verifying the batching improvement as it's built, not just at the finish line. From Phase 4.1 onward the same signals are *also* exported as Prometheus metrics — the two layers coexist and do different jobs: Prometheus aggregates into time series for dashboards and alerting, while the JSONL keeps per-request granularity that aggregates cannot reconstruct.  
3. **One phase at a time.** Finish and verify a phase (run it, check the logs/metrics prove the claim) before starting the next. Don't let scope creep across phase boundaries.  
4. **Everything is core — the plan is to ship all of it.** All ten items below are in scope, including the two formerly-stretch phases (3 streaming, 5 priority tiers). Nothing is planned for cutting. Build in the tracker's listed order; the observability/container/deploy work (4.1–4.3) deliberately comes before Phase 6 so the GPU session produces dashboard graphs and deploys a known image rather than a hand-configured box. *Last-resort fallback only, if the timeline genuinely collapses:* drop Phase 5 first, Phase 3 second — never any of 1, 2, 4, 4.1, 4.2, 4.3, 6, 7. Those three additions earn more externally-visible signal per day invested (a live service, real dashboards) than either stretch phase would.  
5. **Commit after each phase**, not mid-phase, with a message referencing the phase and what was verified.

## Phase status tracker

Update this section as work progresses. Mark each phase `todo` / `in progress` / `done`, and note the key verified metric once done.

Listed in **execution order**, which is no longer the same as numeric order — Phases 4.1–4.3 were inserted after Phase 4, and Phases 3 and 5 now follow the deploy so their features ship as a new container image and land as new Grafana panels. Both still precede Phase 6, so time-to-first-token and priority behaviour are measurable in the GPU session.

- [x] Phase 1 — `done`. Baseline single-request FastAPI endpoint (no batching). **Verified:** throughput flat at **1.80 rps** across concurrency 1→16 (mock backend, 64 tok/req, ~555ms service time); p50 latency grows linearly 559ms → 8878ms; inference time constant at ~555ms while queue wait absorbs all growth (6658ms mean wait at conc 16). Data: `bench/results/phase1_baseline*.csv`.  
- [x] Phase 2 — `done`. Request queue + dynamic batching scheduler + per-request JSONL logging, on the CPU mock. **Verified:** **5.25x peak throughput** (1.80 → 9.44 rps at concurrency 16) and **p50 latency 8892ms → 1680ms**. Gains by concurrency: 2→1.82x, 4→3.18x, 8→5.13x, 16→5.25x. Honest cost: at concurrency 1 batching is 0.98x throughput / +10ms latency (the `MAX_WAIT_MS` deadline). Baseline re-measured on the same code via `MAX_BATCH_SIZE=1` as a control, reproducing Phase 1 within noise. Batch cost model: `MOCK_BATCH_ALPHA=0.08` (bandwidth-bound decode) — recalibrate in Phase 6. Data: `bench/results/phase2_*`, `logs/phase2_batching.jsonl`.  
- [x] Phase 4 — `done`. Backpressure / admission control: reject with **503 + `Retry-After`** once `MAX_QUEUE_DEPTH` (default 16) requests are queued; `MAX_QUEUE_DEPTH=0` disables it as the control condition. Threshold derived from a latency budget, not guessed: depth × (batch_time / max_batch_size) = 16 × 105ms ≈ 1.66s of queue wait, +840ms own pass ≈ 2.5s worst case. **Verified** under sustained open-loop overload (400 requests at 20 rps against a ~9.4 rps system, `bench/burst.py`): unprotected, p50 latency climbs linearly **3.1s → 20.8s** across the run with p99 **22,831ms** and queue depth reaching **214**; protected, p50 is **flat at 2.18s → 2.33s**, p99 **2,535ms**, depth capped at 16, and 199 excess requests rejected in **3.9ms** each. Predicted worst-case queue wait 1,660ms vs measured max 1,693ms — derivation within 2%. **Goodput 9.37 rps unprotected vs 9.27 protected — statistically identical**, so the protection costs ~1% of useful work and buys a latency bound. Data: `bench/results/phase4_*`, `logs/phase4_*.jsonl`.  
- [x] Phase 4.1 — `done`. Prometheus instrumentation (`app/telemetry.py`, 9 metrics at `/metrics`) + auto-provisioned Grafana dashboard (`ops/grafana/dashboards/llm-serving.json`, 10 panels). Runs on Colima/Docker; 1s scrape interval. **Verified** by `ops/verify_dashboards.py`, which extracts the PromQL from the dashboard JSON and runs it against Prometheus — so what passes is what the panels render. All 10 panels return data, and both Phase 4 conditions reproduce live from scrapes: protected → queue depth capped at **16**, e2e p50 **2.30s**, 202 served / 199 rejected; unprotected → depth **207 live / 215 peak**, p50 **16.81s**, p95 **28.12s**. Phase 2 result also visible in the same dashboard (queue wait p50 1.45s vs inference p50 0.824s). **Defect found and fixed during verification:** dashboard p99 read 2.93s vs a measured 2.53s because histogram `le` edges jumped 2.5→3.0, straddling the exact SLO the admission threshold is derived from; added edges at 2.25/2.75, error 16% → 7%. Data: `logs/phase41_*.jsonl`.  
- [x] Phase 4.2 — `done`. Dockerized: one `Dockerfile` (python:3.12-slim, non-root uid 10001, deps-before-source layer caching, interpreter-based healthcheck, **270MB**) and one `docker-compose.yml` covering app + Prometheus + Grafana, with every setting overridable from the shell (`MAX_QUEUE_DEPTH=0 docker compose up -d app` runs the control condition). **Verified:** full cold start — `down -v` → `build --no-cache` → `up -d` — brings all three services healthy in **6.2s**, Grafana dashboard provisioned onto a wiped volume, Prometheus scraping `app:8000`, app writing its log as `appuser`. Phase 4 results reproduce in-container (204/196 accepted/rejected, p50 **2,290ms**, goodput **9.21 rps**, depth capped 16; unprotected p50 **11,788ms**, peak depth **210**) — containerization cost nothing measurable. Grafana opens directly on the dashboard via `GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH`, since its default home lists only already-visited dashboards.  
- [ ] Phase 4.3 — Deploy the mock service (CPU-only, no GPU) to a PaaS — Fly.io, Railway, or Render. Est. 0.5 day. Metric: a public URL serving `/generate` and `/healthz`, linked from the README.  
- [ ] Phase 3 — Token streaming via SSE. Ships as a new image to the running deploy; adds a time-to-first-token panel to Grafana.  
- [ ] Phase 5 — Priority tiers (premium vs. free scheduling). Ships the same way; adds per-tier latency panels.  
- [ ] Phase 6 — Swap in real model on GPU, rerun load tests at scale, capture final p50/p95/p99 latency and max sustained throughput. Deploy the Phase 4.2 image to the rented box rather than hand-configuring it; capture results as Grafana graphs. Also recalibrate `MOCK_BATCH_ALPHA` against the measured real-model batch curve.  
- [ ] Phase 7 — README with architecture, design tradeoffs, benchmark graphs, **a live demo link, and Grafana dashboard screenshots** — not just CSV-derived charts.

## Conventions

- Keep the scope to a single node / single model instance. Multi-replica autoscaling is explicitly out of scope — note it as future work in the README rather than building it. Deploying (Phase 4.3) does not change this: the PaaS instance runs **one replica**, and the demo is there to make the system visible, not to scale it.  
- Prefer simple, explicit code over cleverness — this project is meant to be fully understood and defensible in an interview, not just functional.

