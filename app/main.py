"""HTTP surface: queue + dynamic batching + admission control + SSE streaming.

The handler no longer runs inference. It hands the request to a queue and
awaits a Future; one background scheduler task drains that queue, groups
requests into batches, and runs a single forward pass per batch.

Inference still executes on a ThreadPoolExecutor with max_workers=1, exactly
as in Phase 1 -- one model instance, one forward pass at a time. Nothing about
the hardware constraint changed. The only thing that changed is how much
useful work each of those serialised passes carries.

Set MAX_BATCH_SIZE=1 to reproduce the Phase 1 baseline through this same code
path, which is how the before/after comparison is kept honest.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import telemetry
from app.backends import build_backend
from app.config import get_settings
from app.metrics import MetricsLogger
from app.scheduler import BatchingScheduler, Job, QueueFull, StreamDone
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = build_backend(settings)

    started = time.perf_counter()
    backend.load()
    load_ms = (time.perf_counter() - started) * 1000.0

    # Still one worker: a single model instance cannot run two forward passes
    # concurrently. Batching does not change that -- it makes each pass do
    # more work rather than making passes overlap.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    metrics = MetricsLogger(settings.metrics_path)
    scheduler = BatchingScheduler(backend, settings, executor, metrics)
    scheduler.start()

    # Publish the config the dashboards draw limit lines from, and wire queue
    # depth to a callback so each scrape reads the live value rather than
    # whatever it was when it last changed.
    telemetry.export_config(
        settings.max_batch_size, settings.max_queue_depth,
        scheduler.admission_limit("free"),
    )
    telemetry.QUEUE_DEPTH.set_function(lambda: scheduler.queue_depth)
    telemetry.QUEUE_DEPTH_BY_TIER.labels(tier="paid").set_function(
        lambda: scheduler.queue_depth_for("paid"))
    telemetry.QUEUE_DEPTH_BY_TIER.labels(tier="free").set_function(
        lambda: scheduler.queue_depth_for("free"))

    app.state.settings = settings
    app.state.backend = backend
    app.state.executor = executor
    app.state.metrics = metrics
    app.state.scheduler = scheduler

    print(
        f"[startup] backend={backend.name} loaded in {load_ms:.0f}ms | "
        f"max_batch_size={settings.max_batch_size} "
        f"max_wait_ms={settings.max_wait_ms} | metrics={metrics.path}"
    )
    try:
        yield
    finally:
        await scheduler.stop()
        executor.shutdown(wait=True)
        metrics.close()


app = FastAPI(
    title="LLM Serving Layer",
    description=(
        "Single-node LLM inference serving: request queueing, dynamic batching "
        "and admission control. This instance runs a **mock backend** that "
        "sleeps for a modelled amount of time instead of running a model, so "
        "it demonstrates the scheduling behaviour rather than text quality. "
        "Try `POST /generate` below, then watch `/metrics` change."
    ),
    version="0.4.3",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def index() -> dict:
    """Landing page for humans who click the link. Points at the real
    endpoints rather than greeting them with a 404."""
    settings = app.state.settings
    return {
        "service": "llm-serving-layer",
        "backend": app.state.backend.name,
        "what_this_is": (
            "A single-node LLM serving layer with request queueing, dynamic "
            "batching and admission control. This instance uses a mock model "
            "that sleeps instead of generating, so responses are placeholders "
            "-- the point is the scheduling behaviour, visible in the timing "
            "fields of every response."
        ),
        "try_it": "GET /docs for an interactive console",
        "endpoints": {
            "POST /generate": "submit a prompt; response carries queue_wait_ms, "
                              "batch_size, inference_ms and e2e_ms",
            "GET /healthz": "live scheduler counters",
            "GET /metrics": "Prometheus exposition",
            "GET /docs": "OpenAPI console",
        },
        "scheduler": {
            "max_batch_size": settings.max_batch_size,
            "max_wait_ms": settings.max_wait_ms,
            "max_queue_depth": settings.max_queue_depth,
            "max_queue_depth_free": app.state.scheduler.admission_limit("free"),
            "aging_ms": settings.aging_ms,
            "note": "requests beyond the tier's queue depth limit are refused "
                    "with 503 and a Retry-After header rather than queued; "
                    "paid is served before free, and a free request that has "
                    "waited aging_ms is promoted so it cannot starve",
        },
    }


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    settings = app.state.settings
    scheduler = app.state.scheduler
    dispatched = scheduler.batches_dispatched
    return HealthResponse(
        status="ok",
        backend=app.state.backend.name,
        default_max_tokens=settings.default_max_tokens,
        max_batch_size=settings.max_batch_size,
        max_wait_ms=settings.max_wait_ms,
        max_queue_depth=settings.max_queue_depth,
        max_queue_depth_free=scheduler.admission_limit("free"),
        aging_ms=settings.aging_ms,
        queue_depth=scheduler.queue_depth,
        queue_depth_paid=scheduler.queue_depth_for("paid"),
        queue_depth_free=scheduler.queue_depth_for("free"),
        peak_queue_depth=scheduler.peak_queue_depth,
        aged_promotions=scheduler.aged_promotions,
        batches_dispatched=dispatched,
        requests_served=scheduler.requests_served,
        requests_rejected=scheduler.requests_rejected,
        mean_batch_size=(
            scheduler.requests_served / dispatched if dispatched else 0.0
        ),
    )


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus scrape target. Plain text, not JSON — this is the exposition
    format Prometheus parses, so it must not go through a response model."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    settings = app.state.settings

    max_tokens = req.max_tokens or settings.default_max_tokens
    if max_tokens > settings.max_allowed_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"max_tokens must be <= {settings.max_allowed_tokens}",
        )

    try:
        if req.stream:
            job = app.state.scheduler.submit_stream(req.prompt, max_tokens, req.tier)
        else:
            result = await app.state.scheduler.submit(req.prompt, max_tokens, req.tier)
    except QueueFull as exc:
        # 503, not 429. The deciding question is whose fault the rejection is.
        # 429 Too Many Requests means "you, the client, sent too much" — the
        # code for a per-client rate limit. Admission control here is purely
        # global queue depth: a client's very first request is refused if it
        # arrives at a bad moment. That is server capacity, which is what 503
        # means. (If this ever sat behind a load balancer that ejects backends
        # on 503, switch to 429 — TGI does exactly that, for that reason.)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "server at capacity",
                "tier": req.tier,
                "queue_depth": exc.depth,
                "queue_depth_limit": exc.limit,
                "retry_after_s": settings.retry_after_s,
            },
            headers={"Retry-After": str(settings.retry_after_s)},
        )

    if req.stream:
        # Admission already happened above, so an overloaded server answers
        # with a plain 503 -- never a 200 that opens a stream and then dies.
        return StreamingResponse(
            _sse_events(job, app.state.backend.name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Tell nginx-style proxies not to buffer, or "streaming" turns
                # into "everything arrives at once at the end" behind them.
                "X-Accel-Buffering": "no",
            },
        )
    return GenerateResponse(backend=app.state.backend.name, **result)


def _sse_event(event: str, payload: dict) -> str:
    """One Server-Sent Event: an `event:` line naming it, a `data:` line with
    JSON, and the blank line that terminates it."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _sse_events(job: Job, backend_name: str) -> AsyncIterator[str]:
    """Drain a streaming job's token queue into SSE frames.

    SSE rather than WebSockets because the traffic is one-directional, it is
    plain HTTP (works with curl, survives every proxy, needs no upgrade
    handshake), and browsers have EventSource built in. WebSockets would buy
    nothing here and cost a protocol.

    If the client disconnects, this generator is cancelled. The job keeps
    running inside its batch -- a static batch cannot evict a member -- and
    its remaining tokens land on a queue nobody reads, bounded by max_tokens
    and collected with the Job. Reclaiming that slot mid-batch is continuous
    batching, which is out of scope.
    """
    assert job.token_queue is not None
    while True:
        item = await job.token_queue.get()
        if isinstance(item, StreamDone):
            yield _sse_event("done", {"backend": backend_name, **item.result})
            return
        if isinstance(item, BaseException):
            yield _sse_event("error", {"error": type(item).__name__})
            return
        yield _sse_event("token", {"token": item})
