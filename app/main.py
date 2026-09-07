"""Phase 2: request queue + dynamic batching scheduler.

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

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import telemetry
from app.backends import build_backend
from app.config import get_settings
from app.metrics import MetricsLogger
from app.scheduler import BatchingScheduler, QueueFull
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
    telemetry.export_config(settings.max_batch_size, settings.max_queue_depth)
    telemetry.QUEUE_DEPTH.set_function(lambda: scheduler.queue_depth)

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


app = FastAPI(title="LLM Serving Layer - Phase 2 batching", lifespan=lifespan)


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
        queue_depth=scheduler.queue_depth,
        peak_queue_depth=scheduler.peak_queue_depth,
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
        result = await app.state.scheduler.submit(req.prompt, max_tokens)
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
                "queue_depth": exc.depth,
                "queue_depth_limit": exc.limit,
                "retry_after_s": settings.retry_after_s,
            },
            headers={"Retry-After": str(settings.retry_after_s)},
        )
    return GenerateResponse(backend=app.state.backend.name, **result)
