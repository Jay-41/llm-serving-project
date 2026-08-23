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

from fastapi import FastAPI, HTTPException

from app.backends import build_backend
from app.config import get_settings
from app.metrics import MetricsLogger
from app.scheduler import BatchingScheduler
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
        queue_depth=scheduler.queue_depth,
        batches_dispatched=dispatched,
        requests_served=scheduler.requests_served,
        mean_batch_size=(
            scheduler.requests_served / dispatched if dispatched else 0.0
        ),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    settings = app.state.settings

    max_tokens = req.max_tokens or settings.default_max_tokens
    if max_tokens > settings.max_allowed_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"max_tokens must be <= {settings.max_allowed_tokens}",
        )

    result = await app.state.scheduler.submit(req.prompt, max_tokens)
    return GenerateResponse(backend=app.state.backend.name, **result)
