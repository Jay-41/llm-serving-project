"""Phase 1: baseline single-request serving. No queue, no batching.

Every request runs inference directly, and inference is dispatched to a thread
pool with exactly ONE worker. That single worker is the whole point of this
phase: one model instance cannot serve two requests at once, so under
concurrency requests pile up waiting their turn, latency grows linearly with
concurrency, and throughput flatlines at 1 / service_time.

Those are the "before" numbers Phase 2 has to beat.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.backends import build_backend
from app.config import get_settings
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = build_backend(settings)

    started = time.perf_counter()
    backend.load()
    load_ms = (time.perf_counter() - started) * 1000.0

    # max_workers=1 is deliberate and load-bearing. Raising it would not make a
    # single model instance parallel -- it would just oversubscribe it and hide
    # the queueing behind thread scheduling. Phase 2 replaces this implicit
    # one-at-a-time bottleneck with an explicit asyncio.Queue plus a batching
    # scheduler, which is where the throughput win comes from.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    app.state.settings = settings
    app.state.backend = backend
    app.state.executor = executor

    print(
        f"[startup] backend={backend.name} loaded in {load_ms:.0f}ms "
        f"| default_max_tokens={settings.default_max_tokens}"
    )
    try:
        yield
    finally:
        executor.shutdown(wait=True)


app = FastAPI(title="LLM Serving Layer - Phase 1 baseline", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    settings = app.state.settings
    return HealthResponse(
        status="ok",
        backend=app.state.backend.name,
        default_max_tokens=settings.default_max_tokens,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    settings = app.state.settings
    backend = app.state.backend

    max_tokens = req.max_tokens or settings.default_max_tokens
    if max_tokens > settings.max_allowed_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"max_tokens must be <= {settings.max_allowed_tokens}",
        )

    received = time.perf_counter()
    marks = {}

    def run_inference() -> str:
        # Runs on the single inference thread. The gap between `received` and
        # `marks["started"]` is time spent waiting for that thread to be free,
        # i.e. the queueing this phase does nothing about.
        marks["started"] = time.perf_counter()
        try:
            return backend.generate(req.prompt, max_tokens)
        finally:
            marks["finished"] = time.perf_counter()

    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(app.state.executor, run_inference)

    return GenerateResponse(
        text=text,
        backend=backend.name,
        max_tokens=max_tokens,
        wait_ms=(marks["started"] - received) * 1000.0,
        inference_ms=(marks["finished"] - marks["started"]) * 1000.0,
        total_ms=(time.perf_counter() - received) * 1000.0,
    )
