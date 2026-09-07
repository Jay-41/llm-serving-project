"""Request queue + dynamic batching scheduler.

The Phase 1 endpoint ran inference inline, one request at a time, so
concurrent callers simply queued behind each other in a thread pool and
throughput flatlined. Here the handler instead parks each request on an
asyncio.Queue and awaits a Future; a single background task drains that queue,
groups whatever it finds into a batch, and runs one forward pass for the whole
group.

The batching rule -- dispatch when EITHER the batch is full OR a deadline
expires -- is the whole tradeoff in two lines:

  * max_batch_size caps how much work one forward pass absorbs.
  * max_wait_ms caps how much latency an early arrival pays to wait for
    company. Under load the queue is never empty, batches fill instantly, and
    the deadline never matters. Under light load nothing is waiting, so the
    deadline fires and a lone request pays at most max_wait_ms.

That asymmetry is why the deadline is cheap: it only costs latency exactly
when the system has spare capacity to give.
"""

import asyncio
import itertools
import time
from typing import Any, Dict, List, Optional

from app import telemetry
from app.backends import Backend
from app.config import Settings
from app.metrics import MetricsLogger

# How long to sleep before re-checking an empty queue while a batch is still
# filling. See _collect_batch for why this polls rather than using wait_for.
_POLL_INTERVAL_S = 0.001


class QueueFull(Exception):
    """Raised by submit() when admission control refuses a request.

    Carries the numbers that justified the refusal so the handler can report
    them and the metrics log can record them. A rejection that cannot explain
    itself is indistinguishable from a bug.
    """

    def __init__(self, depth: int, limit: int) -> None:
        super().__init__(f"queue depth {depth} at limit {limit}")
        self.depth = depth
        self.limit = limit


class Job:
    """One in-flight request, from enqueue until its Future is resolved."""

    __slots__ = (
        "request_id",
        "prompt",
        "max_tokens",
        "future",
        "enqueued_at",
        "enqueued_wall",
        "queue_depth_at_enqueue",
    )

    def __init__(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int,
        future: "asyncio.Future",
        enqueued_at: float,
        enqueued_wall: float,
        queue_depth_at_enqueue: int,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.future = future
        self.enqueued_at = enqueued_at
        self.enqueued_wall = enqueued_wall
        self.queue_depth_at_enqueue = queue_depth_at_enqueue


class BatchingScheduler:
    def __init__(
        self,
        backend: Backend,
        settings: Settings,
        executor: Any,
        metrics: MetricsLogger,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._executor = executor
        self._metrics = metrics

        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._ids = itertools.count(1)
        self._task: Optional["asyncio.Task"] = None

        # Cumulative counters, surfaced on /healthz for a quick sanity check
        # without having to parse the JSONL.
        self.batches_dispatched = 0
        self.requests_served = 0
        self.requests_rejected = 0
        self.peak_queue_depth = 0

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def submit(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Enqueue a request and wait for its batch to complete.

        Raises QueueFull if admission control refuses it. The check happens
        BEFORE the job is created and enqueued, so a refused request costs the
        server nothing but the comparison below — that speed is the point. A
        rejection that takes as long as a real request is not backpressure.
        """
        depth = self._queue.qsize()
        limit = self._settings.max_queue_depth
        if limit > 0 and depth >= limit:
            self.requests_rejected += 1
            telemetry.REQUESTS.labels(outcome="rejected").inc()
            self._metrics.log(
                {
                    "event": "rejected",
                    "enqueued_at": time.time(),
                    "queue_depth_at_enqueue": depth,
                    "queue_depth_limit": limit,
                    "backend": self._backend.name,
                }
            )
            raise QueueFull(depth=depth, limit=limit)

        if depth + 1 > self.peak_queue_depth:
            self.peak_queue_depth = depth + 1
            telemetry.QUEUE_DEPTH_PEAK.set(self.peak_queue_depth)

        loop = asyncio.get_running_loop()
        job = Job(
            request_id=next(self._ids),
            prompt=prompt,
            max_tokens=max_tokens,
            future=loop.create_future(),
            enqueued_at=time.perf_counter(),
            enqueued_wall=time.time(),
            # Depth *before* this job joins, i.e. how many were already ahead.
            # Reuse the value the admission check read, so the number that
            # justified accepting the request is the number that gets logged.
            queue_depth_at_enqueue=depth,
        )
        self._queue.put_nowait(job)
        return await job.future

    # -- scheduler loop ----------------------------------------------------

    async def _run(self) -> None:
        while True:
            # Block until there is something to do. No busy-waiting while idle.
            first = await self._queue.get()
            batch = await self._collect_batch(first)
            await self._run_batch(batch)

    async def _collect_batch(self, first: Job) -> List[Job]:
        """Grow the batch until it is full or the wait deadline expires."""
        batch = [first]
        if self._settings.max_batch_size <= 1:
            return batch

        deadline = first.enqueued_at + self._settings.max_wait_ms / 1000.0
        while len(batch) < self._settings.max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
                continue  # fast path: queue is backed up, keep filling
            except asyncio.QueueEmpty:
                pass

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            # Poll rather than `asyncio.wait_for(queue.get(), remaining)`:
            # cancelling an in-flight get() has historically been able to drop
            # a job (bpo-37658), and a dropped job means a client hangs
            # forever. Polling is a millisecond of latency in the idle case
            # and provably cannot lose work. Under load the fast path above
            # means we never reach here at all.
            await asyncio.sleep(min(_POLL_INTERVAL_S, remaining))

        return batch

    async def _run_batch(self, batch: List[Job]) -> None:
        dequeued_at = time.perf_counter()
        dequeued_wall = time.time()

        # A batch runs for a single max_tokens. Real batched generation steps
        # every sequence forward together and stops when the longest one is
        # done, so the batch costs whatever its most demanding member costs.
        max_tokens = max(job.max_tokens for job in batch)
        prompts = [job.prompt for job in batch]

        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            texts = await loop.run_in_executor(
                self._executor, self._backend.generate_batch, prompts, max_tokens
            )
            error: Optional[BaseException] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to every waiter
            texts = []
            error = exc
        inference_ms = (time.perf_counter() - started) * 1000.0

        self.batches_dispatched += 1
        telemetry.BATCHES.inc()
        telemetry.BATCH_SIZE.observe(len(batch))
        telemetry.INFERENCE.observe(inference_ms / 1000.0)
        if error is None:
            self.requests_served += len(batch)
            telemetry.REQUESTS.labels(outcome="served").inc(len(batch))

        # Resolve the waiting clients FIRST, then log. Logging is not part of
        # the latency any caller experiences.
        finished_at = time.perf_counter()
        for index, job in enumerate(batch):
            if job.future.done():
                continue  # client disconnected and cancelled its wait
            if error is not None:
                job.future.set_exception(error)
            else:
                job.future.set_result(
                    {
                        "text": texts[index],
                        "request_id": job.request_id,
                        "batch_size": len(batch),
                        "max_tokens": max_tokens,
                        "queue_depth_at_enqueue": job.queue_depth_at_enqueue,
                        "queue_wait_ms": (dequeued_at - job.enqueued_at) * 1000.0,
                        "inference_ms": inference_ms,
                        "e2e_ms": (finished_at - job.enqueued_at) * 1000.0,
                    }
                )

        for job in batch:
            telemetry.QUEUE_WAIT.observe(dequeued_at - job.enqueued_at)
            telemetry.E2E.observe(finished_at - job.enqueued_at)
            self._metrics.log(
                {
                    "event": "served",
                    "request_id": job.request_id,
                    "enqueued_at": job.enqueued_wall,
                    "dequeued_at": dequeued_wall,
                    "queue_wait_ms": (dequeued_at - job.enqueued_at) * 1000.0,
                    "batch_size": len(batch),
                    "max_tokens": max_tokens,
                    "inference_ms": inference_ms,
                    "e2e_ms": (finished_at - job.enqueued_at) * 1000.0,
                    "queue_depth_at_enqueue": job.queue_depth_at_enqueue,
                    "backend": self._backend.name,
                    "error": None if error is None else type(error).__name__,
                }
            )
