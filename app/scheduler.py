"""Request queue + dynamic batching scheduler + streaming delivery.

The Phase 1 endpoint ran inference inline, one request at a time, so
concurrent callers simply queued behind each other in a thread pool and
throughput flatlined. Here the handler instead parks each request on an
asyncio.Queue; a single background task drains that queue, groups whatever it
finds into a batch, and drives one generation for the whole group.

The batching rule -- dispatch when EITHER the batch is full OR a deadline
expires -- is the whole tradeoff in two lines:

  * max_batch_size caps how much work one forward pass absorbs.
  * max_wait_ms caps how much latency an early arrival pays to wait for
    company. Under load the queue is never empty, batches fill instantly, and
    the deadline never matters. Under light load nothing is waiting, so the
    deadline fires and a lone request pays at most max_wait_ms.

That asymmetry is why the deadline is cheap: it only costs latency exactly
when the system has spare capacity to give.

Streaming (Phase 3): the backend yields one decode step at a time, and each
step carries one token for every sequence in the batch. The scheduler routes
each token as it arrives -- straight to a streaming job's queue, or into a
buffer for a job that wants the whole response at once. Both kinds can share a
batch. What streaming changes is when bytes reach the client, not how much
work the model does; throughput is identical either way.

Priority tiers (Phase 5): the queue is no longer FIFO. Paid requests are
served before free ones, and free requests are refused at a shallower queue
depth, so under overload the system sheds the load that matters least rather
than whoever happened to arrive last. Strict priority has a textbook failure
mode -- starvation, where a free request already in the queue never reaches
the front because paid arrivals keep jumping ahead -- so a free request that
has waited longer than aging_ms is promoted to paid priority. Priority buys a
better seat in line; it does not buy a faster oven. A paid request that lands
in a batch alongside seven free ones still waits for that batch's full pass.

What it does NOT do is continuous batching. A batch is static: once it starts,
nobody joins until it finishes, so a request arriving mid-batch waits for
someone else's remaining steps before its own prefill. That wait shows up
directly in time-to-first-token under load, and it is exactly what systems
like vLLM eliminate. Out of scope here; measured rather than hidden.
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


class StreamDone:
    """Terminal item on a streaming job's queue. Carries the same result dict
    a non-streaming response would have returned, so the client gets the full
    timing breakdown at the end of the stream."""

    __slots__ = ("result",)

    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result


class TieredQueue:
    """Priority queue with aging, for a single consumer.

    A plain list scanned on every take, not a heap. The queue is bounded by
    admission control at max_queue_depth (16 by default), so the scan is
    trivially cheap -- and a heap could not do aging anyway: a heap orders by a
    key fixed at insertion, while aging means a job's priority changes just by
    sitting there. Computing effective priority at take time is both simpler
    and the only correct option.

    Effective priority is (rank, enqueued_at): paid is rank 0, free is rank 1,
    and a free job that has waited at least aging_s is treated as rank 0.
    Ties break by arrival, so within a tier it is still FIFO.
    """

    def __init__(self, aging_s: float) -> None:
        self._items: List["Job"] = []
        self._aging_s = aging_s
        # Set exactly when _items is non-empty; get() blocks on it. Only one
        # consumer (the scheduler task), so no lost-wakeup races to reason
        # about.
        self._not_empty = asyncio.Event()

    def qsize(self) -> int:
        return len(self._items)

    def count(self, tier: str) -> int:
        return sum(1 for j in self._items if j.tier == tier)

    def _rank(self, job: "Job", now: float) -> int:
        if job.tier == "paid":
            return 0
        if self._aging_s > 0 and now - job.enqueued_at >= self._aging_s:
            job.aged = True
            return 0
        return 1

    def put_nowait(self, job: "Job") -> None:
        self._items.append(job)
        self._not_empty.set()

    def get_nowait(self) -> "Job":
        if not self._items:
            raise asyncio.QueueEmpty
        now = time.perf_counter()
        best = min(
            range(len(self._items)),
            key=lambda i: (self._rank(self._items[i], now), self._items[i].enqueued_at),
        )
        job = self._items.pop(best)
        if not self._items:
            self._not_empty.clear()
        return job

    async def get(self) -> "Job":
        while not self._items:
            await self._not_empty.wait()
        return self.get_nowait()


class Job:
    """One in-flight request, from enqueue until its result is delivered."""

    __slots__ = (
        "request_id",
        "prompt",
        "max_tokens",
        "tier",
        "aged",
        "stream",
        "future",
        "token_queue",
        "enqueued_at",
        "enqueued_wall",
        "queue_depth_at_enqueue",
        "first_token_at",
        "completed_at",
        "text_parts",
        "tokens",
        "done",
    )

    def __init__(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int,
        tier: str,
        stream: bool,
        loop: "asyncio.AbstractEventLoop",
        enqueued_at: float,
        enqueued_wall: float,
        queue_depth_at_enqueue: int,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.tier = tier
        self.aged = False  # set by TieredQueue if aging promoted this job
        self.stream = stream
        # Exactly one delivery channel per job. A Future for "give me the whole
        # thing"; a queue of tokens for "give me each one as it exists".
        self.future = None if stream else loop.create_future()
        self.token_queue: Optional["asyncio.Queue"] = (
            asyncio.Queue() if stream else None
        )
        self.enqueued_at = enqueued_at
        self.enqueued_wall = enqueued_wall
        self.queue_depth_at_enqueue = queue_depth_at_enqueue
        self.first_token_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.text_parts: List[str] = []
        self.tokens = 0
        self.done = False


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

        self._queue = TieredQueue(aging_s=settings.aging_ms / 1000.0)
        self._ids = itertools.count(1)
        self._task: Optional["asyncio.Task"] = None

        # Cumulative counters, surfaced on /healthz for a quick sanity check
        # without having to parse the JSONL.
        self.batches_dispatched = 0
        self.requests_served = 0
        self.requests_rejected = 0
        self.peak_queue_depth = 0
        self.aged_promotions = 0

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

    def queue_depth_for(self, tier: str) -> int:
        return self._queue.count(tier)

    async def submit(self, prompt: str, max_tokens: int, tier: str) -> Dict[str, Any]:
        """Enqueue a request and wait for the complete response."""
        job = self._enqueue(prompt, max_tokens, tier, stream=False)
        assert job.future is not None
        return await job.future

    def submit_stream(self, prompt: str, max_tokens: int, tier: str) -> Job:
        """Enqueue a request for streaming delivery. Returns the Job; the
        caller drains job.token_queue until it yields a StreamDone."""
        return self._enqueue(prompt, max_tokens, tier, stream=True)

    # -- admission ---------------------------------------------------------

    def admission_limit(self, tier: str) -> int:
        """Queue depth at which a request of this tier is refused. 0 = never.

        Free uses the lower of the two limits so that setting
        MAX_QUEUE_DEPTH_FREE above MAX_QUEUE_DEPTH cannot accidentally let
        free requests in where paid ones would be refused.
        """
        paid = self._settings.max_queue_depth
        if tier == "paid":
            return paid
        free = self._settings.max_queue_depth_free
        if free <= 0:
            return paid
        if paid <= 0:
            return free
        return min(free, paid)

    def _enqueue(self, prompt: str, max_tokens: int, tier: str, stream: bool) -> Job:
        """Admission control, then enqueue.

        Raises QueueFull if refused. The check happens BEFORE the job is
        created, so a refused request costs the server one integer comparison
        -- that speed is the point. A rejection that takes as long as a real
        request is not backpressure. For a streaming request this also means
        the refusal is a plain HTTP 503, never a half-open stream.

        The threshold depends on tier. Under overload the queue sits between
        the free and paid limits, so free arrivals bounce while paid ones
        still get in: the system sheds the load that matters least.
        """
        depth = self._queue.qsize()
        limit = self.admission_limit(tier)
        if limit > 0 and depth >= limit:
            self.requests_rejected += 1
            telemetry.REQUESTS.labels(outcome="rejected", tier=tier).inc()
            self._metrics.log(
                {
                    "event": "rejected",
                    "tier": tier,
                    "enqueued_at": time.time(),
                    "queue_depth_at_enqueue": depth,
                    "queue_depth_limit": limit,
                    "stream": stream,
                    "backend": self._backend.name,
                }
            )
            raise QueueFull(depth=depth, limit=limit)

        if depth + 1 > self.peak_queue_depth:
            self.peak_queue_depth = depth + 1
            telemetry.QUEUE_DEPTH_PEAK.set(self.peak_queue_depth)

        job = Job(
            request_id=next(self._ids),
            prompt=prompt,
            max_tokens=max_tokens,
            tier=tier,
            stream=stream,
            loop=asyncio.get_running_loop(),
            enqueued_at=time.perf_counter(),
            enqueued_wall=time.time(),
            # Depth *before* this job joins, i.e. how many were already ahead.
            # Reuse the value the admission check read, so the number that
            # justified accepting the request is the number that gets logged.
            queue_depth_at_enqueue=depth,
        )
        self._queue.put_nowait(job)
        return job

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
        # done, so the batch costs whatever its most demanding member costs. A
        # member that asked for fewer tokens is marked done early and stops
        # receiving; the batch keeps running for the others. That idle slot is
        # the waste continuous batching exists to reclaim.
        max_tokens = max(job.max_tokens for job in batch)
        prompts = [job.prompt for job in batch]

        loop = asyncio.get_running_loop()

        # The backend is a blocking generator on the inference thread. Each
        # step has to cross to the event loop, and it does so exactly once per
        # step -- not once per token per client -- via a thread-safe handoff
        # onto this queue. Items are (timestamp, payload) so TTFT is measured
        # from when the token existed, not from when the loop got around to it.
        steps: "asyncio.Queue" = asyncio.Queue()
        _END = object()

        def drive() -> None:
            try:
                for step in self._backend.generate_batch_stream(prompts, max_tokens):
                    loop.call_soon_threadsafe(
                        steps.put_nowait, (time.perf_counter(), step)
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced to every waiter
                loop.call_soon_threadsafe(
                    steps.put_nowait, (time.perf_counter(), exc)
                )
            finally:
                loop.call_soon_threadsafe(
                    steps.put_nowait, (time.perf_counter(), _END)
                )

        started = time.perf_counter()
        drive_future = loop.run_in_executor(self._executor, drive)

        error: Optional[BaseException] = None
        while True:
            ts, payload = await steps.get()
            if payload is _END:
                break
            if isinstance(payload, BaseException):
                error = payload
                break
            for index, job in enumerate(batch):
                if job.done:
                    continue
                token = payload[index]
                if token is None:
                    continue
                if job.first_token_at is None:
                    job.first_token_at = ts
                job.text_parts.append(token)
                job.tokens += 1
                if job.stream and job.token_queue is not None:
                    job.token_queue.put_nowait(token)
                if job.tokens >= job.max_tokens:
                    # This job's response is complete even though the batch
                    # may keep stepping for longer members. Deliver now; its
                    # e2e ends here, not when the batch does.
                    self._finish_job(job, batch, ts, dequeued_at, started)

        await drive_future
        inference_s = time.perf_counter() - started

        self.batches_dispatched += 1
        telemetry.BATCHES.inc()
        telemetry.BATCH_SIZE.observe(len(batch))
        telemetry.INFERENCE.observe(inference_s)

        finished_at = time.perf_counter()
        for job in batch:
            if job.done:
                continue
            if error is not None:
                self._fail_job(job, error)
            else:
                # Backend ended the sequence early (EOS) or the generator ran
                # out of steps before this job hit its own max_tokens.
                self._finish_job(job, batch, finished_at, dequeued_at, started)

        # Log AFTER every client has been delivered to. Logging is not part of
        # the latency any caller experiences.
        for job in batch:
            self._log_job(job, batch, dequeued_wall, dequeued_at, inference_s, error)

    # -- delivery ----------------------------------------------------------

    def _finish_job(
        self,
        job: Job,
        batch: List[Job],
        completed_at: float,
        dequeued_at: float,
        batch_started: float,
    ) -> None:
        job.done = True
        job.completed_at = completed_at
        self.requests_served += 1
        if job.aged:
            self.aged_promotions += 1
            telemetry.AGED.inc()
        telemetry.REQUESTS.labels(outcome="served", tier=job.tier).inc()
        telemetry.TOKENS.inc(job.tokens)
        if job.first_token_at is not None:
            telemetry.TTFT.observe(job.first_token_at - job.enqueued_at)
        telemetry.QUEUE_WAIT.labels(tier=job.tier).observe(dequeued_at - job.enqueued_at)
        telemetry.E2E.labels(tier=job.tier).observe(completed_at - job.enqueued_at)

        result = self._result_for(job, batch, dequeued_at, batch_started, completed_at)
        if job.stream and job.token_queue is not None:
            job.token_queue.put_nowait(StreamDone(result))
        elif job.future is not None and not job.future.done():
            job.future.set_result(result)

    def _fail_job(self, job: Job, error: BaseException) -> None:
        job.done = True
        job.completed_at = time.perf_counter()
        if job.stream and job.token_queue is not None:
            job.token_queue.put_nowait(error)
        elif job.future is not None and not job.future.done():
            job.future.set_exception(error)

    def _result_for(
        self,
        job: Job,
        batch: List[Job],
        dequeued_at: float,
        batch_started: float,
        completed_at: float,
    ) -> Dict[str, Any]:
        ttft_ms = (
            (job.first_token_at - job.enqueued_at) * 1000.0
            if job.first_token_at is not None
            else None
        )
        return {
            "text": "".join(job.text_parts),
            "request_id": job.request_id,
            "tier": job.tier,
            "aged": job.aged,
            "batch_size": len(batch),
            "max_tokens": job.max_tokens,
            "tokens": job.tokens,
            "queue_depth_at_enqueue": job.queue_depth_at_enqueue,
            "queue_wait_ms": (dequeued_at - job.enqueued_at) * 1000.0,
            "ttft_ms": ttft_ms,
            # Cost of the pass this job rode in, up to the point it finished.
            "inference_ms": (completed_at - batch_started) * 1000.0,
            "e2e_ms": (completed_at - job.enqueued_at) * 1000.0,
        }

    def _log_job(
        self,
        job: Job,
        batch: List[Job],
        dequeued_wall: float,
        dequeued_at: float,
        inference_s: float,
        error: Optional[BaseException],
    ) -> None:
        completed = job.completed_at if job.completed_at is not None else time.perf_counter()
        self._metrics.log(
            {
                "event": "served" if error is None else "failed",
                "request_id": job.request_id,
                "tier": job.tier,
                "aged": job.aged,
                "stream": job.stream,
                "enqueued_at": job.enqueued_wall,
                "dequeued_at": dequeued_wall,
                "queue_wait_ms": (dequeued_at - job.enqueued_at) * 1000.0,
                "ttft_ms": (
                    (job.first_token_at - job.enqueued_at) * 1000.0
                    if job.first_token_at is not None
                    else None
                ),
                "batch_size": len(batch),
                "max_tokens": job.max_tokens,
                "tokens": job.tokens,
                "inference_ms": inference_s * 1000.0,
                "e2e_ms": (completed - job.enqueued_at) * 1000.0,
                "queue_depth_at_enqueue": job.queue_depth_at_enqueue,
                "backend": self._backend.name,
                "error": None if error is None else type(error).__name__,
            }
        )
