"""Environment-driven settings.

Everything is read from the environment once, at startup, into a frozen
dataclass. No config framework — the surface is small enough that explicit
parsing is easier to read and to defend than a dependency.
"""

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    # Which inference backend to serve with: "mock" or "qwen".
    # Mock is the default for Phases 1-5 (see spec: "Mock First, GPU Last").
    backend: str

    # --- Mock backend timing ---------------------------------------------
    # Real decoder latency is roughly a fixed prefill cost plus a per-token
    # decode cost, so the mock models it the same way instead of using one
    # flat sleep. This matters: it is what makes the Phase 2 batching delta
    # resemble the delta a real model would show.
    mock_base_ms: float
    mock_per_token_ms: float
    mock_jitter_ms: float

    # How much a batch of N costs relative to a batch of 1:
    #     1 + mock_batch_alpha * (N - 1)
    # Decode on a GPU is memory-bandwidth bound at small batch sizes -- the
    # weights are read from HBM once per decode step regardless of how many
    # sequences share the step -- so batching is much cheaper than linear but
    # not free.
    #
    # CALIBRATED IN PHASE 6. The defaults for base_ms, per_token_ms and alpha
    # are now the values measured on an NVIDIA L4 with Qwen2.5-1.5B-Instruct
    # (bench/results/gpu_probe.txt): prefill 19.6ms, 15.96ms/token, and
    # alpha = 0.0073 -- a batch of 16 costs 14% more per step than a batch of
    # 1. Phases 1-5 were measured with the pre-calibration guesses (40 / 8 /
    # 0.08, MAX_QUEUE_DEPTH 16); those results stand as recorded in the README
    # and are reproducible by setting the old values explicitly. The guess for
    # alpha was ten times too pessimistic.
    mock_batch_alpha: float

    # --- Scheduler --------------------------------------------------------
    # A batch is dispatched as soon as EITHER condition trips: it reaches
    # max_batch_size, or max_wait_ms has passed since the first job landed.
    # max_wait_ms is the latency the system is willing to spend buying
    # throughput; set it to 0 and batching only ever catches what is already
    # queued. Set max_batch_size=1 to reproduce the Phase 1 baseline exactly
    # through this same code path.
    max_batch_size: int
    max_wait_ms: float

    # --- Admission control ------------------------------------------------
    # Reject a new request outright once this many are already waiting.
    # Set to 0 to disable entirely (unbounded queue) — that is the control
    # condition for the Phase 4 comparison, the same trick MAX_BATCH_SIZE=1
    # plays for Phase 2.
    #
    # Picking the number: queue depth is meaningless on its own; what matters
    # is how long that depth takes to DRAIN. Depth converts to promised
    # latency at a fixed rate:
    #
    #     wait_per_queued_request = batch_time / max_batch_size
    #                             = 840ms / 8  =  ~105ms
    #
    # So the threshold is a latency budget in disguise. Default 16 = two full
    # batches of backlog = ~1.7s of queueing, plus the request's own ~840ms
    # pass, for a worst-case accepted latency around 2.5s.
    #
    # Bigger is not safer. A deep queue is a latency bomb: work accepted but
    # not drained in time means the client times out anyway and the GPU burns
    # a pass on a response nobody is waiting for. Smaller is not safer either
    # — arrivals are jittery, and a shallow queue turns every momentary clump
    # into a rejection while leaving capacity idle. This knob is the dial
    # between those two failure modes.
    #
    # Re-derived in Phase 6 against the L4: batch-8 pass measured at ~1090ms,
    # so (2500 - 1090) * 8 / 1090 = 10. Measured p99 under 2x overload with
    # depth 10 came in at 2.98s against the 2.5s target -- 19% over. The
    # derivation is an approximation that ignores the batch already in
    # flight; real hardware also varies more than a deterministic sleep.
    max_queue_depth: int

    # Advertised on rejections via the Retry-After header. Roughly the time to
    # drain a full queue, so a client that honours it comes back to a system
    # that has actually made progress.
    retry_after_s: int

    # --- Priority tiers ---------------------------------------------------
    # Admission threshold for free-tier requests. Paid uses max_queue_depth.
    # Setting this lower than max_queue_depth is what makes backpressure
    # business-aware: under overload the queue sits between the two limits,
    # so free arrivals are refused while paid ones still get in. Set it equal
    # to max_queue_depth (or 0, unbounded) to disable tiered admission.
    max_queue_depth_free: int

    # Priority ordering has a classic failure mode: if paid traffic never
    # stops, a free request already in the queue never reaches the front --
    # every new paid arrival jumps ahead of it. That is starvation. Aging is
    # the standard fix: after waiting this long, a free request is treated as
    # paid priority. Bounds free-tier wait at roughly aging_ms + one batch.
    # 0 disables aging (strict priority), which is the control condition for
    # the starvation experiment.
    aging_ms: float

    # --- Metrics ----------------------------------------------------------
    metrics_path: str

    # --- Real model backend (not exercised until Phase 6) ----------------
    model_name: str
    model_device: str
    model_dtype: str

    # --- Request defaults -------------------------------------------------
    default_max_tokens: int
    max_allowed_tokens: int


def get_settings() -> Settings:
    return Settings(
        backend=_env_str("BACKEND", "mock").strip().lower(),
        mock_base_ms=_env_float("MOCK_BASE_MS", 20.0),
        mock_per_token_ms=_env_float("MOCK_PER_TOKEN_MS", 16.0),
        mock_jitter_ms=_env_float("MOCK_JITTER_MS", 5.0),
        mock_batch_alpha=_env_float("MOCK_BATCH_ALPHA", 0.007),
        max_batch_size=_env_int("MAX_BATCH_SIZE", 8),
        max_wait_ms=_env_float("MAX_WAIT_MS", 10.0),
        max_queue_depth=_env_int("MAX_QUEUE_DEPTH", 10),
        retry_after_s=_env_int("RETRY_AFTER_S", 2),
        max_queue_depth_free=_env_int("MAX_QUEUE_DEPTH_FREE", 5),
        aging_ms=_env_float("AGING_MS", 2000.0),
        metrics_path=_env_str("METRICS_PATH", "logs/requests.jsonl"),
        model_name=_env_str("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        model_device=_env_str("MODEL_DEVICE", "auto"),
        model_dtype=_env_str("MODEL_DTYPE", "auto"),
        default_max_tokens=_env_int("DEFAULT_MAX_TOKENS", 64),
        max_allowed_tokens=_env_int("MAX_ALLOWED_TOKENS", 512),
    )
