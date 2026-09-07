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
    # not free. alpha=0.08 puts a batch of 8 at ~1.56x the cost of a batch of
    # 1, i.e. ~5.1x the throughput. Recalibrate against measured Qwen numbers
    # in Phase 6.
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
    # Re-derive it in Phase 6: batch_time changes on real hardware.
    max_queue_depth: int

    # Advertised on rejections via the Retry-After header. Roughly the time to
    # drain a full queue, so a client that honours it comes back to a system
    # that has actually made progress.
    retry_after_s: int

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
        mock_base_ms=_env_float("MOCK_BASE_MS", 40.0),
        mock_per_token_ms=_env_float("MOCK_PER_TOKEN_MS", 8.0),
        mock_jitter_ms=_env_float("MOCK_JITTER_MS", 5.0),
        mock_batch_alpha=_env_float("MOCK_BATCH_ALPHA", 0.08),
        max_batch_size=_env_int("MAX_BATCH_SIZE", 8),
        max_wait_ms=_env_float("MAX_WAIT_MS", 10.0),
        max_queue_depth=_env_int("MAX_QUEUE_DEPTH", 16),
        retry_after_s=_env_int("RETRY_AFTER_S", 2),
        metrics_path=_env_str("METRICS_PATH", "logs/requests.jsonl"),
        model_name=_env_str("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        model_device=_env_str("MODEL_DEVICE", "auto"),
        model_dtype=_env_str("MODEL_DTYPE", "auto"),
        default_max_tokens=_env_int("DEFAULT_MAX_TOKENS", 64),
        max_allowed_tokens=_env_int("MAX_ALLOWED_TOKENS", 512),
    )
