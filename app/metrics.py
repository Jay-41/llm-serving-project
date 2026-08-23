"""Per-request structured logging, one JSON object per line.

The spec calls for this from Phase 2 rather than Phase 6, so the batching win
can be verified as it is built instead of reconstructed afterwards. Every
record carries the fields needed to prove (or disprove) that claim:

    enqueued_at / dequeued_at   -> when the job arrived and when a batch took it
    queue_wait_ms               -> the gap between those two
    batch_size                  -> how many requests shared the forward pass
    inference_ms                -> cost of that forward pass
    e2e_ms                      -> enqueue until the response was ready
    queue_depth_at_enqueue      -> how backed up things were on arrival

JSONL rather than CSV because records are small, append-only, and trivially
readable with pandas (`pd.read_json(path, lines=True)`) or jq.
"""

import json
import os
from typing import Any, Dict, Optional, TextIO


class MetricsLogger:
    """Append-only JSONL writer.

    Writes are synchronous and happen on the event loop, after the waiting
    client has already been handed its result -- so the cost lands outside the
    latency being measured. At this scale (a few hundred bytes per request)
    that is cheaper and far easier to reason about than a background writer
    task, and it cannot lose the tail of a run on shutdown.
    """

    def __init__(self, path: Optional[str]) -> None:
        self._path = path
        self._handle: Optional[TextIO] = None
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._handle = open(path, "a", buffering=1)  # line buffered

    @property
    def path(self) -> Optional[str]:
        return self._path

    def log(self, record: Dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
