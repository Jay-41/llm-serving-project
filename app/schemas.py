"""Request/response models for the serving API."""

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt to complete.")
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Tokens to generate. Falls back to DEFAULT_MAX_TOKENS.",
    )


class GenerateResponse(BaseModel):
    text: str
    backend: str
    request_id: int
    max_tokens: int

    # How many requests shared this request's forward pass. This is the number
    # that has to rise with load for batching to be doing anything.
    batch_size: int
    queue_depth_at_enqueue: int

    # Timings in milliseconds. The split matters: batching should shrink
    # queue_wait_ms dramatically while barely moving inference_ms.
    queue_wait_ms: float = Field(..., description="Enqueue until a batch took it.")
    inference_ms: float = Field(..., description="Cost of the batched forward pass.")
    e2e_ms: float = Field(..., description="Enqueue until the response was ready.")


class HealthResponse(BaseModel):
    status: str
    backend: str
    default_max_tokens: int
    max_batch_size: int
    max_wait_ms: float
    queue_depth: int
    batches_dispatched: int
    requests_served: int
    mean_batch_size: float
