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
    max_tokens: int

    # Per-request timings, in milliseconds. Phase 1 reports these so the
    # baseline's cost can be split into "waiting for the model to be free"
    # versus "actually generating" -- the same split Phase 2 has to improve.
    wait_ms: float = Field(
        ..., description="Handler entry until inference actually started."
    )
    inference_ms: float = Field(..., description="Time inside backend.generate().")
    total_ms: float = Field(..., description="Handler entry until response built.")


class HealthResponse(BaseModel):
    status: str
    backend: str
    default_max_tokens: int
