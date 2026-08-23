"""Inference backends.

Two implementations behind one interface:

  MockBackend  - sleeps for a modelled amount of time. Used for Phases 1-5.
  QwenBackend  - real Hugging Face transformers generation. Phase 6 only.

Both are *blocking* functions. The server dispatches them to a single worker
thread (see app/main.py), which is what makes the swap honest: changing
BACKEND=mock to BACKEND=qwen changes what generate() does and nothing at all
about the serving path around it.

Why blocking rather than `async def ... await asyncio.sleep()`: a real model
instance cannot run concurrent forward passes. If the mock yielded to the event
loop, N concurrent requests would all "infer" in parallel and the Phase 1
baseline would report throughput no real single-GPU server could reach --
making Phase 2's batching look like a regression. Serialising through one
thread reproduces the actual constraint.
"""

import random
import time
from typing import List

try:  # Python 3.8+
    from typing import Protocol
except ImportError:  # pragma: no cover - only for very old interpreters
    Protocol = object  # type: ignore[assignment,misc]

from app.config import Settings


class Backend(Protocol):
    """The one thing the serving layer needs from a model."""

    name: str

    def load(self) -> None:
        """Do any expensive setup. Called once, at startup."""
        ...

    def generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Blocking. Completes every prompt in one forward pass.

        Returns one completion per prompt, in the same order. A single prompt
        is just a batch of one, so this is the only entry point the serving
        layer needs.
        """
        ...


class MockBackend:
    """Fake inference: sleeps for base + per_token * max_tokens (+ jitter).

    The two-part cost model mirrors a real decoder -- a fixed prefill cost, then
    a cost proportional to the number of tokens generated. Tune the constants
    via MOCK_BASE_MS / MOCK_PER_TOKEN_MS so they sit in the neighbourhood of the
    real model's measured latency; the closer they are, the more the Phase 2
    batching result predicts the Phase 6 one.
    """

    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self._base_ms = settings.mock_base_ms
        self._per_token_ms = settings.mock_per_token_ms
        self._jitter_ms = settings.mock_jitter_ms
        self._batch_alpha = settings.mock_batch_alpha
        self._rng = random.Random(0xC0FFEE)

    def load(self) -> None:
        # Nothing to load. Defined so the interface matches the real backend.
        return None

    def sleep_ms_for(self, batch_size: int, max_tokens: int) -> float:
        """The modelled service time for one batched forward pass.

        Exposed rather than inlined so the scheduler's expected behaviour can
        be reasoned about (and tested) without re-deriving the formula.
        """
        batch_factor = 1.0 + self._batch_alpha * (batch_size - 1)
        planned = self._base_ms + self._per_token_ms * max_tokens * batch_factor
        if self._jitter_ms > 0:
            planned += self._rng.uniform(-self._jitter_ms, self._jitter_ms)
        return max(0.0, planned)

    def generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        # One sleep for the whole batch -- that IS the batching win. Every
        # sequence in a real batch advances together, one decode step at a
        # time, so the batch costs one (slightly inflated) generation, not N.
        time.sleep(self.sleep_ms_for(len(prompts), max_tokens) / 1000.0)
        return [
            f"[mock completion for {len(p)}-char prompt, {max_tokens} tokens, "
            f"batch={len(prompts)}]"
            for p in prompts
        ]


class QwenBackend:
    """Real generation via Hugging Face transformers.

    Wired up now so the interface is settled, but not exercised until Phase 6.
    torch/transformers are imported inside load() so that Phases 1-5 run
    without those packages installed at all (see requirements-model.txt).
    """

    name = "qwen"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        import torch  # noqa: F401  (imported for dtype/device resolution)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self._settings
        device = s.model_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype = None if s.model_dtype == "auto" else getattr(torch, s.model_dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(s.model_name)
        # Decoder-only models must be LEFT-padded for batched generation:
        # generation continues from the rightmost position, so right-padding
        # would have the model continue from pad tokens instead of the prompt.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            s.model_name,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()
        self._device = device

    def generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("QwenBackend.load() was not called")

        import torch

        texts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Strip the prompt tokens; keep only what was newly generated. With
        # left padding every row's prompt ends at the same column, so one
        # offset works for the whole batch.
        prompt_len = inputs["input_ids"].shape[1]
        return [
            self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            for row in output_ids
        ]


def build_backend(settings: Settings) -> Backend:
    if settings.backend == "mock":
        return MockBackend(settings)
    if settings.backend == "qwen":
        return QwenBackend(settings)
    raise ValueError(
        f"Unknown BACKEND={settings.backend!r}. Expected 'mock' or 'qwen'."
    )
