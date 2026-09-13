"""Inference backends.

Two implementations behind one interface:

  MockBackend  - sleeps for a modelled amount of time. Used for Phases 1-5.
  QwenBackend  - real Hugging Face transformers generation. Phase 6 only.

Both are *blocking* generators. The server drives them from a single worker
thread (see app/scheduler.py), which is what makes the swap honest: changing
BACKEND=mock to BACKEND=qwen changes what happens inside the decode loop and
nothing at all about the serving path around it.

Why blocking rather than `async`: a real model instance cannot run concurrent
forward passes. If the mock yielded to the event loop, N concurrent requests
would all "infer" in parallel and the Phase 1 baseline would report throughput
no real single-GPU server could reach -- making Phase 2's batching look like a
regression. Serialising through one thread reproduces the actual constraint.

Why a generator (Phase 3): a transformer produces text one decode step at a
time, and every step advances EVERY sequence in the batch by one token, in
lockstep. Streaming is therefore not in tension with batching -- it is just
handing each sequence its token after each step instead of collecting them all
and returning a bucket at the end. The backend yields per step; the scheduler
decides whether to forward tokens immediately or accumulate them.
"""

import random
import time
from typing import Iterator, List, Optional

try:  # Python 3.8+
    from typing import Protocol
except ImportError:  # pragma: no cover - only for very old interpreters
    Protocol = object  # type: ignore[assignment,misc]

from app.config import Settings

# One decode step's output: one entry per prompt in the batch, in order. None
# means that sequence has already finished and produced nothing this step.
Step = List[Optional[str]]


class Backend(Protocol):
    """The one thing the serving layer needs from a model."""

    name: str

    def load(self) -> None:
        """Do any expensive setup. Called once, at startup."""
        ...

    def generate_batch_stream(
        self, prompts: List[str], max_tokens: int
    ) -> Iterator[Step]:
        """Blocking generator. Runs one forward pass per decode step for the
        whole batch and yields each step's tokens, one entry per prompt.

        A single prompt is a batch of one, and a non-streaming response is just
        every step concatenated, so this is the only entry point the serving
        layer needs.
        """
        ...


# Vocabulary for the mock's output. Cycling through real words rather than
# emitting "tok17" makes a streamed response look like a stream, which matters
# when demonstrating SSE to someone in a browser or terminal.
_WORDS = (
    "the quick brown fox jumps over the lazy dog while the serving layer "
    "batches requests behind it and streams each token the moment the decode "
    "step that produced it completes"
).split()


def _sleep_until(deadline: float) -> None:
    """Block until perf_counter() reaches deadline. No-op if already past."""
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


class MockBackend:
    """Fake inference with a two-part cost model.

        prefill:  base_ms                                  (once per batch)
        decode:   per_token_ms * (1 + alpha * (N - 1))     (once per step)

    Total for a batch of N over T steps is identical to the Phase 2 single-sleep
    version -- base + per_token * T * (1 + alpha(N-1)) -- so throughput numbers
    carry over unchanged. The only difference is that the sleep is spent one
    step at a time, which is what makes each token available as it is produced.
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

    def batch_factor(self, batch_size: int) -> float:
        return 1.0 + self._batch_alpha * (batch_size - 1)

    def sleep_ms_for(self, batch_size: int, max_tokens: int) -> float:
        """Total modelled service time for one batched generation. Exposed so
        the scheduler's expected behaviour can be reasoned about without
        re-deriving the formula."""
        return self._base_ms + self._per_token_ms * max_tokens * self.batch_factor(batch_size)

    def generate_batch_stream(
        self, prompts: List[str], max_tokens: int
    ) -> Iterator[Step]:
        n = len(prompts)
        # Prefill: the whole batch's prompts go through once. Jitter is applied
        # here rather than per step so the per-step cost stays deterministic
        # and inter-token latency reads as a clean number.
        prefill_s = self._base_ms / 1000.0
        if self._jitter_ms > 0:
            prefill_s += self._rng.uniform(-self._jitter_ms, self._jitter_ms) / 1000.0
        prefill_s = max(0.0, prefill_s)
        step_s = (self._per_token_ms * self.batch_factor(n)) / 1000.0

        # Sleep to absolute deadlines, not for durations. time.sleep() reliably
        # overshoots by a millisecond or two, and across 64 steps that
        # compounds to ~20% -- enough to make the Phase 2 regression check fail
        # for a reason that has nothing to do with the design. Anchoring every
        # step to t0 lets one step's overshoot be absorbed by the next, so the
        # total lands on the modelled figure.
        t0 = time.perf_counter()
        _sleep_until(t0 + prefill_s)

        # Decode: one step per token, every sequence advances together. This
        # is the lockstep that makes batched streaming possible.
        for step in range(max_tokens):
            _sleep_until(t0 + prefill_s + (step + 1) * step_s)
            word = _WORDS[step % len(_WORDS)]
            yield [word + " "] * n


class QwenBackend:
    """Real generation via Hugging Face transformers.

    Wired up now so the interface is settled, but not exercised until Phase 6.
    torch/transformers are imported inside load() so that Phases 1-5 run
    without those packages installed at all (see requirements-model.txt).

    Streaming uses a manual decode loop with a KV cache rather than
    model.generate(): generate() returns only when every sequence is done, and
    TextIteratorStreamer only handles a single sequence. The loop below is the
    same thing generate() does internally, made visible one step at a time.
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

    def generate_batch_stream(
        self, prompts: List[str], max_tokens: int
    ) -> Iterator[Step]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("QwenBackend.load() was not called")

        import torch

        tok = self._tokenizer
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(self._device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        n = input_ids.shape[0]
        eos = tok.eos_token_id

        finished = [False] * n
        past = None
        next_input = input_ids

        with torch.no_grad():
            for _ in range(max_tokens):
                out = self._model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                # Greedy, to match the Phase 2 do_sample=False behaviour.
                next_tokens = out.logits[:, -1, :].argmax(dim=-1)

                step: Step = []
                for i in range(n):
                    if finished[i]:
                        step.append(None)
                        continue
                    t = int(next_tokens[i])
                    if t == eos:
                        finished[i] = True
                        step.append(None)
                    else:
                        step.append(tok.decode([t], skip_special_tokens=True))
                yield step

                if all(finished):
                    break

                next_input = next_tokens.unsqueeze(-1)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((n, 1))], dim=-1
                )


def build_backend(settings: Settings) -> Backend:
    if settings.backend == "mock":
        return MockBackend(settings)
    if settings.backend == "qwen":
        return QwenBackend(settings)
    raise ValueError(
        f"Unknown BACKEND={settings.backend!r}. Expected 'mock' or 'qwen'."
    )
