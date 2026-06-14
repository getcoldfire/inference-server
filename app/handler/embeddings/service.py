"""High-level embedding service.

Wraps the trio of `loader.load_embedding_model`, the `BertModel` encoder
forward pass, `pooling.apply_pooling`, and `pooling.l2_normalize` into a
single `EmbeddingService` object that takes a list of strings and returns
a list of unit-norm embedding vectors plus token-count metadata.

This is the runtime-facing entry point — the OpenAI-compatible `/v1/embeddings`
route handler in `app/api/endpoints.py` calls into this service via the
`MLXEmbeddingsHandler` wrapper (which adds the inference-worker queue +
process-isolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from app.handler.embeddings.loader import load_embedding_model
from app.handler.embeddings.pooling import apply_pooling, l2_normalize
from app.handler.embeddings_common import apply_dimensions


@dataclass
class EmbeddingResult:
    """Return type of `EmbeddingService.embed`.

    Attributes
    ----------
    embeddings : list[list[float]]
        One vector per input string. Each vector has length
        `hidden_size` (or `matryoshka_dim` when truncation is configured)
        and L2 norm 1.0.
    prompt_tokens : int
        Sum of non-padding tokens across the batch. We do not separately
        count completion tokens for embedding requests, so total == prompt.
    total_tokens : int
        Same as `prompt_tokens` (kept for OpenAI usage-shape parity).
    """

    embeddings: list[list[float]]
    prompt_tokens: int
    total_tokens: int


class EmbeddingService:
    """Stateful embedding service that loads a model once and serves many requests.

    Lifecycle:
    - `__init__(model_path)` loads the model, tokenizer, and pooling/matryoshka
      configuration up-front. This is expensive; instantiate once per process.
    - `embed(inputs)` tokenizes the inputs, runs the encoder forward pass,
      pools to one vector per input, optionally truncates to `matryoshka_dim`,
      and L2-normalizes. Returns an `EmbeddingResult`.
    """

    def __init__(self, model_path: str):
        (
            self.model,
            self.tokenizer,
            self.pooling_mode,
            self.matryoshka_dim,
        ) = load_embedding_model(model_path)
        # Force every lazy MLX allocation in the model to materialize on the
        # *loader* thread (the subprocess main thread, where ``__init__`` is
        # called). The inference worker thread later runs the model under
        # *its* thread-local stream; without this warm-up, deferred
        # allocations made on first use can capture a stream reference from
        # the loader thread that the worker thread cannot resolve, surfacing
        # as ``RuntimeError: There is no Stream(gpu, N) in current thread.``
        # at the first ``embed(...)`` call. Mirror of the same workaround
        # applied in ``app.core.batch_scheduler.BatchScheduler.start()``
        # for the chat path.
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one trivial forward pass on the loader thread to flush lazy state."""
        try:
            # Encode the simplest possible non-empty input. ``"a"`` works for
            # every tokenizer we ship with; the result is discarded.
            self.embed(["a"])
        except Exception:  # noqa: BLE001 — best-effort warm-up; surface later
            # If warm-up fails the model is broken anyway; let the first real
            # ``embed(...)`` call surface the same exception with the same
            # traceback (which is more useful than a sanitized warm-up error).
            pass

    def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        """Embed a batch of input strings.

        Parameters
        ----------
        inputs : List[str]
            Input texts to embed. Empty list short-circuits to an empty
            result with zero token counts.
        dimensions : int | None, optional
            Per-request matryoshka truncation. When set, overrides the
            model-config ``matryoshka_dim`` for this call only — the
            returned vectors will have ``dimensions`` components instead
            of the model's native hidden size. Must be >0 and <= the
            model's native dim. Truncation happens before L2 normalize
            so cosine similarity stays well-defined.

        Empty ``inputs`` -> empty result (no tokenizer/model call). This
        keeps the OpenAI ``/v1/embeddings`` happy when callers pass ``[]``.
        """
        if not inputs:
            return EmbeddingResult(embeddings=[], prompt_tokens=0, total_tokens=0)

        # tokenizer.model_max_length is sometimes a sentinel value
        # (`int(1e30)`) for models that declare "no max"; clamp to 8192
        # so tokenize() doesn't allocate an absurd buffer.
        raw_max = self.tokenizer.model_max_length  # type: ignore[attr-defined]
        max_len = raw_max if raw_max < 10000 else 8192

        encoded = self.tokenizer(  # type: ignore[operator]
            inputs,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=max_len,
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # nomic-embed and many other modern embedding models do NOT use
        # token_type_ids. Pass None when absent; the encoder's BertEmbeddings
        # module is wired to skip the token-type embedding term in that case.
        if "token_type_ids" in encoded:
            token_type_ids = mx.array(encoded["token_type_ids"])
        else:
            token_type_ids = None

        hidden = self.model(input_ids, token_type_ids, attention_mask)
        pooled = apply_pooling(hidden, attention_mask, self.pooling_mode)

        # Matryoshka MUST be applied BEFORE L2 normalization. If we truncated
        # AFTER normalizing, the truncated vector would have norm < 1 and
        # downstream cosine similarity would be off. Truncating before lets
        # the final normalize step re-project onto the unit sphere.
        #
        # Precedence: per-request `dimensions` overrides the model-config
        # `matryoshka_dim`. This mirrors the OpenAI API behavior where the
        # request field always wins.
        effective_dim = dimensions if dimensions is not None else self.matryoshka_dim

        prompt_tokens = int(attention_mask.sum().item())

        if effective_dim is not None:
            # Delegate truncation + L2 renormalization to the shared helper so
            # the BERT path and the llama-cpp path apply identical matryoshka
            # semantics. apply_dimensions validates the target dim, truncates,
            # and renormalizes per vector — equivalent to the former inline
            # batch truncation + l2_normalize(pooled[:, :effective_dim]).
            raw_rows: list[list[float]] = pooled.tolist()  # type: ignore[assignment]
            embeddings: list[list[float]] = [
                apply_dimensions(np.array(row, dtype=np.float32), effective_dim).tolist()
                for row in raw_rows
            ]
        else:
            normalized = l2_normalize(pooled)
            # tolist() on a 2-D mlx.array returns list[list[float]] at runtime,
            # but the stub signature is the generic union — narrow it explicitly.
            embeddings = normalized.tolist()  # type: ignore[assignment]

        return EmbeddingResult(
            embeddings=embeddings,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        )
