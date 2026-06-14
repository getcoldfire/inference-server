"""llama-cpp embeddings handler — wraps LlamaCppEmbeddingsService behind the
same public surface as MLXEmbeddingsHandler so the multi-model lifespan and
route dispatcher treat both identically.

Wire advertisement: ``handler_type = "embeddings"`` (the FUNCTIONAL kind).
Internal dispatch key: ``model_type = "llama-cpp"`` in config/admin schema.
Consumers and cli-v2 modelprobe classify on the functional kind; they don't
need to learn the internal runtime kind.
"""

from __future__ import annotations

import gc
import time
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException
from loguru import logger

from ..core.inference_worker import InferenceWorker
from ..schemas.openai import EmbeddingRequest
from ..utils.errors import create_error_response
from .llama_cpp.loader import LlamaCppConfig, LlamaCppEmbeddingsLoader
from .llama_cpp.service import LlamaCppEmbeddingsService


class LlamaCppEmbeddingsHandler:
    """llama-cpp-python embeddings handler.

    Holds a single ``LlamaCppEmbeddingsService`` and forwards
    ``/v1/embeddings`` requests through the shared ``InferenceWorker``
    queue so blocking llama-cpp computation doesn't stall the asyncio
    event loop.

    Surface area (public-facing — mirrors MLXEmbeddingsHandler):

    - ``handler_type = "embeddings"`` — functional kind; routes to
      ``/v1/embeddings`` and is advertised verbatim in ``/v1/models``.
    - ``initialize(config)`` — starts the inference worker thread.
    - ``get_models()`` — lists this model under the OpenAI ``/v1/models``
      shape.
    - ``generate_embeddings_response(request)`` — hot path; returns
      ``{"embeddings": list[list[float]], "usage": {...}}``.
    - ``get_queue_stats()`` — exposes worker stats for ``/queue``.
    - ``cleanup()`` — stops the inference worker.
    """

    handler_type: str = "embeddings"

    def __init__(
        self,
        model_path: str,
        hf_file: str | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        n_threads: int | None = None,
    ) -> None:
        """Configure and lazily prepare the llama-cpp embedding model.

        Parameters
        ----------
        model_path : str
            HuggingFace repo ID (``"org/repo"``) OR absolute local ``.gguf``
            path.  HF repo requires ``hf_file``.
        hf_file : str | None
            Filename inside the HF repo (e.g. ``"model.f16.gguf"``).  Ignored
            for local ``model_path``.
        n_gpu_layers : int | None
            Number of layers offloaded to Metal.  Defaults to ``-1`` (all).
        n_ctx : int | None
            Context window size.  ``None`` uses llama-cpp defaults.
        n_batch : int | None
            Batch size for prompt processing.  ``None`` uses llama-cpp
            defaults.
        n_threads : int | None
            CPU threads for inference.  ``None`` uses llama-cpp defaults.
        """
        self.model_path = model_path
        self.model_created = int(time.time())

        cfg = LlamaCppConfig(
            model_path=model_path,
            hf_file=hf_file,
            # Preserve user intent: None means "not set"; apply our default
            # (-1 = all layers to Metal) only when the caller omitted the arg.
            n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else -1,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
        )
        loader = LlamaCppEmbeddingsLoader(cfg)
        self.service = LlamaCppEmbeddingsService(loader)

        # Inference worker is (re)initialized in initialize() with proper
        # queue_size / timeout from the multi-model server config.
        self.inference_worker = InferenceWorker()

        logger.info(
            "Initialized LlamaCppEmbeddingsHandler with model path: {}",
            model_path,
        )

    async def get_models(self) -> list[dict[str, Any]]:
        """Return OpenAI-shape ``/v1/models`` payload for this handler."""
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                }
            ]
        except Exception as e:
            logger.error("Error getting models: {}", str(e))
            return []

    async def initialize(self, config: dict[str, Any]) -> None:
        """Re-build the inference worker with the multi-model server's queue settings.

        Parameters
        ----------
        config : dict[str, Any]
            ``queue_size`` (int, default 100) and ``timeout`` (float seconds,
            default 300) for the inference worker.
        """
        self.inference_worker = InferenceWorker(
            queue_size=config.get("queue_size", 100),
            timeout=config.get("timeout", 300),
        )
        self.inference_worker.start()
        logger.info(
            "Initialized LlamaCppEmbeddingsHandler and started inference worker"
        )

    async def generate_embeddings_response(
        self, request: EmbeddingRequest
    ) -> dict[str, Any]:
        """Embed ``request.input`` and return ``{"embeddings": [...], "usage": {...}}``.

        llama-cpp-python does not expose per-request token counts, so ``usage``
        is approximated by summing ``len(text.split())`` across inputs — same
        order of magnitude, good enough for quota tracking.

        Raises HTTPException(500) on inference failure with an OpenAI-shape
        error body for consistent client behaviour.
        """
        try:
            if isinstance(request.input, str):
                inputs: list[str] = [request.input]
            else:
                inputs = list(request.input)

            # Submit to the inference worker — keeps blocking llama-cpp call
            # off the asyncio event loop.
            embeddings: list[list[float]] = await self.inference_worker.submit(
                self.service.embed, inputs, request.dimensions
            )

            # Approximate token usage (llama-cpp doesn't expose exact counts).
            approx_tokens = sum(len(t.split()) for t in inputs)

            return {
                "embeddings": embeddings,
                "usage": {
                    "prompt_tokens": approx_tokens,
                    "total_tokens": approx_tokens,
                },
            }

        except Exception as e:
            logger.error("Error in llama-cpp embeddings generation: {}", str(e))
            content = create_error_response(
                f"Failed to generate embeddings: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content) from e

    async def get_queue_stats(self) -> dict[str, Any]:
        """Inference-worker stats: ``queue_stats`` sub-dict for the ``/queue`` route."""
        return {
            "queue_stats": self.inference_worker.get_stats(),
        }

    async def cleanup(self) -> None:
        """Stop the inference worker and release the llama-cpp model from memory."""
        try:
            logger.info("Cleaning up LlamaCppEmbeddingsHandler resources")
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()
            # Drop refs — llama-cpp-python frees the underlying C++ object
            # via its own __del__; no explicit unload API needed.
            if hasattr(self, "service"):
                self.service = None
            gc.collect()
            logger.info("LlamaCppEmbeddingsHandler cleanup completed successfully")
        except Exception as e:
            logger.error("Error during LlamaCppEmbeddingsHandler cleanup: {}", str(e))
            raise

    def __del__(self) -> None:
        """Best-effort guard against double-cleanup; prefer explicit ``await cleanup()``."""
        if hasattr(self, "_cleaned") and self._cleaned:
            return
        self._cleaned = True
