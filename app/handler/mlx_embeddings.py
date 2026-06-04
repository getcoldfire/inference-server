"""Embeddings handler: wires the OpenAI `/v1/embeddings` route to our
permissively-licensed `EmbeddingService` via the shared `InferenceWorker`
infrastructure.

Replaces the upstream handler that depended on the GPLv3 `mlx-embeddings`
package. The wrapper class name (`MLXEmbeddingsHandler`) and surface area
(handler_type, initialize, get_models, generate_embeddings_response,
get_queue_stats, cleanup) are preserved so the multi-model server lifespan
in `app/server.py` can swap implementations without further changes.
"""
import gc
from http import HTTPStatus
import time
from typing import Any

from fastapi import HTTPException
from loguru import logger

from ..core import InferenceWorker
from ..schemas.openai import EmbeddingRequest
from ..utils.errors import create_error_response
from .embeddings.service import EmbeddingService


class MLXEmbeddingsHandler:
    """Coldfire embeddings handler — permissively-licensed reimplementation.

    Holds a single `EmbeddingService` and forwards `/v1/embeddings`
    requests through the shared `InferenceWorker` queue so blocking MLX
    computation doesn't stall the asyncio event loop.

    Surface area (public-facing):

    - `handler_type = "embeddings"` — used by the route dispatcher to
      validate that this model is appropriate for `/v1/embeddings`.
    - `initialize(config)` — starts the inference worker thread.
    - `get_models()` — lists this model under the OpenAI `/v1/models`
      shape.
    - `generate_embeddings_response(request)` — the hot path: tokenize,
      forward, pool, normalize. Returns `list[list[float]]`.
    - `get_queue_stats()` — exposes inference-worker stats for `/queue`.
    - `cleanup()` — stops the inference worker and frees the model.
    """

    handler_type: str = "embeddings"

    def __init__(self, model_path: str):
        """Load the embedding model from disk or HF hub.

        Parameters
        ----------
        model_path : str
            Local directory OR HuggingFace repo ID. See
            `app.handler.embeddings.loader.load_embedding_model` for the
            exact filesystem layout expected.
        """
        self.model_path = model_path
        self.service = EmbeddingService(model_path)
        self.model_created = int(time.time())

        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation. Re-initialized in initialize()
        # with the correct queue_size + timeout from the multi-model config.
        self.inference_worker = InferenceWorker()

        logger.info(f"Initialized MLXEmbeddingsHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """Return OpenAI-shape `/v1/models` payload for this handler."""
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
            logger.error(f"Error getting models: {e!s}")
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
        logger.info("Initialized MLXEmbeddingsHandler and started inference worker")

    async def generate_embeddings_response(
        self, request: EmbeddingRequest
    ) -> list[list[float]]:
        """Embed `request.input` and return the list of vectors.

        The OpenAI route in `app/api/endpoints.py` wraps this output into
        an `EmbeddingResponse` with proper `data[i].embedding` shape and
        `usage` (currently None — we plan to add usage on the route side
        once the response wrapper supports it).

        Raises HTTPException(500) on inference failure with an OpenAI-shape
        error body for consistent client behavior.
        """
        try:
            if isinstance(request.input, str):
                inputs: list[str] = [request.input]
            else:
                inputs = request.input

            # Submit to the inference worker. We pass a no-arg callable
            # because EmbeddingService.embed is the unit of work.
            result = await self.inference_worker.submit(self.service.embed, inputs)
            return result.embeddings

        except Exception as e:
            logger.error(f"Error in embeddings generation: {e!s}")
            content = create_error_response(
                f"Failed to generate embeddings: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    async def get_queue_stats(self) -> dict[str, Any]:
        """Inference-worker stats: `queue_stats` sub-dict for the `/queue` route."""
        return {
            "queue_stats": self.inference_worker.get_stats(),
        }

    async def cleanup(self) -> None:
        """Stop the inference worker and release the model from memory."""
        try:
            logger.info("Cleaning up MLXEmbeddingsHandler resources")
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()
            # Drop refs so MLX GC can free GPU memory.
            if hasattr(self, "service"):
                self.service = None
            gc.collect()
            logger.info("MLXEmbeddingsHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXEmbeddingsHandler cleanup: {e!s}")
            raise

    def __del__(self):
        """Best-effort guard against double-cleanup; prefer explicit `await cleanup()`."""
        if hasattr(self, "_cleaned") and self._cleaned:
            return
        self._cleaned = True
