"""Tests that the embeddings path warms the model on the loader thread.

``EmbeddingService.__init__`` runs one embed on the thread that loads the
model, which is the same thread that later calls
``MLXEmbeddingsHandler.initialize()`` and spawns the ``InferenceWorker``.
Without that warm-up, the model's deferred MLX state binds to the worker
thread on first use and cross-thread evaluations raise
``RuntimeError: There is no Stream(gpu, N) in current thread``.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest

from app.handler.embeddings.service import EmbeddingResult, EmbeddingService
from app.handler.mlx_embeddings import MLXEmbeddingsHandler
from app.schemas.openai import EmbeddingRequest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_bert"


@pytest.mark.asyncio
async def test_warm_up_runs_on_caller_thread_before_worker_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first embed happens on the caller thread, before the worker thread exists."""
    main_tid = threading.get_ident()
    call_threads: list[int] = []
    original_embed = EmbeddingService.embed

    def _spy_embed(self: EmbeddingService, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        call_threads.append(threading.get_ident())
        return original_embed(self, inputs, dimensions)

    monkeypatch.setattr(EmbeddingService, "embed", _spy_embed)

    # Construction happens before ``initialize()`` spawns the worker thread.
    handler = MLXEmbeddingsHandler(model_path=str(FIXTURE))
    assert call_threads == [main_tid], f"warm-up must run once on the caller thread; got {call_threads}"

    await handler.initialize({})
    try:
        response = await handler.generate_embeddings_response(EmbeddingRequest(input="hello world", model="tiny"))
    finally:
        handler.inference_worker.stop()

    # The real request ran on the worker thread, not the caller thread.
    assert call_threads[-1] != main_tid
    vec = response["embeddings"][0]
    assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-5


def test_warm_up_failure_does_not_block_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing warm-up embed is logged, not raised, so startup still completes."""

    def _broken_embed(self: EmbeddingService, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        raise ValueError("boom")

    monkeypatch.setattr(EmbeddingService, "embed", _broken_embed)

    service = EmbeddingService(model_path=str(FIXTURE))
    assert service.model is not None
