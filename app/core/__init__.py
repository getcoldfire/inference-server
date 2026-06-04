"""Core server utilities with lazy backend imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BatchChunk",
    "BatchScheduler",
    "HandlerProcessProxy",
    "InferenceWorker",
    "ModelRegistry",
]


def __getattr__(name: str) -> Any:
    """Lazily import core helpers so one backend does not initialize all stacks."""
    if name == "BatchChunk":
        from .batch_scheduler import BatchChunk

        return BatchChunk
    if name == "BatchScheduler":
        from .batch_scheduler import BatchScheduler

        return BatchScheduler
    if name == "HandlerProcessProxy":
        from .handler_process import HandlerProcessProxy

        return HandlerProcessProxy
    if name == "InferenceWorker":
        from .inference_worker import InferenceWorker

        return InferenceWorker
    if name == "ModelRegistry":
        from .model_registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
