"""
MLX model handlers for text, multimodal, and embeddings models.
"""

from typing import Any

__all__ = [
    "MLXLMHandler",
    "MLXVLMHandler",
    "MLXEmbeddingsHandler",
]


def __getattr__(name: str) -> Any:
    """Lazily import handlers so one backend does not initialize all MLX stacks."""
    if name == "MLXLMHandler":
        from .mlx_lm import MLXLMHandler

        return MLXLMHandler
    if name == "MLXVLMHandler":
        from .mlx_vlm import MLXVLMHandler

        return MLXVLMHandler
    if name == "MLXEmbeddingsHandler":
        from .mlx_embeddings import MLXEmbeddingsHandler

        return MLXEmbeddingsHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
