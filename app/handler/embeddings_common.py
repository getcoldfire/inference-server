# app/handler/embeddings_common.py
"""Shared post-processing for embedding handlers.

Both the BERT-encoder path (app/handler/embeddings/) and the llama-cpp
path (app/handler/llama_cpp/) call this module so matryoshka truncation
behaves identically across runtimes. Drift between the two is a silent
correctness bug for cross-handler vector compatibility.
"""
from __future__ import annotations

import numpy as np


def apply_dimensions(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Truncate `vec` to `target_dim` and L2-renormalize.

    Accepts 1-D (single vector) or 2-D (batch of vectors, shape (batch, dim)).
    For 2-D input, truncation and renormalization happen per row.

    Behavior:
      - target_dim == vec.shape[-1]: no truncation, but L2-renormalize is still applied.
      - target_dim <  vec.shape[-1]: truncate then renormalize.
      - target_dim >  vec.shape[-1]: ValueError ("exceeds model embedding size").
      - target_dim <  1:            ValueError ("must be positive").
      - Rows that are all-zero in 2-D input (or a 1-D zero vector): returned as zeros (no division).

    The returned array's dtype matches the input dtype.
    """
    if target_dim < 1:
        raise ValueError(f"dimensions must be positive integer; got {target_dim}")
    native_dim = vec.shape[-1]
    if target_dim > native_dim:
        raise ValueError(
            f"dimensions {target_dim} exceeds model embedding size {native_dim}"
        )
    truncated = vec[..., :target_dim]
    norms = np.linalg.norm(truncated, axis=-1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero; the zero rows stay zero.
    safe = np.where(norms == 0.0, 1.0, norms)
    return (truncated / safe).astype(vec.dtype)
