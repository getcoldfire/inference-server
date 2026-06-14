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

    Behavior:
      - target_dim == len(vec): pass-through (no truncation; already L2-unit).
      - target_dim < len(vec):  truncate then renormalize.
      - target_dim > len(vec):  ValueError ("exceeds model embedding size").
      - target_dim < 1:         ValueError ("must be positive").
      - vec is all-zero:        return zeros of length target_dim (no division).

    The returned array's dtype matches the input dtype.
    """
    if target_dim < 1:
        raise ValueError(f"dimensions must be positive integer; got {target_dim}")
    if target_dim > len(vec):
        raise ValueError(
            f"dimensions {target_dim} exceeds model embedding size {len(vec)}"
        )
    truncated = vec[:target_dim]
    norm = float(np.linalg.norm(truncated))
    if norm == 0.0:
        return truncated.copy()
    return (truncated / norm).astype(vec.dtype)
