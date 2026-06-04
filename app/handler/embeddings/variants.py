"""Position embedding and MLP variants for BERT-family encoders.

Implements:

- `rope_frequencies(seq_len, head_dim, base)` — precompute RoPE cos/sin tables.
- `rotary_apply(x, cos, sin)` — apply RoPE rotation to Q/K tensors of shape
  `(batch, n_heads, seq, head_dim)`.
- `SwiGLUMLP` — gated SwiGLU MLP block used by nomic-bert and similar
  modern encoders.

These are referenced by `app.handler.embeddings.encoder.BertModel` when
the model config requests `position_embedding_type="rotary"` or
`hidden_act="swiglu"`.
"""
from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def rope_frequencies(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> Tuple[mx.array, mx.array]:
    """Precompute cos/sin tables for Rotary Position Embedding.

    Args:
        seq_len: Number of positions to precompute (sequence length).
        head_dim: Per-head channel dimension. Must be even.
        base: RoPE theta (default 10000.0).

    Returns:
        (cos, sin) — each shape `(seq_len, head_dim // 2)`, dtype float32.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    half = head_dim // 2
    # 1 / base ^ (2i / head_dim) for i in [0, half)
    inv_freq = 1.0 / (
        base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
    )
    positions = mx.arange(seq_len).astype(mx.float32)
    # Outer product: (seq_len, half)
    freqs = positions[:, None] * inv_freq[None, :]
    return mx.cos(freqs), mx.sin(freqs)


def rotary_apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE rotation to a Q/K tensor.

    Args:
        x: `(batch, n_heads, seq, head_dim)`. head_dim must be even.
        cos: `(seq, head_dim // 2)` precomputed cosine table.
        sin: `(seq, head_dim // 2)` precomputed sine table.

    Returns:
        Rotated tensor with the same shape as `x`.

    The rotation interleaves consecutive (even, odd) channel pairs:
    given `(x_even, x_odd)`, output is
    `(x_even * cos - x_odd * sin, x_even * sin + x_odd * cos)`.
    """
    # Split into even/odd channels along the last dim.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Broadcast cos/sin to (1, 1, seq, half) so they line up against
    # (batch, n_heads, seq, half).
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b
    # Re-interleave the two halves back into (..., head_dim).
    # Stacking on a new last axis gives (..., half, 2); reshape collapses that
    # to (..., head_dim) interleaved as even,odd,even,odd,...
    stacked = mx.stack([rot_even, rot_odd], axis=-1)
    return stacked.reshape(x.shape)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP variant used by nomic-bert and many modern transformers.

    Forward: `down(silu(gate(x)) * up(x))`. No biases on the linear layers
    (mirrors the convention used by `nomic-embed-text-v1.5`).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))
