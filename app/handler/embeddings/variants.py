"""Position embedding primitives for BERT-family encoders.

Implements:

- `rope_frequencies(seq_len, head_dim, base)` — precompute RoPE cos/sin tables.
- `rotary_apply(x, cos, sin)` — apply RoPE rotation to Q/K tensors of shape
  `(batch, n_heads, seq, head_dim)`.

Referenced by `app.handler.embeddings.encoder.BertSelfAttention` when the
model config requests `position_embedding_type="rotary"`. The SwiGLU MLP
itself is implemented inline as `BertSwiGLUBlock` in `encoder.py` so that
its parameter names match the HuggingFace nomic-bert layout exactly
(`mlp.gate.weight`, `mlp.up.weight`, etc.) — wrapping it in a separate
`SwiGLUMLP` here would force an extra `.mlp.mlp.*` prefix that breaks the
safetensors key mapping.
"""

from __future__ import annotations

import mlx.core as mx


def rope_frequencies(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[mx.array, mx.array]:
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
    # 1 / base ^ (2i / head_dim) for i in [0, head_dim // 2)
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    positions = mx.arange(seq_len).astype(mx.float32)
    # Outer product: (seq_len, half)
    freqs = positions[:, None] * inv_freq[None, :]
    return mx.cos(freqs), mx.sin(freqs)


def rotary_apply(x: mx.array, cos: mx.array, sin: mx.array, *, interleaved: bool = False) -> mx.array:
    """Apply RoPE rotation to a Q/K tensor.

    Args:
        x: `(batch, n_heads, seq, head_dim)`. head_dim must be even.
        cos: `(seq, head_dim // 2)` precomputed cosine table.
        sin: `(seq, head_dim // 2)` precomputed sine table.
        interleaved: If True, use the GPT-J ("interleaved") convention that
            pairs adjacent channels (0,1),(2,3),... and rotates each pair.
            If False (default), use the GPT-NeoX / LLaMA ("half-split")
            convention that pairs channel ``i`` with channel ``i + dim/2``.
            **nomic-bert sets ``rotary_emb_interleaved: false`` in its
            config — i.e., it uses the half-split convention. Picking the
            wrong convention here scrambles every Q and K projection
            silently (output stays L2-unit and finite, but semantically
            random).

    Returns:
        Rotated tensor with the same shape as `x`.
    """
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    if interleaved:
        # GPT-J style: rotate adjacent (even, odd) channel pairs.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos_b - x_odd * sin_b
        rot_odd = x_even * sin_b + x_odd * cos_b
        stacked = mx.stack([rot_even, rot_odd], axis=-1)
        return stacked.reshape(x.shape)
    # GPT-NeoX / LLaMA / nomic-bert style: split the head_dim into two
    # halves and rotate them as a pair. cos/sin are precomputed for the
    # FIRST half; we use the same table for the second half (which is
    # what the standard "rotate_half" formulation produces — the second
    # half's contribution is `-x_second * sin` and `x_first * sin`).
    head_dim = x.shape[-1]
    half = head_dim // 2
    x_first = x[..., :half]
    x_second = x[..., half:]
    rot_first = x_first * cos_b - x_second * sin_b
    rot_second = x_second * cos_b + x_first * sin_b
    return mx.concatenate([rot_first, rot_second], axis=-1)
