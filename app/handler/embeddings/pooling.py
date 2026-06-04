"""Pooling strategies + L2 normalization for embedding models.

Embedding models emit one hidden state per token; downstream consumers
need a single fixed-length vector per input. `apply_pooling` collapses
the per-token dim with the mode declared by the model's
`1_Pooling/config.json` sidecar (sentence-transformers convention).

`l2_normalize` is the final step applied AFTER any matryoshka truncation
so that the returned vector lives on the unit hypersphere (which is what
cosine-similarity downstream consumers expect).
"""
from __future__ import annotations

import mlx.core as mx


def apply_pooling(
    hidden_states: mx.array, attention_mask: mx.array, mode: str
) -> mx.array:
    """Pool a `(batch, seq, hidden)` tensor along the seq axis.

    Parameters
    ----------
    hidden_states : mx.array
        Shape `(batch, seq, hidden)` — encoder output.
    attention_mask : mx.array
        Shape `(batch, seq)`. 1 for real tokens, 0 for padding.
    mode : str
        One of "mean", "cls", "last_token", "max".

    Returns
    -------
    mx.array
        Shape `(batch, hidden)`.

    Raises
    ------
    ValueError
        If `mode` is not one of the supported strings.
    """
    if mode == "cls":
        # Position 0 is the [CLS] token in BERT-style tokenizers.
        return hidden_states[:, 0, :]

    if mode == "mean":
        # Broadcast mask to (batch, seq, 1), then sum over the seq axis
        # and divide by the count of unmasked positions per row. Clamp the
        # divisor to 1 so an all-padding row doesn't blow up to NaN.
        mask = attention_mask[:, :, None].astype(hidden_states.dtype)
        summed = (hidden_states * mask).sum(axis=1)
        counts = mask.sum(axis=1)
        return summed / mx.maximum(counts, mx.array(1.0, dtype=hidden_states.dtype))

    if mode == "last_token":
        # The last unmasked index per row. `sum(mask) - 1` gives that index
        # assuming the mask is contiguous (i.e. real tokens first, padding
        # last) which matches the HF tokenizer convention with right-padding.
        seq_lens = attention_mask.sum(axis=1).astype(mx.int32) - 1
        # Clamp negatives (the "all-masked" pathological case) to 0 so we
        # don't index out-of-bounds.
        seq_lens = mx.maximum(seq_lens, mx.array(0, dtype=mx.int32))
        batch_size = hidden_states.shape[0]
        batch_idx = mx.arange(batch_size).astype(mx.int32)
        # Fancy indexing: hidden_states[i, seq_lens[i], :]
        return hidden_states[batch_idx, seq_lens]

    if mode == "max":
        # Set masked positions to a large negative bias so they can't win
        # the per-channel max. We don't mutate hidden_states.
        mask_bias = (attention_mask[:, :, None] == 0).astype(hidden_states.dtype) * -1e9
        return (hidden_states + mask_bias).max(axis=1)

    raise ValueError(f"unknown pooling mode: {mode!r}")


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    """Row-wise L2 normalize a 2-D matrix.

    Parameters
    ----------
    x : mx.array
        Shape `(batch, hidden)`.
    eps : float
        Numerical floor on the norm to avoid division-by-zero on all-zero rows.

    Returns
    -------
    mx.array
        Shape `(batch, hidden)`, each row has L2 norm ~= 1 (or 0 for an
        all-zero input row).
    """
    norm = mx.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)
    return x / norm
