"""BERT-family encoder for embedding generation.

Adapted from ml-explore/mlx-examples/bert (MIT). Original copyright Apple Inc.
Modifications for HuggingFace-compatible weight naming and config-driven variant
dispatch (RoPE, SwiGLU, optional token-type-ids, matryoshka truncation) are
Copyright (c) 2026 Coldfire, also MIT-licensed.

The class hierarchy mirrors HuggingFace's `BertModel` attribute layout so that
safetensors saved by `transformers` load directly via `nn.Module.load_weights`:

    BertModel
        embeddings
            word_embeddings, [position_embeddings], [token_type_embeddings], LayerNorm
        encoder
            layer[i]
                attention.self.{query, key, value}
                attention.output.{dense, LayerNorm}
                intermediate.dense              (vanilla GeLU path)
                output.{dense, LayerNorm}       (vanilla GeLU path)
                mlp.{gate, up, down, LayerNorm} (SwiGLU path)

Per `position_embedding_type`:

- "absolute" — adds a learned position embedding inside `BertEmbeddings`.
- "rotary"   — skips the position embedding term; applies RoPE inside
               `BertSelfAttention` to Q and K each forward pass.

Per `hidden_act`:

- "gelu"     — uses `intermediate` + `output` Linear pair.
- "swiglu"   — uses gate/up/down SwiGLU triple.

`token_type_ids` may be `None`; the embedding contribution is skipped when the
config has `type_vocab_size == 0` (as in nomic-bert).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn

from app.handler.embeddings.variants import (
    rope_frequencies,
    rotary_apply,
)


@dataclass
class BertConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_act: str = "gelu"
    position_embedding_type: str = "absolute"
    layer_norm_eps: float = 1e-12
    rope_theta: float = 10000.0
    matryoshka_dim: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> BertConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ---------- Embeddings ----------


class BertEmbeddings(nn.Module):
    """Word + (optional) position + (optional) token-type embeddings + LayerNorm.

    The position term is only included when `position_embedding_type == "absolute"`.
    The token-type term is only included when `type_vocab_size > 0` AND the caller
    passes `token_type_ids`.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None,
    ) -> mx.array:
        x = self.word_embeddings(input_ids)
        if self.config.position_embedding_type == "absolute":
            positions = mx.arange(input_ids.shape[1])
            x = x + self.position_embeddings(positions)
        if token_type_ids is not None and self.config.type_vocab_size > 0:
            x = x + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(x)


# ---------- Self-attention ----------


class BertSelfAttention(nn.Module):
    """Multi-head self-attention with HF weight names and optional RoPE.

    HF layout: `attention.self.{query, key, value}` are three separate Linear
    layers each `(hidden_size, hidden_size)`. We reshape into multi-head form
    inside __call__.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        B, T, _ = x.shape
        q = self.query(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.config.position_embedding_type == "rotary":
            cos, sin = rope_frequencies(T, self.head_dim, self.config.rope_theta)
            q = rotary_apply(q, cos, sin)
            k = rotary_apply(k, cos, sin)

        scale = self.head_dim**-0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        if attention_mask is not None:
            # attention_mask: (B, T) with 1 for real, 0 for pad.
            # Broadcast to (B, 1, 1, T) and add a large negative bias on pads.
            mask_bias = (1.0 - attention_mask.astype(scores.dtype))[:, None, None, :] * -1e9
            scores = scores + mask_bias

        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return out


class BertSelfOutput(nn.Module):
    """Dense + residual-add + LayerNorm after self-attention."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(residual + self.dense(x))


class BertAttention(nn.Module):
    """Wrapper combining self-attention + output projection (HF layout)."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        attn_out = self.self(x, attention_mask)
        return self.output(attn_out, x)


# ---------- MLP variants ----------


class BertGeLUMLP(nn.Module):
    """Vanilla BERT MLP: intermediate + GeLU + output + residual + LayerNorm.

    Mirrors HF layout exactly:
        intermediate.dense  : Linear(H, I)
        output.dense        : Linear(I, H)
        output.LayerNorm    : LayerNorm(H)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.intermediate = _IntermediateBlock(config)
        self.output = _OutputBlock(config)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.intermediate(x)
        return self.output(h, x)


class _IntermediateBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.dense(x))


class _OutputBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(residual + self.dense(x))


class BertSwiGLUBlock(nn.Module):
    """Nomic-bert SwiGLU MLP block: SwiGLU + residual + LayerNorm.

    Layout matches our nomic fixture safetensors keys:
        mlp.gate.weight       : Linear(H, I) no bias
        mlp.up.weight         : Linear(H, I) no bias
        mlp.down.weight       : Linear(I, H) no bias
        mlp.LayerNorm.{weight,bias}
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        # Inner SwiGLU module — its sub-Linear names (gate/up/down) become
        # `mlp.gate`, `mlp.up`, `mlp.down` since this module is attached as `.mlp`
        # on the BertLayer.
        # We expose gate/up/down as direct children so weights load as
        # `mlp.gate.weight`, `mlp.up.weight`, `mlp.down.weight`.
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        gated = self.down(nn.silu(self.gate(x)) * self.up(x))
        return self.LayerNorm(x + gated)


# ---------- Layer + Encoder ----------


class BertLayer(nn.Module):
    """Single transformer encoder block.

    HF naming convention:
        attention.{self.{query,key,value}, output.{dense,LayerNorm}}
        intermediate.dense   (vanilla)
        output.{dense,LayerNorm} (vanilla)
        mlp.{gate,up,down,LayerNorm}  (SwiGLU)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.attention = BertAttention(config)
        if config.hidden_act == "swiglu":
            self.mlp = BertSwiGLUBlock(config)
        else:
            # vanilla path: exposes self.intermediate and self.output, matching
            # HF BERT weight names.
            self.intermediate = _IntermediateBlock(config)
            self.output = _OutputBlock(config)

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        attn = self.attention(x, attention_mask)
        if self.config.hidden_act == "swiglu":
            return self.mlp(attn)
        h = self.intermediate(attn)
        return self.output(h, attn)


class BertEncoder(nn.Module):
    """Stack of `num_hidden_layers` BertLayer modules."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, attention_mask: mx.array | None) -> mx.array:
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


# ---------- Top-level model ----------


class BertModel(nn.Module):
    """HF-compatible BERT-family encoder.

    Forward signature: `model(input_ids, token_type_ids, attention_mask)`
    returns the final hidden states `(batch, seq, hidden_size)`. No pooling
    happens here; callers apply mean/cls/etc. pooling on top.

    `token_type_ids` may be `None` for models like nomic-bert that don't use
    type ids.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None,
        attention_mask: mx.array | None,
    ) -> mx.array:
        x = self.embeddings(input_ids, token_type_ids)
        x = self.encoder(x, attention_mask)
        return x
