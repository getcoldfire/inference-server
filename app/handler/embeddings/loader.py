"""HuggingFace safetensors -> MLX weight loader for embedding models.

Reads `config.json`, `model.safetensors`, tokenizer files, and the optional
sentence-transformers `1_Pooling/config.json` + `2_Matryoshka/config.json`
sidecar files. Validates that the variant declared by the config is one we
actually support (see `SUPPORTED_ACTIVATIONS`, `SUPPORTED_POSITION_EMBEDDINGS`)
and raises `UnsupportedModelError` early — fail loud, never silently
fall back to a different code path.

Real HuggingFace repos use a handful of different weight-key conventions
(e.g. nomic-bert's combined `attn.Wqkv` vs upstream BERT's separate
`attention.self.{query,key,value}`); `_remap_hf_to_internal` collapses those
to the single key layout used by `app.handler.embeddings.encoder.BertModel`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors.numpy import load_file
from transformers import AutoTokenizer

from app.handler.embeddings.encoder import BertConfig, BertModel

SUPPORTED_ACTIVATIONS = {"gelu", "swiglu"}
SUPPORTED_POSITION_EMBEDDINGS = {"absolute", "rotary"}


class UnsupportedModelError(ValueError):
    """Raised when a model config requests a variant we do not implement.

    The error message names the offending field and lists the supported
    values, so the operator can decide between (a) filing an issue to add
    support, (b) replacing the model with a supported variant, or (c)
    extending `SUPPORTED_ACTIVATIONS` / `SUPPORTED_POSITION_EMBEDDINGS`
    if we already know how to handle the variant.
    """


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ``cfg`` with nomic-bert-style fields normalized to BertConfig naming.

    The real ``nomic-ai/nomic-embed-text-v1.5`` config uses a mix of conventions
    that don't match ``BertConfig``'s expected field names:

    - ``hidden_act='silu'`` — nomic's marker for the SwiGLU MLP triple
      (``fc11`` / ``fc12`` / ``fc2``). We rewrite to ``'swiglu'`` so the
      encoder dispatches to ``BertSwiGLUBlock``.
    - ``activation_function='swiglu'`` — alternate field nomic also sets; we
      let it override ``hidden_act`` when present.
    - No ``position_embedding_type`` field, but ``rotary_emb_fraction > 0``
      implies rotary embeddings. We infer ``'rotary'`` in that case.
    - ``layer_norm_epsilon`` (note the trailing ``on``) instead of
      ``layer_norm_eps`` — we copy the value across so ``BertConfig`` reads it.
    - ``rotary_emb_base`` (or ``rope_parameters.rope_theta``) instead of
      ``rope_theta`` — we copy the value across.
    """
    out = dict(cfg)

    # Activation normalization. ``activation_function`` takes precedence
    # because the nomic config sets both — and ``activation_function`` already
    # uses the canonical ``'swiglu'`` while ``hidden_act='silu'`` is the
    # nomic-internal shorthand.
    if "activation_function" in out:
        out["hidden_act"] = out["activation_function"]
    if out.get("hidden_act") == "silu":
        out["hidden_act"] = "swiglu"

    # Position embedding inference. ``rotary_emb_fraction > 0`` is the nomic
    # signal that rotary is in use; the canonical ``position_embedding_type``
    # field is absent.
    if "position_embedding_type" not in out:
        rotary_frac = out.get("rotary_emb_fraction", 0)
        if rotary_frac and rotary_frac > 0:
            out["position_embedding_type"] = "rotary"

    # Layer norm epsilon — nomic uses the trailing-``on`` spelling.
    if "layer_norm_eps" not in out and "layer_norm_epsilon" in out:
        out["layer_norm_eps"] = out["layer_norm_epsilon"]

    # RoPE theta — nomic uses ``rotary_emb_base`` or nests in ``rope_parameters``.
    if "rope_theta" not in out:
        if "rotary_emb_base" in out:
            out["rope_theta"] = float(out["rotary_emb_base"])
        elif isinstance(out.get("rope_parameters"), dict):
            rt = out["rope_parameters"].get("rope_theta")
            if rt is not None:
                out["rope_theta"] = float(rt)

    return out


def _validate_config(cfg: dict[str, Any], model_id: str) -> None:
    """Fail loud if the config declares an unsupported variant.

    Operates on a *normalized* config (see ``_normalize_config``) so the
    nomic-bert quirks (``hidden_act='silu'``, implicit rotary) are already
    rewritten to the canonical ``BertConfig`` field names.
    """
    act = cfg.get("hidden_act", "gelu")
    if act not in SUPPORTED_ACTIVATIONS:
        raise UnsupportedModelError(
            f"Cannot load embedding model {model_id!r}. "
            f"Reason: hidden_act={act!r} is not supported. "
            f"Supported activations: {sorted(SUPPORTED_ACTIVATIONS)}. "
            f"Please file an issue at https://github.com/getcoldfire/mlx-openai-server."
        )
    pos = cfg.get("position_embedding_type", "absolute")
    if pos not in SUPPORTED_POSITION_EMBEDDINGS:
        raise UnsupportedModelError(
            f"Cannot load embedding model {model_id!r}. "
            f"Reason: position_embedding_type={pos!r} is not supported. "
            f"Supported: {sorted(SUPPORTED_POSITION_EMBEDDINGS)}."
        )


def _read_pooling_mode(model_dir: Path) -> str:
    """Read the sentence-transformers `1_Pooling/config.json` if present.

    Defaults to "cls" when the sidecar directory is missing (plain BERT
    convention from HuggingFace's reference implementation).
    """
    cfg_path = model_dir / "1_Pooling" / "config.json"
    if not cfg_path.exists():
        return "cls"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("pooling_mode_mean_tokens"):
        return "mean"
    if cfg.get("pooling_mode_cls_token"):
        return "cls"
    if cfg.get("pooling_mode_max_tokens"):
        return "max"
    if cfg.get("pooling_mode_lasttoken") or cfg.get("pooling_mode_last_token"):
        return "last_token"
    return "cls"


def _read_matryoshka_dim(model_dir: Path, cfg: dict[str, Any]) -> int | None:
    """Resolve the matryoshka truncation dim, if declared.

    Two declaration sites supported, checked in this order:
    1. A top-level `matryoshka_dim` field on `config.json`.
    2. A `2_Matryoshka/config.json` sidecar with a `matryoshka_dim` field.
    Returns `None` when neither exists.
    """
    if cfg.get("matryoshka_dim"):
        return int(cfg["matryoshka_dim"])
    mat_cfg = model_dir / "2_Matryoshka" / "config.json"
    if mat_cfg.exists():
        d = json.loads(mat_cfg.read_text())
        val = int(d.get("matryoshka_dim", 0))
        return val or None
    return None


def _remap_hf_to_internal(
    weights: dict[str, Any],
    *,
    is_swiglu: bool = False,
) -> dict[str, Any]:
    """Remap HuggingFace weight key conventions to our encoder's expected keys.

    Real HF repos use a few different conventions; this remap collapses them
    to the layout that ``app.handler.embeddings.encoder.BertModel`` (HF
    BERT-style) expects:

    - Strip a leading ``model.`` prefix (some sentence-transformers repos
      add it).
    - ``.layers.`` (plural) -> ``.layer.`` (HF BERT singular).
    - ``.attn.Wqkv.`` (nomic-bert combined QKV projection) -> split into
      separate ``.attention.self.{query,key,value}.`` matrices of shape
      ``(H, H)`` each. Both ``weight`` and ``bias`` are split if present.
    - ``.attn.out_proj.`` -> ``.attention.output.dense.``
    - ``norm1`` (after attention) -> ``attention.output.LayerNorm.``
    - ``norm2`` (after MLP) -> ``mlp.LayerNorm.`` (SwiGLU branch) or
      ``output.LayerNorm.`` (GeLU branch).
    - Top-level ``emb_ln.`` -> ``embeddings.LayerNorm.``
    - Vanilla BERT MLP: ``.mlp.fc1.`` -> ``.intermediate.dense.``,
      ``.mlp.fc2.`` -> ``.output.dense.``
    - Nomic-bert SwiGLU MLP: ``.mlp.fc11.`` -> ``.mlp.gate.``,
      ``.mlp.fc12.`` -> ``.mlp.up.``, ``.mlp.fc2.`` -> ``.mlp.down.``.
      ``is_swiglu`` selects this branch (resolved from the normalized
      config's ``hidden_act``).
    - Modern decoder-style naming for the SwiGLU triple (used by some
      HF nomic-bert repos that align with the LLaMA MLP convention):
      ``.mlp.gate_proj.`` / ``.mlp.up_proj.`` / ``.mlp.down_proj.`` ->
      ``.mlp.gate.`` / ``.mlp.up.`` / ``.mlp.down.``.
    """
    out: dict[str, Any] = {}
    for raw_key, val in weights.items():
        key = raw_key.removeprefix("model.")

        # Top-level embedding LayerNorm. Nomic stores it as ``emb_ln.*``
        # rather than nesting under ``embeddings.``.
        if key.startswith("emb_ln."):
            out["embeddings.LayerNorm." + key[len("emb_ln.") :]] = val
            continue

        # Nomic-bert: split combined QKV.
        # We support both .attn.Wqkv.weight and .attn.Wqkv.bias.
        if ".attn.Wqkv." in key:
            base = key.replace(".attn.Wqkv.", ".attention.self.")
            # Normalize ``layers`` -> ``layer`` on the path stem too.
            base = base.replace(".layers.", ".layer.")
            # base ends in either ".weight" or ".bias"; we want to inject
            # query/key/value before that suffix.
            third = val.shape[0] // 3
            q_slice = val[:third]
            k_slice = val[third : 2 * third]
            v_slice = val[2 * third :]
            # base e.g. "encoder.layer.0.attention.self.weight"
            # -> we want "encoder.layer.0.attention.self.query.weight" etc.
            if base.endswith(".weight"):
                stem = base[: -len(".weight")]
                out[f"{stem}.query.weight"] = q_slice
                out[f"{stem}.key.weight"] = k_slice
                out[f"{stem}.value.weight"] = v_slice
            elif base.endswith(".bias"):
                stem = base[: -len(".bias")]
                out[f"{stem}.query.bias"] = q_slice
                out[f"{stem}.key.bias"] = k_slice
                out[f"{stem}.value.bias"] = v_slice
            else:
                # Unexpected suffix; pass through unchanged to surface the issue.
                out[key] = val
            continue

        # Plural-vs-singular layer name (must run before substring rewrites
        # below that reference ``.layer.``).
        key = key.replace(".layers.", ".layer.")

        # Nomic-bert per-layer LayerNorms: ``norm1`` after attention,
        # ``norm2`` after MLP. Order matters: replace before the
        # ``mlp.fc*`` rewrites so we don't accidentally double-substitute.
        key = key.replace(".norm1.", ".attention.output.LayerNorm.")
        if is_swiglu:
            key = key.replace(".norm2.", ".mlp.LayerNorm.")
        else:
            key = key.replace(".norm2.", ".output.LayerNorm.")

        # Nomic-bert attention output projection.
        key = key.replace(".attn.out_proj.", ".attention.output.dense.")

        # MLP key rewrites. Order matters: the SwiGLU branch's ``fc2`` is the
        # down projection (different role from vanilla BERT's ``fc2``); the
        # ``fc11`` / ``fc12`` substrings would otherwise overlap with a naive
        # ``fc1`` rewrite, so SwiGLU is matched first.
        if is_swiglu:
            # Nomic-bert SwiGLU: fc11 = gate, fc12 = up, fc2 = down.
            key = key.replace(".mlp.fc11.", ".mlp.gate.")
            key = key.replace(".mlp.fc12.", ".mlp.up.")
            key = key.replace(".mlp.fc2.", ".mlp.down.")
        else:
            # Vanilla BERT MLP pair.
            key = key.replace(".mlp.fc1.", ".intermediate.dense.")
            key = key.replace(".mlp.fc2.", ".output.dense.")

        # Modern decoder-style naming for the SwiGLU triple (used by some
        # HF nomic-bert repos that align with the LLaMA MLP convention).
        key = key.replace(".mlp.gate_proj.", ".mlp.gate.")
        key = key.replace(".mlp.up_proj.", ".mlp.up.")
        key = key.replace(".mlp.down_proj.", ".mlp.down.")
        out[key] = val
    return out


def load_embedding_model(
    model_path: str,
) -> tuple[BertModel, Any, str, int | None]:
    """Load a HuggingFace embedding model into an MLX `BertModel`.

    Parameters
    ----------
    model_path : str
        Either a local directory containing `config.json` + `model.safetensors`
        + tokenizer files (and optional `1_Pooling/`/`2_Matryoshka/` sidecars),
        or a HuggingFace repo ID (e.g. `nomic-ai/nomic-embed-text-v1.5`).
        Repo IDs are downloaded via `huggingface_hub.snapshot_download`.

    Returns
    -------
    model : BertModel
        Initialized MLX model with weights loaded (strict=False).
    tokenizer : AutoTokenizer
        HuggingFace tokenizer for the model.
    pooling_mode : str
        One of "mean", "cls", "max", "last_token".
    matryoshka_dim : int | None
        Truncation dimension if the model declares matryoshka; else None.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        # Treat as a HuggingFace repo ID.
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(model_path))

    cfg_raw = json.loads((model_dir / "config.json").read_text())
    cfg = _normalize_config(cfg_raw)
    _validate_config(cfg, model_path)
    bert_cfg = BertConfig.from_dict(cfg)

    raw_weights = load_file(str(model_dir / "model.safetensors"))
    # First map to our internal key naming (no MLX conversion yet — keep
    # ndarrays so the remap can do shape-aware splits like Wqkv). The remap
    # needs to know whether the model is SwiGLU vs vanilla GeLU so it can
    # route ``mlp.fc2`` to the right destination (``mlp.down`` for SwiGLU,
    # ``output.dense`` for vanilla BERT).
    remapped = _remap_hf_to_internal(raw_weights, is_swiglu=(bert_cfg.hidden_act == "swiglu"))
    mlx_weights = {k: mx.array(v) for k, v in remapped.items()}

    model = BertModel(bert_cfg)
    # strict=False because:
    #   - Some keys may not be present in the safetensors file (e.g. position
    #     embedding when the config requests rotary).
    #   - And vice versa — extra keys (e.g. a `pooler.dense.weight` we don't use).
    model.load_weights(list(mlx_weights.items()), strict=False)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    pooling_mode = _read_pooling_mode(model_dir)
    matryoshka_dim = _read_matryoshka_dim(model_dir, cfg)
    return model, tokenizer, pooling_mode, matryoshka_dim


# Re-export numpy for use in shape-aware operations (Wqkv split).
__all__ = [
    "SUPPORTED_ACTIVATIONS",
    "SUPPORTED_POSITION_EMBEDDINGS",
    "UnsupportedModelError",
    "_normalize_config",
    "_remap_hf_to_internal",
    "load_embedding_model",
]
