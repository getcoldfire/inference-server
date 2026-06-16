"""HuggingFace cache inspection helpers used by the `models` CLI.

All three exported helpers operate on the local cache (no network):

  list_cached_models(loadable_only=True)  -> list[CachedModel]
  cache_path_for(hf_id)                   -> Path | None
  is_loadable(repo_info)                  -> bool

Built on huggingface_hub.scan_cache_dir() so we don't hand-roll the
`models--<org>--<repo>` ↔ `<hf-id>` translation. The loadability
heuristic decides what counts as servable by any of our handlers for
the default `list` filter:

  1. Namespace `mlx-community/*` (MLX-LM handler).
  2. Repo ID ending in `-GGUF` (case-insensitive) — llama-cpp handler.
  3. Any cached snapshot contains a `*.gguf` file — llama-cpp handler.
  4. config.json with a "quantization" key — MLX-LM handler.
  5. config.json `model_type` in MLX_LM_MODEL_TYPES — MLX-LM handler.
  6. config.json `model_type` in BERT_ENCODER_MODEL_TYPES — BERT handler.

`--all` on the `list` command bypasses the heuristic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE


def _ensure_hub_dir() -> None:
    """Pre-create ~/.cache/huggingface/hub (or whatever HF_HUB_CACHE points
    to) before any scan_cache_dir() call.

    On a fresh machine where the HF cache has never been written to,
    scan_cache_dir() raises CacheNotFound — which used to surface as
    `coldfire-ctl models pull` failing on the first invocation of a new
    node, blocking the install flow. The directory is trivially benign
    to create; doing it lazily here lets list/pull/rm just work without
    forcing the operator to mkdir up front.
    """
    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)

# Files in mlx_lm.models that ship utility code (no `class Model`) — exclude
# from the discovered model_type set so we don't false-positive a repo whose
# config.json happened to declare e.g. `"model_type": "cache"`. Verified
# against the installed mlx_lm via `grep -L "class Model" mlx_lm/models/*.py`.
_MLX_MODELS_NONMODEL_FILES: frozenset[str] = frozenset(
    {
        "activations",
        "bitlinear_layers",
        "cache",
        "gated_delta",
        "mla",
        "pipeline",
        "rope_utils",
        "ssm",
        "switch_layers",
    }
)


def _discover_mlx_model_types() -> frozenset[str]:
    """Discover candidate `model_type` strings by scanning the installed
    `mlx_lm.models` package for `*.py` files. Excludes dunder/private
    files, `base.py`, and known utility modules
    (`_MLX_MODELS_NONMODEL_FILES`).

    This avoids hand-typing a list that drifts as upstream mlx_lm adds
    architectures (qwen3, gemma3, dbrx, etc.). On import failure (mlx_lm
    not installed, models dir missing) returns an empty set — the rest
    of the MLX heuristic (namespace + quantization key) still works as
    fallbacks.

    Note: this is a filename-based heuristic. If mlx_lm later adds a
    non-model utility file with a normal-looking name, update
    `_MLX_MODELS_NONMODEL_FILES`. Verification:
    `grep -L "class Model" mlx_lm/models/*.py`.
    """
    try:
        from mlx_lm import models as _mlx_models

        models_dir = Path(_mlx_models.__file__).parent
        return frozenset(
            f.stem
            for f in models_dir.glob("*.py")
            if not f.stem.startswith("_") and f.stem != "base" and f.stem not in _MLX_MODELS_NONMODEL_FILES
        )
    except (ImportError, OSError, AttributeError):
        return frozenset()


MLX_LM_MODEL_TYPES: frozenset[str] = _discover_mlx_model_types()

# BERT-encoder handler accepts BERT-family architectures regardless of
# explicit model_type, but cache-scan time we only have config.json. The
# common encoder model_type strings that the handler can load with its
# vanilla / RoPE / SwiGLU variants. Hand-curated rather than discovered
# because the handler is shape-driven (loader.py: SUPPORTED_ACTIVATIONS,
# SUPPORTED_POSITION_EMBEDDINGS), so there's no registry to scan.
# Extending this set is cheap if a new encoder family lands.
BERT_ENCODER_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "bert",
        "nomic_bert",
        "mpnet",
        "roberta",
        "xlm-roberta",
    }
)


@dataclass
class CachedModel:
    """A model present in the local HF cache."""

    name: str  # HF repo ID e.g. "mlx-community/Foo-4bit"
    size_bytes: int  # Total bytes on disk (HF's size_on_disk)
    last_used: datetime | None  # huggingface_hub's last_accessed (lib-internal,
    # NOT filesystem atime). None if not tracked yet.
    is_loadable: bool  # Passes the loadability heuristic (any handler can load it)
    path: Path  # Cache directory path


def list_cached_models(loadable_only: bool = True) -> list[CachedModel]:
    """Return a list of CachedModel for every repo in the HF cache.

    When loadable_only is True (default), filters to repos that pass
    is_loadable(). Set False to see every cached repo regardless.
    """
    _ensure_hub_dir()
    info = scan_cache_dir()
    out: list[CachedModel] = []
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        loadable = is_loadable(repo)
        if loadable_only and not loadable:
            continue
        last_used: datetime | None = None
        if repo.last_accessed is not None:
            last_used = datetime.fromtimestamp(repo.last_accessed, tz=UTC)
        out.append(
            CachedModel(
                name=repo.repo_id,
                size_bytes=int(repo.size_on_disk),
                last_used=last_used,
                is_loadable=loadable,
                path=Path(repo.repo_path),
            )
        )
    return out


def cache_path_for(hf_id: str) -> Path | None:
    """Return the cache directory for hf_id, or None if not cached.

    Uses scan_cache_dir's repo_id field to avoid hand-rolling the name
    translation (HF cache uses `models--<org>--<repo>`).
    """
    _ensure_hub_dir()
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_type == "model" and repo.repo_id == hf_id:
            return Path(repo.repo_path)
    return None


def is_loadable(repo: Any) -> bool:
    """Whether a cached repo can be loaded by any of this server's handlers
    (MLX-LM, BERT-encoder, or llama-cpp).

    Signals (any one is sufficient):
      1. Namespace matches ``mlx-community/*`` — MLX-LM handler.
      2. Repo ID ends with ``-GGUF`` (case-insensitive) — llama-cpp;
         matches the cli-v2 classifier in ``mlxBackendModelType``.
      3. Any cached snapshot contains a ``*.gguf`` file — llama-cpp,
         catches GGUFs in non-suffixed repos (rare but legitimate).
      4. config.json contains a ``"quantization"`` key — MLX-LM.
      5. config.json ``model_type`` is in MLX_LM_MODEL_TYPES — MLX-LM.
      6. config.json ``model_type`` is in BERT_ENCODER_MODEL_TYPES —
         BERT-encoder handler (vanilla / RoPE / SwiGLU variants).
    """
    repo_id = getattr(repo, "repo_id", "") or ""
    if repo_id.startswith("mlx-community/"):
        return True
    if repo_id.lower().endswith("-gguf"):
        return True
    if _snapshot_has_gguf(repo):
        return True

    config = _read_config(repo)
    if not config:
        return False
    if "quantization" in config:
        return True
    model_type = config.get("model_type", "")
    return model_type in MLX_LM_MODEL_TYPES or model_type in BERT_ENCODER_MODEL_TYPES


def _snapshot_has_gguf(repo: Any) -> bool:
    """Return True iff the repo's most-recent snapshot contains any
    ``*.gguf`` file. Used by is_loadable to flag GGUF repos that
    don't follow the ``-GGUF`` suffix convention.
    """
    repo_path = getattr(repo, "repo_path", None)
    if repo_path is None:
        return False
    snapshots_dir = Path(repo_path) / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    try:
        snaps = sorted(
            snapshots_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return False
    for snap in snaps:
        try:
            for entry in snap.iterdir():
                if entry.name.lower().endswith(".gguf"):
                    return True
        except OSError:
            continue
        return False  # Only inspect the most-recent snapshot.
    return False


def _read_config(repo: Any) -> dict[str, Any] | None:
    """Read config.json from the repo's most-recent snapshot, if any.

    Sorts snapshots by mtime descending so a freshly-pulled revision
    wins over a stale one (rather than the alphabetically-first SHA).
    """
    repo_path = getattr(repo, "repo_path", None)
    if repo_path is None:
        return None
    snapshots_dir = Path(repo_path) / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    # Most recent snapshot first.
    try:
        snaps = sorted(
            snapshots_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for snap in snaps:
        cfg_path = snap / "config.json"
        if cfg_path.is_file():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                return None
    return None
