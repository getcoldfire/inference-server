"""Unit tests for HF cache inspection helpers.

Tests monkey-patch huggingface_hub.scan_cache_dir to return a fake
HFCacheInfo built from a tmp_path tree. This keeps tests deterministic
and avoids hitting real HF Hub or the operator's actual cache.

Loadability heuristic (matches is_loadable in app/utils/hf_cache.py):
  (1) namespace matches mlx-community/*  (MLX-LM)
  (2) repo ID ends in -GGUF (case-insensitive)  (llama-cpp)
  (3) snapshot contains a .gguf file  (llama-cpp)
  (4) config.json has a "quantization" key  (MLX-LM)
  (5) config.json model_type in MLX_LM_MODEL_TYPES  (MLX-LM)
  (6) config.json model_type in BERT_ENCODER_MODEL_TYPES  (BERT)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.utils.hf_cache import (
    cache_path_for,
    is_loadable,
    list_cached_models,
)


def _make_repo(root: Path, repo_id: str, config: dict[str, Any] | None, has_safetensors: bool = True) -> Path:
    """Build a fake HF cache repo dir at root/models--<org>--<repo>/."""
    safe_name = "models--" + repo_id.replace("/", "--")
    repo_dir = root / safe_name
    snap_dir = repo_dir / "snapshots" / "abc123"
    snap_dir.mkdir(parents=True)
    if config is not None:
        (snap_dir / "config.json").write_text(json.dumps(config))
    if has_safetensors:
        (snap_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    return repo_dir


def _stub_scan(repos: list[dict[str, Any]]):
    """Build a fake HFCacheInfo from a list of {repo_id, size, mtime, path}."""
    info = MagicMock()
    repo_objs = []
    for r in repos:
        repo = MagicMock()
        repo.repo_id = r["repo_id"]
        repo.repo_type = "model"
        repo.size_on_disk = r["size"]
        repo.last_accessed = r["mtime"]
        repo.repo_path = r["path"]
        repo_objs.append(repo)
    info.repos = repo_objs
    return info


@pytest.fixture
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fake HF cache with 3 repos: 1 MLX-community, 1 generic safetensors
    with mlx-compatible model_type, 1 generic non-MLX. Patch scan_cache_dir."""
    _make_repo(
        tmp_path,
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        {"model_type": "llama", "quantization": {"group_size": 64}},
    )
    _make_repo(tmp_path, "Qwen/Qwen2.5-7B-Instruct", {"model_type": "qwen2"})  # MLX-compat model_type, no namespace
    _make_repo(tmp_path, "sentence-transformers/all-MiniLM-L6-v2", {"model_type": "bert"})  # non-MLX

    repos = [
        {
            "repo_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "size": 712_000_000,
            "mtime": datetime(2026, 6, 5, 14, 0, tzinfo=UTC).timestamp(),
            "path": tmp_path / "models--mlx-community--Llama-3.2-1B-Instruct-4bit",
        },
        {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "size": 4_400_000_000,
            "mtime": datetime(2026, 6, 8, 9, 0, tzinfo=UTC).timestamp(),
            "path": tmp_path / "models--Qwen--Qwen2.5-7B-Instruct",
        },
        {
            "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
            "size": 90_000_000,
            "mtime": datetime(2026, 5, 1, 12, 0, tzinfo=UTC).timestamp(),
            "path": tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2",
        },
    ]
    monkeypatch.setattr(
        "app.utils.hf_cache.scan_cache_dir",
        lambda *a, **kw: _stub_scan(repos),
    )
    return tmp_path


def test_list_cached_models_default_filters_to_loadable(fake_cache):
    """loadable_only=True (default) keeps every handler-loadable repo.

    The fixture's three repos all happen to be loadable now: MLX namespace,
    qwen2 model_type (MLX-LM), and bert model_type (BERT encoder). To prove
    the filter still does something, the fixture is extended (below) by
    individual tests when they need a genuinely-non-loadable repo.
    """
    rows = list_cached_models()
    names = {r.name for r in rows}
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in names
    assert "Qwen/Qwen2.5-7B-Instruct" in names  # model_type qwen2 (MLX-LM)
    # BERT model_type now passes (covered by the BERT encoder handler).
    assert "sentence-transformers/all-MiniLM-L6-v2" in names


def test_list_cached_models_all_includes_non_loadable(fake_cache):
    """loadable_only=False returns everything."""
    rows = list_cached_models(loadable_only=False)
    assert len(rows) == 3


def test_list_cached_models_fields(fake_cache):
    rows = sorted(list_cached_models(loadable_only=False), key=lambda r: r.name)
    qwen = next(r for r in rows if r.name == "Qwen/Qwen2.5-7B-Instruct")
    assert qwen.size_bytes == 4_400_000_000
    assert qwen.last_used.year == 2026 and qwen.last_used.month == 6
    assert qwen.is_loadable is True
    assert qwen.path.name == "models--Qwen--Qwen2.5-7B-Instruct"


def test_cache_path_for_existing(fake_cache, tmp_path):
    p = cache_path_for("mlx-community/Llama-3.2-1B-Instruct-4bit")
    assert p is not None
    assert p.name == "models--mlx-community--Llama-3.2-1B-Instruct-4bit"


def test_cache_path_for_missing(fake_cache):
    assert cache_path_for("nonexistent/model") is None


@pytest.mark.parametrize(
    "config,namespace,expected",
    [
        ({"model_type": "llama"}, "mlx-community/Foo", True),  # namespace forces MLX
        ({"model_type": "llama", "quantization": {}}, "Org/Foo", True),  # quantization key
        ({"model_type": "qwen2"}, "Org/Foo", True),  # MLX-LM model_type
        ({"model_type": "phi3"}, "Org/Foo", True),
        # BERT-encoder model_types are loadable since v0.4.3.
        ({"model_type": "bert"}, "Org/Foo", True),
        ({"model_type": "nomic_bert"}, "nomic-ai/nomic-embed-text-v1.5", True),
        ({"model_type": "mpnet"}, "Org/Foo", True),
        ({"model_type": "roberta"}, "Org/Foo", True),
        # Architectures no handler supports remain non-loadable.
        ({"model_type": "t5"}, "Org/Foo", False),
        ({"model_type": "vit"}, "Org/Foo", False),
        ({}, "Org/Foo", False),  # no model_type, no namespace
        (None, "mlx-community/Foo", True),  # no config but mlx ns
        # GGUF / llama-cpp: namespace suffix wins regardless of config.
        (None, "nomic-ai/nomic-embed-text-v1.5-GGUF", True),
        (None, "Qwen/Qwen3-Embedding-4B-gguf", True),  # case-insensitive suffix
        ({"model_type": "t5"}, "Some/Org-GGUF", True),  # suffix overrides non-loadable model_type
    ],
)
def test_is_loadable(config, namespace, expected, tmp_path, monkeypatch: pytest.MonkeyPatch):
    # Pin the MLX-LM set so tests don't depend on what mlx_lm happens to
    # ship on the runner. BERT_ENCODER_MODEL_TYPES is hand-curated and
    # doesn't need pinning.
    monkeypatch.setattr(
        "app.utils.hf_cache.MLX_LM_MODEL_TYPES",
        frozenset({"llama", "qwen2", "phi3", "mistral", "gemma2"}),
    )
    repo_dir = _make_repo(tmp_path, namespace, config)
    repo = MagicMock()
    repo.repo_id = namespace
    repo.repo_path = repo_dir
    assert is_loadable(repo) is expected


def test_is_loadable_detects_gguf_in_snapshot_without_suffix(tmp_path):
    """A non-suffixed repo whose snapshot ships a .gguf file should still
    be flagged loadable — covers manual pulls into oddly-named repos."""
    # Build a repo whose ID doesn't end in -GGUF and whose config is t5
    # (no handler covers it), so neither the namespace nor model_type
    # rule applies. Drop a .gguf file into the snapshot and expect the
    # .gguf scan to flip the verdict.
    namespace = "operator/local-conversion"
    repo_dir = _make_repo(tmp_path, namespace, {"model_type": "t5"})
    snap = next((repo_dir / "snapshots").iterdir())
    (snap / "weights.q4_K_M.gguf").write_bytes(b"\x00" * 16)

    repo = MagicMock()
    repo.repo_id = namespace
    repo.repo_path = repo_dir
    assert is_loadable(repo) is True
