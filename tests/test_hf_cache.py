"""Unit tests for HF cache inspection helpers.

Tests monkey-patch huggingface_hub.scan_cache_dir to return a fake
HFCacheInfo built from a tmp_path tree. This keeps tests deterministic
and avoids hitting real HF Hub or the operator's actual cache.

MLX heuristic: a cached repo is MLX-shaped if any of:
  (1) namespace matches mlx-community/* (always treat as MLX)
  (2) config.json declares a "quantization" key
  (3) config.json's "model_type" is in MLX_SUPPORTED_MODEL_TYPES
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
    is_mlx_shaped,
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


def test_list_cached_models_default_filters_to_mlx(fake_cache):
    """mlx_only=True (default) drops the non-MLX bert repo."""
    rows = list_cached_models()
    names = {r.name for r in rows}
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in names
    assert "Qwen/Qwen2.5-7B-Instruct" in names  # model_type "qwen2" passes
    assert "sentence-transformers/all-MiniLM-L6-v2" not in names


def test_list_cached_models_all_includes_non_mlx(fake_cache):
    """mlx_only=False returns everything."""
    rows = list_cached_models(mlx_only=False)
    assert len(rows) == 3


def test_list_cached_models_fields(fake_cache):
    rows = sorted(list_cached_models(mlx_only=False), key=lambda r: r.name)
    qwen = next(r for r in rows if r.name == "Qwen/Qwen2.5-7B-Instruct")
    assert qwen.size_bytes == 4_400_000_000
    assert qwen.last_used.year == 2026 and qwen.last_used.month == 6
    assert qwen.is_mlx is True
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
        ({"model_type": "qwen2"}, "Org/Foo", True),  # supported model_type
        ({"model_type": "phi3"}, "Org/Foo", True),
        ({"model_type": "bert"}, "Org/Foo", False),  # unknown model_type
        ({"model_type": "t5"}, "Org/Foo", False),
        ({}, "Org/Foo", False),  # no model_type, no namespace
        (None, "mlx-community/Foo", True),  # no config but mlx ns
    ],
)
def test_is_mlx_shaped(config, namespace, expected, tmp_path, monkeypatch: pytest.MonkeyPatch):
    # Pin the supported-types set so tests don't depend on what mlx_lm
    # happens to ship on the machine running this suite.
    monkeypatch.setattr(
        "app.utils.hf_cache.MLX_SUPPORTED_MODEL_TYPES",
        frozenset({"llama", "qwen2", "phi3", "mistral", "gemma2"}),
    )
    repo_dir = _make_repo(tmp_path, namespace, config)
    repo = MagicMock()
    repo.repo_id = namespace
    repo.repo_path = repo_dir
    assert is_mlx_shaped(repo) is expected
