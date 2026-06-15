"""Tests for `coldfire-inference-server models rm <hf-id>`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from app.cli import cli


@pytest.fixture
def fake_cache_dir(tmp_path):
    """Build a fake cache dir with one model on disk (so shutil.rmtree
    has something real to delete)."""
    d = tmp_path / "models--mlx-community--Foo-4bit"
    snap = d / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\x00" * 1024)
    return d


def test_rm_happy_path(fake_cache_dir):
    """Cache dir exists, no server running → delete + success message."""
    with (
        patch("app.cli_models.cache_path_for", return_value=fake_cache_dir),
        patch("app.cli_models.is_model_serving", return_value=False),
        patch("app.cli_models._lookup_size_bytes", return_value=1024),
    ):
        result = CliRunner().invoke(cli, ["models", "rm", "mlx-community/Foo-4bit"])
    assert result.exit_code == 0, result.output
    assert "Removed mlx-community/Foo-4bit" in result.output
    assert not fake_cache_dir.exists()


def test_rm_not_cached_exits_1():
    with patch("app.cli_models.cache_path_for", return_value=None):
        result = CliRunner().invoke(cli, ["models", "rm", "not/cached"])
    assert result.exit_code == 1
    assert "not in the local cache" in result.output


def test_rm_refuses_when_serving(fake_cache_dir):
    """Default: refuses if a running fork advertises the model."""
    with (
        patch("app.cli_models.cache_path_for", return_value=fake_cache_dir),
        patch("app.cli_models.is_model_serving", return_value=True),
    ):
        result = CliRunner().invoke(cli, ["models", "rm", "mlx-community/Foo-4bit"])
    assert result.exit_code == 1
    assert "currently being served" in result.output or "serving" in result.output
    assert "--force" in result.output  # actionable hint
    assert fake_cache_dir.exists(), "cache dir must NOT be deleted on refusal"


def test_rm_force_bypasses_safety(fake_cache_dir):
    """--force deletes the cache dir even if the model is serving."""
    with (
        patch("app.cli_models.cache_path_for", return_value=fake_cache_dir),
        patch("app.cli_models.is_model_serving", return_value=True),
        patch("app.cli_models._lookup_size_bytes", return_value=1024),
    ):
        result = CliRunner().invoke(cli, ["models", "rm", "mlx-community/Foo-4bit", "--force"])
    assert result.exit_code == 0
    assert "Removed" in result.output
    assert not fake_cache_dir.exists()


def test_rm_port_passed_to_probe(fake_cache_dir):
    """--port flows through to is_model_serving."""
    called = {}

    def cap(hf_id, port=8000, timeout=0.5):
        called["port"] = port
        return False

    with (
        patch("app.cli_models.cache_path_for", return_value=fake_cache_dir),
        patch("app.cli_models.is_model_serving", side_effect=cap),
        patch("app.cli_models._lookup_size_bytes", return_value=0),
    ):
        result = CliRunner().invoke(cli, ["models", "rm", "mlx-community/Foo-4bit", "--port", "22222"])
    assert result.exit_code == 0
    assert called["port"] == 22222
