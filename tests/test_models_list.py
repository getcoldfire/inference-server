"""Tests for `coldfire-inference-server models list`."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from app.cli import cli
from app.utils.hf_cache import CachedModel


@pytest.fixture
def fake_rows():
    now = datetime.now(tz=UTC)
    return [
        CachedModel(
            name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            size_bytes=712_000_000,
            last_used=now - timedelta(days=3),
            is_mlx=True,
            path=Path("/fake/llama"),
        ),
        CachedModel(
            name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            size_bytes=4_400_000_000,
            last_used=now - timedelta(hours=1),
            is_mlx=True,
            path=Path("/fake/qwen"),
        ),
    ]


def test_list_human_table(fake_rows):
    with (
        patch("app.cli_models.list_cached_models", return_value=fake_rows),
        patch("app.cli_models.serving_model_ids", return_value=set()),
    ):
        result = CliRunner().invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "NAME" in result.output and "SIZE" in result.output and "LAST USED" in result.output
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in result.output
    # Base-10 (SI) byte rendering: 712_000_000 -> "712 MB", 4_400_000_000 -> "4.4 GB"
    assert "712 MB" in result.output
    assert "4.4 GB" in result.output
    # Footer with rollup
    assert "Total:" in result.output
    assert "2 models" in result.output


def test_list_status_column_when_serving(fake_rows):
    """STATUS column shows 'serving' for any model id in the served set."""
    served = {"mlx-community/Qwen2.5-7B-Instruct-4bit"}
    with (
        patch("app.cli_models.list_cached_models", return_value=fake_rows),
        patch("app.cli_models.serving_model_ids", return_value=served),
    ):
        result = CliRunner().invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "serving" in result.output


def test_list_json_shape(fake_rows):
    with (
        patch("app.cli_models.list_cached_models", return_value=fake_rows),
        patch("app.cli_models.serving_model_ids", return_value=set()),
    ):
        result = CliRunner().invoke(cli, ["models", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert len(payload) == 2
    qwen = next(p for p in payload if p["name"].endswith("Qwen2.5-7B-Instruct-4bit"))
    assert qwen["size_bytes"] == 4_400_000_000
    assert qwen["is_mlx"] is True
    assert qwen["serving"] is False
    assert "last_used" in qwen  # ISO-8601 string or None


def test_list_all_flag_passes_through():
    """--all calls list_cached_models(mlx_only=False)."""
    with (
        patch("app.cli_models.list_cached_models", return_value=[]) as m,
        patch("app.cli_models.serving_model_ids", return_value=set()),
    ):
        result = CliRunner().invoke(cli, ["models", "list", "--all"])
    assert result.exit_code == 0
    m.assert_called_once_with(mlx_only=False)


def test_list_empty_cache():
    """Empty cache prints an empty table, exits 0."""
    with (
        patch("app.cli_models.list_cached_models", return_value=[]),
        patch("app.cli_models.serving_model_ids", return_value=set()),
    ):
        result = CliRunner().invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "Total:" in result.output  # rollup line still appears
    assert "0 models" in result.output


def test_list_port_passed_to_probe(fake_rows):
    """--port N flows through to serving_model_ids's port kwarg.
    Single probe per command — not per row."""
    called_with = {}

    def capture(port=8000, timeout=0.5):
        called_with["port"] = port
        return set()

    with (
        patch("app.cli_models.list_cached_models", return_value=fake_rows),
        patch("app.cli_models.serving_model_ids", side_effect=capture),
    ):
        result = CliRunner().invoke(cli, ["models", "list", "--port", "22222"])
    assert result.exit_code == 0
    assert called_with["port"] == 22222
