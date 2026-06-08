"""Smoke test: the `models` subgroup is discoverable via `coldfire-mlx-server models --help`."""

from __future__ import annotations

from click.testing import CliRunner

from app.cli import cli


def test_models_subgroup_help_succeeds():
    """`coldfire-mlx-server models --help` returns 0 and lists the subcommand."""
    result = CliRunner().invoke(cli, ["models", "--help"])
    assert result.exit_code == 0, result.output
    assert "Manage the local HuggingFace cache" in result.output
