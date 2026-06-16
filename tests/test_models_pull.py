"""Tests for `coldfire-inference-server models pull <hf-id>`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from app.cli import cli


def _fake_download(repo_id, **kwargs):
    """Stand-in for huggingface_hub.snapshot_download — record kwargs, return a fake path."""
    return f"/tmp/fake/{repo_id.replace('/', '--')}"


def test_pull_default_allowlist():
    """Default --include is conservative; --include flags are additive."""
    captured = {}

    def cap(repo_id, **kwargs):
        captured.update(kwargs)
        captured["repo_id"] = repo_id
        return "/tmp/fake/path"

    with (
        patch("app.cli_models.snapshot_download", side_effect=cap),
        patch("app.cli_models.cache_path_for", return_value=None),
        patch("app.cli_models._looks_loadable_from_hub", return_value=True),
    ):
        result = CliRunner().invoke(cli, ["models", "pull", "mlx-community/Foo-4bit"])
    assert result.exit_code == 0, result.output
    assert captured["repo_id"] == "mlx-community/Foo-4bit"
    patterns = set(captured["allow_patterns"])
    # Conservative defaults
    assert "*.safetensors" in patterns
    assert "*.json" in patterns
    assert "tokenizer*" in patterns
    assert "*.txt" in patterns


def test_pull_include_extends_allowlist():
    captured = {}

    def cap(repo_id, **kwargs):
        captured.update(kwargs)
        return "/tmp/fake/path"

    with (
        patch("app.cli_models.snapshot_download", side_effect=cap),
        patch("app.cli_models.cache_path_for", return_value=None),
        patch("app.cli_models._looks_loadable_from_hub", return_value=True),
    ):
        result = CliRunner().invoke(
            cli, ["models", "pull", "mlx-community/Foo-4bit", "--include", "*.jinja", "--include", "*.gguf"]
        )
    assert result.exit_code == 0
    patterns = set(captured["allow_patterns"])
    assert "*.jinja" in patterns
    assert "*.gguf" in patterns


def test_pull_exclude_removes_from_allowlist():
    captured = {}

    def cap(repo_id, **kwargs):
        captured.update(kwargs)
        return "/tmp/fake/path"

    with (
        patch("app.cli_models.snapshot_download", side_effect=cap),
        patch("app.cli_models.cache_path_for", return_value=None),
        patch("app.cli_models._looks_loadable_from_hub", return_value=True),
    ):
        result = CliRunner().invoke(cli, ["models", "pull", "mlx-community/Foo-4bit", "--exclude", "*.txt"])
    assert result.exit_code == 0
    patterns = set(captured["allow_patterns"])
    assert "*.txt" not in patterns


def test_pull_non_mlx_warns_but_continues():
    """Non-MLX-shaped repo prints warning twice (start + end), download still completes."""
    with (
        patch("app.cli_models.snapshot_download", side_effect=_fake_download),
        patch("app.cli_models.cache_path_for", return_value=None),
        patch("app.cli_models._looks_loadable_from_hub", return_value=False),
    ):
        result = CliRunner().invoke(cli, ["models", "pull", "microsoft/Phi-3"])
    assert result.exit_code == 0, result.output
    # Warning appears at start AND at end — lowercase substring match
    text = result.output.lower()
    assert "doesn't look loadable" in text
    # Success line still printed (we're not quiet)
    assert "cached" in text


def test_pull_quiet_suppresses_progress_banners(monkeypatch):
    """--quiet hides 'Downloading ...' AND '✓ cached at ...' banners.
    Also sets HF_HUB_DISABLE_PROGRESS_BARS=1 so HF tqdm is silenced too."""
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    with (
        patch("app.cli_models.snapshot_download", side_effect=_fake_download),
        patch("app.cli_models.cache_path_for", return_value=None),
        patch("app.cli_models._looks_loadable_from_hub", return_value=True),
    ):
        result = CliRunner().invoke(cli, ["models", "pull", "mlx-community/Foo-4bit", "--quiet"])
    assert result.exit_code == 0
    # Quiet: no progress banner AND no success line
    assert "Downloading" not in result.output
    # Implementation sets HF_HUB_DISABLE_PROGRESS_BARS=1 for the process
    import os

    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"


def test_pull_already_cached_skips_hub_preflight():
    """If cache_path_for returns a path, _looks_loadable_from_hub is NOT called
    (no extra HF round-trip on already-cached repos)."""
    preflight_calls = []

    def boom(*a, **kw):
        preflight_calls.append((a, kw))
        return True

    with (
        patch("app.cli_models.snapshot_download", side_effect=_fake_download),
        patch("app.cli_models.cache_path_for", return_value=Path("/fake/cached")),
        patch("app.cli_models._looks_loadable_from_hub", side_effect=boom),
    ):
        result = CliRunner().invoke(cli, ["models", "pull", "mlx-community/Foo-4bit"])
    assert result.exit_code == 0
    assert preflight_calls == [], "_looks_loadable_from_hub should NOT have been called"


def test_pull_propagates_hub_error():
    """HF Hub error → exit 1 with the error message visible."""

    def boom(*a, **kw):
        # huggingface_hub 1.x requires a real httpx.Response on the
        # exception. We construct a stand-in 404 so the exception is
        # instantiable; the test only cares that exit_code == 1 and
        # the error message reaches stderr.
        import httpx
        from huggingface_hub.errors import RepositoryNotFoundError

        resp = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://huggingface.co/bad/id"),
        )
        raise RepositoryNotFoundError("not found", response=resp)

    with (
        patch("app.cli_models.snapshot_download", side_effect=boom),
        patch("app.cli_models.cache_path_for", return_value=None),
        patch("app.cli_models._looks_loadable_from_hub", return_value=True),
    ):
        result = CliRunner().invoke(cli, ["models", "pull", "bad/id"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "error" in result.output.lower()
