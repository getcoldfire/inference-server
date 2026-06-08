"""Slow smoke: pull then rm the same model, assert dir gone."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from app.cli import cli

SMALL_MODEL = "mlx-community/SmolLM2-360M-Instruct-6bit"


@pytest.mark.slow
def test_pull_then_rm_real_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # See test_pull_real_hub for why env vars + both module constants are
    # patched. rm specifically depends on the _cache_manager patch since
    # cache_path_for() routes through scan_cache_dir().
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.utils._cache_manager.HF_HUB_CACHE", str(tmp_path))

    pull_result = CliRunner().invoke(cli, ["models", "pull", SMALL_MODEL, "--quiet"])
    assert pull_result.exit_code == 0, pull_result.output

    cache_dir = tmp_path / f"models--{SMALL_MODEL.replace('/', '--')}"
    assert cache_dir.is_dir()

    rm_result = CliRunner().invoke(cli, ["models", "rm", SMALL_MODEL])
    assert rm_result.exit_code == 0, rm_result.output
    assert "Removed" in rm_result.output

    assert not cache_dir.exists(), f"cache dir not removed: {cache_dir}"
