"""Slow smoke: real HF Hub download of a tiny MLX model.

Skipped unless `pytest -m slow` is passed. Uses HF_HOME pointed at a
tmp_path so the operator's real cache is untouched.

Target model: mlx-community/SmolLM2-360M-Instruct-6bit (~270 MB). Small
enough to download in <60s on a residential link, large enough to
exercise the multi-file snapshot path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from app.cli import cli

SMALL_MODEL = "mlx-community/SmolLM2-360M-Instruct-6bit"


@pytest.mark.slow
def test_pull_real_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # huggingface_hub resolves HF_HUB_CACHE into module constants at IMPORT
    # time, so setenv() after import is ignored. We patch the two binding
    # sites used downstream: `constants.HF_HUB_CACHE` (attribute access by
    # snapshot_download) and `utils._cache_manager.HF_HUB_CACHE` (imported
    # by name and used by scan_cache_dir). Env vars are also set for any
    # subprocess path. HF_HUB_CACHE points DIRECTLY at the cache root (not
    # a hub/ subdir) so cached repos land at tmp_path/models--... — if
    # dropped, repos land at tmp_path/hub/models--... and the path
    # assertion silently breaks.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.utils._cache_manager.HF_HUB_CACHE", str(tmp_path))

    result = CliRunner().invoke(cli, ["models", "pull", SMALL_MODEL, "--quiet"])
    assert result.exit_code == 0, result.output

    cache_dir = tmp_path / f"models--{SMALL_MODEL.replace('/', '--')}"
    assert cache_dir.is_dir(), f"cache dir not created: {cache_dir}"

    # Confirm at least one safetensors file landed.
    safetensors = list(cache_dir.rglob("*.safetensors"))
    assert safetensors, "no .safetensors files downloaded"
    # And the conservative allowlist kept other formats out:
    bins = list(cache_dir.rglob("*.bin"))
    assert not bins, f"unexpected PyTorch .bin files: {bins}"
