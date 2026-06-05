"""Coverage for the cli-v2 launch-flag contract (Task 20).

Verifies the surface area required by ``docs/superpowers/specs/
2026-06-03-mlx-openai-server-fork-spec.md`` §4 — the daemon launches
``coldfire-mlx-server`` (this CLI) with exactly the documented flags
and expects them to behave as specified:

  --host (default 127.0.0.1)
  --port (required, integer)
  --model (required, repo ID or local path)
  --max-concurrency (alias of --decode-concurrency, sets
                     batch_completion_size)
  --queue-size
  --idle-unload-seconds (>0 flips ``on_demand`` and sets the idle timeout)
  --log-level (info|debug|warn|error, case-insensitive)
  --licenses (prints attributions and exits)
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest
from click.testing import CliRunner


def _load_cli_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Re-import ``app.cli`` with a lightweight ``app.main`` stub."""
    fake_main = types.ModuleType("app.main")

    async def _placeholder(_config: Any) -> None:
        return None

    fake_main.start = _placeholder
    fake_main.start_multi = _placeholder
    monkeypatch.setitem(sys.modules, "app.main", fake_main)
    monkeypatch.delitem(sys.modules, "app.cli", raising=False)
    cli_module = importlib.import_module("app.cli")
    return importlib.reload(cli_module)


def _run_launch(cli_module: Any, *args: str) -> tuple[Any, dict[str, Any]]:
    """Invoke ``cli launch`` with stubbed ``start_multi`` capturing config."""
    captured: dict[str, Any] = {}

    async def _capture(config: Any) -> None:
        captured["config"] = config

    runner = CliRunner()
    cli_module.start_multi = _capture  # type: ignore[attr-defined]
    result = runner.invoke(cli_module.cli, ["launch", *args])
    return result, captured


def test_model_flag_is_accepted_as_alias_for_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = _load_cli_module(monkeypatch)
    result, captured = _run_launch(cli_module, "--model", "dummy-model")
    assert result.exit_code == 0, result.output
    assert captured["config"].models[0].model_path == "dummy-model"


def test_host_default_is_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """cli-v2 spec §4: default host is 127.0.0.1, not 0.0.0.0."""
    cli_module = _load_cli_module(monkeypatch)
    result, captured = _run_launch(cli_module, "--model", "dummy-model")
    assert result.exit_code == 0, result.output
    assert captured["config"].host == "127.0.0.1"


def test_max_concurrency_wires_to_batch_completion_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = _load_cli_module(monkeypatch)
    result, captured = _run_launch(cli_module, "--model", "dummy", "--max-concurrency", "9")
    assert result.exit_code == 0, result.output
    assert captured["config"].models[0].batch_completion_size == 9


def test_idle_unload_seconds_flips_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = _load_cli_module(monkeypatch)
    result, captured = _run_launch(cli_module, "--model", "dummy", "--idle-unload-seconds", "45")
    assert result.exit_code == 0, result.output
    entry = captured["config"].models[0]
    assert entry.on_demand is True
    assert entry.on_demand_idle_timeout == 45


def test_idle_unload_seconds_zero_keeps_model_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = _load_cli_module(monkeypatch)
    result, captured = _run_launch(cli_module, "--model", "dummy")
    assert result.exit_code == 0, result.output
    assert captured["config"].models[0].on_demand is False


@pytest.mark.parametrize(
    ("flag_value", "expected"),
    [
        ("info", "INFO"),
        ("DEBUG", "DEBUG"),
        ("warn", "WARNING"),
        ("WARN", "WARNING"),
        ("error", "ERROR"),
    ],
)
def test_log_level_accepts_lowercase_and_warn_alias(
    monkeypatch: pytest.MonkeyPatch, flag_value: str, expected: str
) -> None:
    cli_module = _load_cli_module(monkeypatch)
    result, captured = _run_launch(cli_module, "--model", "dummy", "--log-level", flag_value)
    assert result.exit_code == 0, result.output
    assert captured["config"].log_level == expected


def test_licenses_flag_prints_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--licenses`` on the top-level group prints and exits without booting."""
    cli_module = _load_cli_module(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["--licenses"])
    assert result.exit_code == 0, result.output
    assert ("NOTICES" in result.output) or ("third-party" in result.output)


def test_missing_model_raises_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = _load_cli_module(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["launch"])
    assert result.exit_code != 0
    assert "model" in result.output.lower()
