"""Tests for Bug 1 (stderr-only logging) and Bug 2 (zero-models config).

Bug 1: app/server.py was writing logs to a relative path ``logs/app.log``.
Under macOS launchd the cwd is ``/``, which makes the path ``/logs/app.log``
on a read-only filesystem.  Fix: drop the file handler entirely; stderr is
the only sink.

Bug 2: app/config.py rejected ``models: []`` with a ValueError.  The
/admin/models/load endpoint makes empty-then-hot-add a first-class workflow,
so an empty list must now boot cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Bug 2 — zero-models config must be accepted
# ---------------------------------------------------------------------------


def test_config_accepts_empty_models_list(tmp_path: Path) -> None:
    """load_config_from_yaml must not raise when ``models: []``."""
    from app.config import load_config_from_yaml

    config_path = tmp_path / "empty.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")

    config = load_config_from_yaml(str(config_path))
    assert config.models == [], (
        "Expected an empty models list; got: " + repr(config.models)
    )


def test_config_accepts_empty_models_list_with_server_section(tmp_path: Path) -> None:
    """load_config_from_yaml must not raise when models is empty but server section is present."""
    from app.config import load_config_from_yaml

    config_path = tmp_path / "empty_with_server.yaml"
    config_path.write_text(
        "server:\n"
        "  host: 127.0.0.1\n"
        "  port: 18435\n"
        "models: []\n",
        encoding="utf-8",
    )

    config = load_config_from_yaml(str(config_path))
    assert config.models == []
    assert config.host == "127.0.0.1"
    assert config.port == 18435


def test_multimodel_config_empty_models_list_is_valid() -> None:
    """MultiModelServerConfig with models=[] must construct without error."""
    from app.config import MultiModelServerConfig

    cfg = MultiModelServerConfig(models=[])
    assert cfg.models == []


# ---------------------------------------------------------------------------
# Bug 1 — configure_logging must not create any files
# ---------------------------------------------------------------------------


def test_logging_no_file_handler_by_default(tmp_path: Path, monkeypatch: Any) -> None:
    """configure_logging() must not create any files in the working directory.

    Change to tmp_path, call configure_logging with no args (the default
    path), then verify the directory is still empty.  Loguru's stderr handler
    is expected to remain active — we capture stderr instead of asserting
    silence.
    """
    monkeypatch.chdir(tmp_path)

    from app.server import configure_logging

    configure_logging()  # default args — previously created logs/app.log

    created = list(tmp_path.rglob("*"))
    assert created == [], (
        "configure_logging() must not create any files; found: "
        + ", ".join(str(p) for p in created)
    )


def test_logging_no_file_handler_with_explicit_log_file_arg(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Passing log_file= must also be a no-op for file creation.

    Since we dropped file-handler support entirely, passing log_file should
    not create anything either.
    """
    monkeypatch.chdir(tmp_path)

    from app.server import configure_logging

    # Even if a caller accidentally passes a path, no file should appear
    configure_logging(log_file=str(tmp_path / "should_not_exist.log"))

    log_files = list(tmp_path.glob("*.log"))
    assert log_files == [], (
        "configure_logging() must not create log files; found: "
        + ", ".join(str(p) for p in log_files)
    )


def test_logging_configure_logging_signature_still_accepts_old_kwargs() -> None:
    """configure_logging must still accept log_file, no_log_file, log_level kwargs.

    Callers that pass these args (e.g. setup_server) must not get a TypeError.
    The args are accepted but no file is created.
    """
    from app.server import configure_logging

    # Must not raise
    configure_logging(log_file=None, no_log_file=True, log_level="WARNING")
    configure_logging(log_file="/tmp/ignored.log", no_log_file=False, log_level="DEBUG")


# ---------------------------------------------------------------------------
# Regression: MultiModelServerConfig(models=[]) boots server app without crash
# ---------------------------------------------------------------------------


def test_setup_server_with_empty_models_does_not_raise() -> None:
    """setup_server must succeed when models is empty (zero-models boot)."""
    from app import server as server_module
    from app.config import MultiModelServerConfig

    cfg = MultiModelServerConfig(models=[], host="127.0.0.1", port=0)
    # setup_server should not raise
    server_module.setup_server(cfg)
    assert server_module.app is not None
