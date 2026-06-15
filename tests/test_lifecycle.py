"""SIGTERM clean-exit contract tests.

Two layers of coverage:

* ``test_arm_shutdown_watchdog_sets_should_exit``: a fast unit test that
  the signal handler does what its docstring promises (no server boot,
  no model load).
* ``test_sigterm_exits_within_5s``: an integration test that boots the
  real server with the local tiny_bert embeddings fixture, sends
  ``SIGTERM`` to the parent, and asserts a clean exit (code 0) inside
  the 5 s budget specified by the cli-v2 contract.

The integration test is marked ``integration`` and is skipped off
Apple Silicon (MLX requires Metal).
"""

from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from app.main import (
    SHUTDOWN_DEADLINE_SECONDS,
    _arm_shutdown_watchdog,
)

# ---------------------------------------------------------------------------
# Unit test: signal handler behavior (no server boot, runs everywhere)
# ---------------------------------------------------------------------------


class _FakeServer:
    """Stand-in for ``uvicorn.Server`` exposing only ``should_exit``."""

    def __init__(self) -> None:
        self.should_exit: bool = False


def test_arm_shutdown_watchdog_sets_should_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """First SIGTERM must flip ``server.should_exit`` and arm the deadline."""
    installed: dict[int, object] = {}

    def fake_signal(signum: int, handler: object) -> None:
        installed[signum] = handler

    monkeypatch.setattr(signal, "signal", fake_signal)

    server = _FakeServer()
    triggered = _arm_shutdown_watchdog(server)  # type: ignore[arg-type]

    assert signal.SIGTERM in installed
    assert signal.SIGINT in installed
    assert not triggered.is_set()
    assert server.should_exit is False

    # Simulate SIGTERM delivery.
    handler = installed[signal.SIGTERM]
    handler(signal.SIGTERM, None)  # type: ignore[operator]

    assert triggered.is_set()
    assert server.should_exit is True


def test_shutdown_deadline_is_five_seconds() -> None:
    """Contract constant: cli-v2 expects SIGTERM → exit 0 inside 5 s."""
    assert SHUTDOWN_DEADLINE_SECONDS == 5.0


# ---------------------------------------------------------------------------
# Integration test: spawn real server, SIGTERM, time the exit.
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BERT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "tiny_bert"

requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="MLX server requires Apple Silicon (arm64 Darwin)",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_healthz(port: int, proc: subprocess.Popen, timeout: float = 60.0) -> bool:
    """Poll ``/healthz`` until 200 or the subprocess dies."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(0.25)
    return False


@pytest.mark.integration
@requires_apple_silicon
def test_sigterm_exits_within_5s() -> None:
    """SIGTERM must result in exit code 0 within ``SHUTDOWN_DEADLINE_SECONDS``."""
    assert _BERT_FIXTURE.exists(), f"Missing test fixture: {_BERT_FIXTURE}"

    port = _free_port()
    env = os.environ.copy()
    # Server stdio -> temp file. PIPE without a reader can deadlock the
    # server once the OS pipe buffer fills (see _boot_server in
    # tests/integration/conftest.py).
    log_path = Path(tempfile.mkstemp(prefix=f"lifecycle-{port}-", suffix=".log")[1])
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "app.main",
            "launch",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(_BERT_FIXTURE),
            "--model-type",
            "embeddings",
            "--log-level",
            "WARNING",
        ],
        env=env,
        cwd=str(_REPO_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    log_fh.close()

    try:
        ready = _wait_for_healthz(port, proc, timeout=60.0)
        if not ready:
            out = log_path.read_text(errors="replace") if log_path.exists() else "(no log)"
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            pytest.fail(f"Server never became ready on :{port}.\nOutput:\n{out}")

        # Send SIGTERM and time the exit.
        proc.send_signal(signal.SIGTERM)
        start = time.monotonic()
        try:
            proc.wait(timeout=SHUTDOWN_DEADLINE_SECONDS + 2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"Process did not exit within {SHUTDOWN_DEADLINE_SECONDS + 2.0}s of SIGTERM")
        elapsed = time.monotonic() - start

        out_for_msg = log_path.read_text(errors="replace") if log_path.exists() else ""
        assert proc.returncode == 0, (
            f"expected exit code 0, got {proc.returncode}; output:\n{out_for_msg}"
        )
        assert elapsed < SHUTDOWN_DEADLINE_SECONDS, (
            f"SIGTERM exit took {elapsed:.2f}s, budget is {SHUTDOWN_DEADLINE_SECONDS}s"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        log_path.unlink(missing_ok=True)

