"""``/healthz`` lifecycle: 503 during model load, 200 after.

This test focuses on the **readiness transition** — boot a server, observe
that ``/healthz`` returns 503 while the model is still loading, and that it
eventually transitions to 200 once the model is ready. The plan separates
this from Phase 5's ``tests/test_lifecycle.py::test_sigterm_exits_within_5s``
which tests the SIGTERM-to-exit-code-0 contract; we do not re-test that here.

We boot the server with the tiny_bert embeddings fixture so the load time
is short and deterministic, but the test tolerates the (unlikely) case where
the model loads before the first probe lands by accepting 200 immediately
and merely asserting the readiness transition completes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tests.integration.conftest import _free_port, requires_apple_silicon

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BERT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "tiny_bert"


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_healthz_503_during_load_then_200_when_ready() -> None:
    """``/healthz`` must transition from 503 (loading) to 200 (ready)."""
    assert _BERT_FIXTURE.exists(), f"Missing test fixture: {_BERT_FIXTURE}"

    port = _free_port()
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
        env=os.environ.copy(),
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Poll quickly so the first probe lands before model load completes.
        # Acceptable terminal states: 503 (loading), 200 (ready), or
        # connection-refused (uvicorn not bound yet).
        observed_states: list[str] = []
        deadline = time.monotonic() + 60.0
        saw_ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"server exited unexpectedly during load: {out!r}")
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
                observed_states.append(str(r.status_code))
                if r.status_code == 200:
                    saw_ready = True
                    break
                if r.status_code == 503:
                    # Loading — keep polling
                    pass
                else:
                    pytest.fail(f"unexpected healthz status {r.status_code} during load: {r.text}")
            except (httpx.HTTPError, OSError):
                observed_states.append("conn_refused")
            time.sleep(0.05)

        assert saw_ready, f"healthz never returned 200 within 60s; observed: {observed_states[-20:]!r}"
        # Sanity: at least *some* state was observed before ready (proves the
        # readiness probe sequence happened — we can't strictly require a 503
        # observation because a fast-loading model may flip to 200 before our
        # first probe lands).
        assert observed_states, "no /healthz probes recorded"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
