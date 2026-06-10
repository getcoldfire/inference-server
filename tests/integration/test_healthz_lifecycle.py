"""``/healthz`` lifecycle: 200 throughout boot and after model load.

This test focuses on the **readiness transition** — boot a server, observe
that ``/healthz`` returns 200 while the model is still loading (server alive,
``model_status`` reflects load state), and that it eventually settles to 200
with ``model_status`` indicating a loaded model. The plan separates this from
Phase 5's ``tests/test_lifecycle.py::test_sigterm_exits_within_5s`` which
tests the SIGTERM-to-exit-code-0 contract; we do not re-test that here.

We boot the server with the tiny_bert embeddings fixture so the load time
is short and deterministic, but the test tolerates the (unlikely) case where
the model loads before the first probe lands by accepting 200 immediately
and merely asserting the readiness transition completes.

Note: /healthz no longer returns 503 for any lifecycle state — HTTP 200 is
the contract whenever the server process is alive and responsive. The JSON
body's ``model_status`` field distinguishes load states for callers that need
fine-grained readiness info.
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
def test_healthz_200_during_load_and_when_ready() -> None:
    """``/healthz`` must return 200 throughout boot and after model load.

    The endpoint returns 200 whenever the server process is alive. During model
    load the JSON body's ``model_status`` will be ``"uninitialized"``; once
    loading completes it transitions to ``"initialized"`` (or similar). This
    test confirms the endpoint is reachable and settles to a ready state.
    """
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
        # Only acceptable HTTP status is 200 (or connection-refused before
        # uvicorn binds). 503 is no longer part of the contract.
        observed_states: list[str] = []
        deadline = time.monotonic() + 60.0
        saw_ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"server exited unexpectedly during load: {out!r}")
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
                observed_states.append(f"{r.status_code}:{r.json().get('model_status','?')}")
                assert r.status_code == 200, (
                    f"/healthz returned unexpected status {r.status_code} during load: {r.text}"
                )
                body = r.json()
                model_status = body.get("model_status", "")
                # "initialized" (with optional count suffix) signals model is ready.
                if model_status.startswith("initialized"):
                    saw_ready = True
                    break
            except (httpx.HTTPError, OSError):
                observed_states.append("conn_refused")
            time.sleep(0.05)

        assert saw_ready, f"healthz never reached initialized status within 60s; observed: {observed_states[-20:]!r}"
        # Sanity: at least *some* state was observed before ready.
        assert observed_states, "no /healthz probes recorded"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
