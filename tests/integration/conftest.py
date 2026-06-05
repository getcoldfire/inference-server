"""Shared fixtures for `tests/integration/` smoke tests.

Boots the real `python -m app.main launch ...` server with a real MLX model
and exposes a base URL plus the parent subprocess handle. The fixture is
session-scoped so all chat-smoke tests share one server boot — model downloads
and warm-up are expensive (≥ 1 minute even with cached weights), so paying
that cost once per session is meaningful.

The embeddings smoke uses a different model and gets its own module-scoped
server fixture defined in `test_embeddings_smoke.py`.
"""

from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Common constants / helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# Use the 1B variant for chat smoke — exercises the same scheduler/streaming
# code paths as the 3B at ~750MB download instead of ~2GB. The Llama-3.2-1B
# tokenizer and chat template are identical to the 3B's, so wire format is
# the same. Streaming, KV cache, prompt cache — all the relevant code paths
# run unchanged.
CHAT_MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# nomic-embed-text-v1.5 is the validated tier embedding model. ~280MB.
# Uses RoPE + SwiGLU, so this is also the first time `_remap_hf_to_internal`
# meets real nomic-bert weights — see test_embeddings_smoke.py.
# NOTE: The plan referenced `mlx-community/nomic-embed-text-v1.5` but that
# repo doesn't exist on HF; the canonical upstream repo is `nomic-ai/...`.
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"


requires_apple_silicon = pytest.mark.skipif(
    platform.machine() != "arm64" or platform.system() != "Darwin",
    reason="MLX server requires Apple Silicon (arm64 Darwin)",
)


def _free_port() -> int:
    """Pick a free TCP port on localhost. Race-prone but fine for tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_healthz(
    port: int,
    proc: subprocess.Popen,
    timeout: float = 300.0,
) -> bool:
    """Poll `/healthz` until 200 or the subprocess dies.

    Default 5-minute deadline accommodates first-time model download
    (~1GB combined for chat + embeddings on a slow network).
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(1.0)
    return False


def _boot_server(model_id: str, model_type: str = "lm") -> tuple[subprocess.Popen, int]:
    """Spawn `python -m app.main launch` for the given model.

    Returns `(proc, port)`. Caller is responsible for SIGTERM teardown.
    """
    port = _free_port()
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-m",
        "app.main",
        "launch",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        model_id,
        "--model-type",
        model_type,
        "--no-log-file",
        "--log-level",
        "WARNING",
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, port


def _teardown_server(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL if it doesn't exit in time."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Session-scoped chat-server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chat_server() -> Iterator[tuple[str, int]]:
    """Boot one chat server per session; yield `(base_url, pid)`.

    Tests that need only the URL can ignore the pid; the soak tests use it
    to attach `psutil` for RSS sampling.
    """
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        pytest.skip("MLX requires Apple Silicon")

    proc, port = _boot_server(CHAT_MODEL_ID, model_type="lm")
    try:
        ready = _wait_for_healthz(port, proc, timeout=300.0)
        if not ready:
            try:
                out, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"chat server failed to become healthy on :{port}.\nOutput:\n{out}")

        yield (f"http://127.0.0.1:{port}", proc.pid)
    finally:
        _teardown_server(proc)
