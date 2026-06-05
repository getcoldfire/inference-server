"""Shared fixtures for ``tests/slow/`` soak tests.

Re-exports the session-scoped ``chat_server`` fixture from
``tests/integration/conftest.py`` so the slow tests don't duplicate the
server-boot logic. Adds a module-scoped ``embedding_server`` fixture that
boots the real ``nomic-embed-text-v1.5`` model (~280MB) for the correctness
soaks in Task 26.

All fixtures here are session/module-scoped — model load is expensive and we
want to pay it once per test run, not per test.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

import pytest

# Re-export from integration conftest so pytest can resolve ``chat_server``
# in slow tests without redefining the fixture.
from tests.integration.conftest import (  # noqa: F401
    CHAT_MODEL_ID,
    EMBEDDING_MODEL_ID,
    _boot_server,
    _free_port,
    _teardown_server,
    _wait_for_healthz,
    chat_server,
    requires_apple_silicon,
)


@pytest.fixture(scope="module")
def embedding_server() -> Iterator[tuple[str, int]]:
    """Boot one nomic-embed-text-v1.5 server per module; yield ``(base_url, pid)``.

    Module-scoped so the embedding correctness tests (nomic + mxbai) each get
    a fresh server but tests within one module share. Yields the PID so soaks
    can attach psutil for RSS sampling if needed.
    """
    proc, port = _boot_server(EMBEDDING_MODEL_ID, model_type="embeddings")
    try:
        ready = _wait_for_healthz(port, proc, timeout=300.0)
        if not ready:
            try:
                out, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"embedding server failed to become healthy on :{port}.\nOutput:\n{out}")
        yield (f"http://127.0.0.1:{port}", proc.pid)
    finally:
        _teardown_server(proc)
