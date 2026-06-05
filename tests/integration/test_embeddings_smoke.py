"""Embeddings smoke tests.

Two tests:

* ``test_embeddings_tiny_bert_smoke`` — boots an embeddings server with the
  in-repo ``tests/fixtures/tiny_bert`` model and verifies the
  ``/v1/embeddings`` endpoint returns L2-normalized vectors of the expected
  dimension. This is the deterministic, always-runnable smoke test and the
  primary acceptance signal for Phase 6 embeddings coverage.

* ``test_embeddings_nomic_v1_5_smoke`` — boots an embeddings server with the
  real ``nomic-ai/nomic-embed-text-v1.5`` HuggingFace model and verifies
  768-dim L2-normalized vectors.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.integration.conftest import (
    EMBEDDING_MODEL_ID,
    _boot_server,
    _free_port,
    _teardown_server,
    _wait_for_healthz,
    requires_apple_silicon,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BERT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "tiny_bert"


@pytest.fixture(scope="module")
def tiny_bert_server() -> Iterator[str]:
    """Boot an embeddings server with the in-repo tiny_bert fixture."""
    assert _BERT_FIXTURE.exists(), f"Missing fixture: {_BERT_FIXTURE}"

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
            "--no-log-file",
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
        ready = _wait_for_healthz(port, proc, timeout=60.0)
        if not ready:
            try:
                out, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"tiny_bert server never became ready on :{port}.\nOutput:\n{out}")
        yield f"http://127.0.0.1:{port}"
    finally:
        _teardown_server(proc)


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_embeddings_tiny_bert_smoke(tiny_bert_server: str) -> None:
    """``/v1/embeddings`` returns L2-normalized vectors for 3 inputs (tiny_bert)."""
    base_url = tiny_bert_server
    # Served-model-name defaults to the model path when --served-model-name is unset.
    model_id = str(_BERT_FIXTURE)
    r = httpx.post(
        f"{base_url}/v1/embeddings",
        json={
            "model": model_id,
            "input": ["hello", "world", "embeddings smoke test"],
        },
        timeout=30.0,
    )
    assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("object") == "list", f"unexpected top-level: {body}"
    data = body.get("data", [])
    assert len(data) == 3, f"expected 3 embeddings, got {len(data)}: {data!r}"
    # tiny_bert fixture has hidden_size=32 (see scripts/generate_tiny_bert_fixture.py).
    for i, entry in enumerate(data):
        vec = entry.get("embedding")
        assert vec is not None, f"missing 'embedding' on item {i}: {entry}"
        assert len(vec) == 32, f"item {i}: expected 32-dim, got {len(vec)}"
        norm_sq = sum(x * x for x in vec)
        norm = norm_sq**0.5
        assert abs(norm - 1.0) < 1e-4, f"item {i}: L2 norm {norm} != 1.0"


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_embeddings_nomic_v1_5_smoke() -> None:
    """``/v1/embeddings`` against real nomic-embed-text-v1.5: 768-dim L2-norm."""
    proc, port_back = _boot_server(EMBEDDING_MODEL_ID, model_type="embeddings")
    try:
        ready = _wait_for_healthz(port_back, proc, timeout=300.0)
        if not ready:
            try:
                out, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"nomic-embed server never became ready on :{port_back}.\nOutput:\n{out}")

        r = httpx.post(
            f"http://127.0.0.1:{port_back}/v1/embeddings",
            json={
                "model": EMBEDDING_MODEL_ID,
                "input": ["hello", "world", "nomic smoke test"],
            },
            timeout=30.0,
        )
        assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text}"
        body = r.json()
        data = body.get("data", [])
        assert len(data) == 3, f"expected 3 embeddings, got {len(data)}"
        for i, entry in enumerate(data):
            vec = entry["embedding"]
            assert len(vec) == 768, f"item {i}: expected 768-dim, got {len(vec)}"
            norm = sum(x * x for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-4, f"item {i}: L2 norm {norm} != 1.0"
    finally:
        _teardown_server(proc)
