"""Smoke test for the llama-cpp embeddings handler.

Apple-Silicon only — requires Metal and a real GGUF download from HuggingFace.
First call triggers ~250MB download + Metal init (~5-15 min depending on
cache + network); cached on subsequent runs.

If this test HANGS or returns garbage vectors (NaN, all-zero, or wildly
different from run to run), it's the same thread-affinity quirk that MLX
exhibits (cf. fork buildout doc commits 8547c67, b75515c). Fix by adding
a warm-up call inside LlamaCppEmbeddingsLoader.ensure_loaded() that calls
llama.create_embedding(input="warm") once on the loader thread before
returning. See plan Phase 5 Step 4.
"""
from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.integration.conftest import (
    REPO_ROOT,
    requires_apple_silicon,
    _free_port,
    _wait_for_healthz,
    _teardown_server,
)


PILOT_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5-GGUF"
PILOT_HF_FILE = "nomic-embed-text-v1.5.f16.gguf"

_CONFIG_PATH = REPO_ROOT / "tests" / "integration" / "_llama_cpp_smoke_config.yaml"


@pytest.fixture(scope="module")
def llama_cpp_server() -> Iterator[tuple[str, int]]:
    """Boot one llama-cpp embedding server per module; yield (base_url, pid)."""
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        pytest.skip("llama-cpp Metal path requires Apple Silicon")

    # Use a YAML config so we can pass hf_file (single-model --model-path
    # mode has no flag for it).
    port = _free_port()
    _CONFIG_PATH.write_text(
        "server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {port}\n"
        "  log_level: WARNING\n"
        "models:\n"
        f"  - model_path: {PILOT_MODEL_ID}\n"
        f"    hf_file: {PILOT_HF_FILE}\n"
        "    model_type: llama-cpp\n"
        f"    served_model_name: {PILOT_MODEL_ID}\n"
    )
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main", "launch", "--config", str(_CONFIG_PATH)],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # 15-minute deadline accommodates first-time GGUF download.
        ready = _wait_for_healthz(port, proc, timeout=900.0)
        if not ready:
            try:
                out, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"llama-cpp server failed to become healthy on :{port}.\nOutput:\n{out}")
        yield (f"http://127.0.0.1:{port}", proc.pid)
    finally:
        _teardown_server(proc)
        _CONFIG_PATH.unlink(missing_ok=True)


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_llama_cpp_full_dim_embedding(llama_cpp_server):
    base, _ = llama_cpp_server
    resp = httpx.post(
        f"{base}/v1/embeddings",
        json={
            "model": PILOT_MODEL_ID,
            "input": "the quick brown fox jumps over the lazy dog",
        },
        timeout=120,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    vec = body["data"][0]["embedding"]
    assert isinstance(vec, list)
    assert len(vec) == 768, f"expected 768-d, got {len(vec)}-d"
    # Sanity: not all zeros, not all NaN — thread-affinity quirk would
    # produce one of these if it triggers.
    nz = sum(1 for x in vec if x != 0.0)
    assert nz > 100, f"only {nz} non-zero components; suspect garbage output"
    for x in vec:
        assert not math.isnan(x), "NaN in vector — thread-affinity quirk?"


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_llama_cpp_matryoshka_truncation(llama_cpp_server):
    base, _ = llama_cpp_server
    resp = httpx.post(
        f"{base}/v1/embeddings",
        json={
            "model": PILOT_MODEL_ID,
            "input": "the quick brown fox jumps over the lazy dog",
            "dimensions": 256,
        },
        timeout=120,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    vec = body["data"][0]["embedding"]
    assert len(vec) == 256
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4, f"not L2-unit: |v|={norm}"
