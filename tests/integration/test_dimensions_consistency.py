"""Both BERT and llama-cpp embedding handlers must honor `dimensions`
identically — they share the same apply_dimensions helper.
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


BERT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
LLAMA_CPP_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
LLAMA_CPP_HF_FILE = "nomic-embed-text-v1.5.f16.gguf"

_CONFIG_PATH = REPO_ROOT / "tests" / "integration" / "_dual_embed_config.yaml"


@pytest.fixture(scope="module")
def dual_embed_server() -> Iterator[str]:
    """Boot ONE multi-model server with BOTH BERT and llama-cpp embedders.

    Module-scoped: both models load once for all tests in this file.
    First boot can take 10-20 minutes for two ~250MB downloads.
    """
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        pytest.skip("Apple Silicon only")

    port = _free_port()
    _CONFIG_PATH.write_text(
        "server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {port}\n"
        "  log_level: WARNING\n"
        "models:\n"
        f"  - model_path: {BERT_MODEL}\n"
        "    model_type: embeddings\n"
        f"    served_model_name: {BERT_MODEL}\n"
        f"  - model_path: {LLAMA_CPP_MODEL}\n"
        f"    hf_file: {LLAMA_CPP_HF_FILE}\n"
        "    model_type: llama-cpp\n"
        f"    served_model_name: {LLAMA_CPP_MODEL}\n"
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
        ready = _wait_for_healthz(port, proc, timeout=1500.0)  # 25 min for cold double-download
        if not ready:
            try:
                out, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            pytest.fail(f"dual-embed server failed to become healthy on :{port}.\nOutput:\n{out}")
        yield f"http://127.0.0.1:{port}"
    finally:
        _teardown_server(proc)
        _CONFIG_PATH.unlink(missing_ok=True)


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_both_handlers_honor_dimensions_uniformly(dual_embed_server):
    """SHAPE + NORM assertion only. Different model architectures produce
    different vectors; we don't check values, only that:

      - Both responses are 256-d
      - Both are L2-unit (within float tolerance)
    """
    base = dual_embed_server
    payload = {"input": "consistency check input"}

    bert_resp = httpx.post(
        f"{base}/v1/embeddings",
        json={**payload, "model": BERT_MODEL, "dimensions": 256},
        timeout=120,
    )
    assert bert_resp.status_code == 200, bert_resp.text
    bert_vec = bert_resp.json()["data"][0]["embedding"]

    llama_resp = httpx.post(
        f"{base}/v1/embeddings",
        json={**payload, "model": LLAMA_CPP_MODEL, "dimensions": 256},
        timeout=300,  # cold path on first run after multi-boot
    )
    assert llama_resp.status_code == 200, llama_resp.text
    llama_vec = llama_resp.json()["data"][0]["embedding"]

    assert len(bert_vec) == 256
    assert len(llama_vec) == 256

    for vec, label in [(bert_vec, "bert"), (llama_vec, "llama-cpp")]:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4, f"{label} not L2-unit: |v|={norm}"
