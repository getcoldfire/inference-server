"""5-minute continuous /v1/embeddings against the llama-cpp pilot model.

Memory growth ceiling: 200 MB RSS over the 5-minute run. Initial generous
target; tighten after first real run per existing pattern (see
test_streaming_soak.py for the same approach).

Apple-Silicon only; gated by @pytest.mark.slow so it's excluded from
default ``make test``. Invoked via ``make test-soak`` (together with the
other slow soaks) or directly:

    pytest tests/slow/test_llama_cpp_soak.py -m slow -v
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import psutil
import pytest

from tests.integration.conftest import (
    REPO_ROOT,
    _free_port,
    _teardown_server,
    _wait_for_healthz,
    requires_apple_silicon,
)

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5-GGUF"
HF_FILE = "nomic-embed-text-v1.5.f16.gguf"

_CONFIG_PATH = REPO_ROOT / "tests" / "slow" / "_llama_cpp_soak_config.yaml"


@pytest.fixture(scope="module")
def llama_cpp_soak_server() -> Iterator[tuple[str, int]]:
    """Boot one llama-cpp server for the soak run; yield (base_url, pid).

    Module-scoped so warm-up cost is paid once. Config written to a temp
    YAML file (the CLI ``--model-path`` flag has no ``--hf-file`` counterpart
    in single-model mode, so YAML config is the supported path for GGUF).
    """
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        pytest.skip("llama-cpp Metal path requires Apple Silicon")

    port = _free_port()
    _CONFIG_PATH.write_text(
        "server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {port}\n"
        "  log_level: WARNING\n"
        "models:\n"
        f"  - model_path: {MODEL_ID}\n"
        f"    hf_file: {HF_FILE}\n"
        "    model_type: llama-cpp\n"
        f"    served_model_name: {MODEL_ID}\n"
    )
    # IMPORTANT: a 5-min soak emits enough log volume to fill the OS pipe
    # buffer (~64 KB on macOS). If we used subprocess.PIPE without a reader
    # thread, the server would block on write() halfway through and drop
    # throughput from ~200 req/s to <1 req/s — exactly what surfaced as
    # "10/88 failures" the first time this test ran. DEVNULL is the right
    # fix because the soak doesn't need server output; healthz polling
    # already covers liveness.
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main", "launch", "--config", str(_CONFIG_PATH)],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # 15-minute deadline accommodates first-time GGUF download (~250 MB).
        ready = _wait_for_healthz(port, proc, timeout=900.0)
        if not ready:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            pytest.fail(f"llama-cpp soak server failed to become healthy on :{port}")
        yield (f"http://127.0.0.1:{port}", proc.pid)
    finally:
        _teardown_server(proc)
        _CONFIG_PATH.unlink(missing_ok=True)


@requires_apple_silicon
@pytest.mark.slow
def test_llama_cpp_5min_continuous_embedding_soak(
    llama_cpp_soak_server: tuple[str, int],
) -> None:
    """Continuous /v1/embeddings for 5 minutes; assert RSS growth < 200 MB.

    The 200 MB ceiling is the initial calibration target — intentionally
    generous. Tighten to ~50 MB once a real baseline has been observed from
    the first successful run (same rationale as test_streaming_soak.py).
    """
    base, pid = llama_cpp_soak_server
    proc = psutil.Process(pid)

    # Warm the model — settles any one-time allocation before we snapshot RSS.
    with httpx.Client(timeout=900) as c:
        c.post(
            f"{base}/v1/embeddings",
            json={"model": MODEL_ID, "input": "warmup"},
        ).raise_for_status()

    baseline_rss = proc.memory_info().rss
    deadline = time.monotonic() + 300  # 5 minutes
    request_count = 0
    failures = 0

    with httpx.Client(timeout=30) as c:
        while time.monotonic() < deadline:
            assert proc.is_running(), f"server died after {request_count} requests"
            try:
                r = c.post(
                    f"{base}/v1/embeddings",
                    json={"model": MODEL_ID, "input": "soak request"},
                )
                r.raise_for_status()
                request_count += 1
            except Exception:
                failures += 1

    final_rss = proc.memory_info().rss
    growth_mb = (final_rss - baseline_rss) / (1024 * 1024)

    print(
        f"\nllama-cpp soak: {request_count} requests in 300s. "
        f"Baseline RSS: {baseline_rss / 1024 / 1024:.1f} MB. "
        f"Final RSS: {final_rss / 1024 / 1024:.1f} MB. "
        f"Growth: {growth_mb:.1f} MB. "
        f"Failures: {failures}."
    )

    assert failures == 0, f"{failures}/{request_count + failures} requests failed during soak"
    assert growth_mb < 200, f"RSS grew {growth_mb:.1f} MB over 5 min (ceiling 200 MB)"
