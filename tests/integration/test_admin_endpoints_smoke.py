"""End-to-end against a real coldfire-inference-server subprocess.

Skipped unless COLDFIRE_MLX_INTEGRATION=1. Apple Silicon only.
Requires Llama-3.2-1B-Instruct-4bit cached in ~/.cache/huggingface/hub.

Pinned to coldfire-inference-server >= v0.1.1 by an explicit --version
assertion in the fixture; an older binary would silently lack admin
endpoints and the tests would meaningless-pass via 404.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest

from .conftest import _free_port, _teardown_server, _wait_for_healthz

pytestmark = pytest.mark.skipif(
    os.environ.get("COLDFIRE_MLX_INTEGRATION") != "1",
    reason="set COLDFIRE_MLX_INTEGRATION=1 to run; spawns real subprocess + HF",
)

FIXTURE_TEMPLATE = Path(__file__).parent.parent / "fixtures" / "admin_smoke_config.yaml"


def _assert_min_version(min_major: int, min_minor: int, min_patch: int) -> None:
    out = subprocess.check_output(["coldfire-inference-server", "--version"], text=True)
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if not m:
        pytest.fail(f"could not parse coldfire-inference-server --version output: {out!r}")
    have = tuple(int(g) for g in m.groups())
    need = (min_major, min_minor, min_patch)
    if have < need:
        pytest.fail(
            f"coldfire-inference-server version {have} is too old; need >= {need}. Run `brew upgrade coldfire-inference-server`."
        )


@pytest.fixture(scope="module")
def server():
    _assert_min_version(0, 1, 1)

    # Template the fixture YAML with an ephemeral port so parallel runs and
    # leftover servers from prior runs don't collide. Also keeps log_level
    # at WARNING (set in the template) to avoid PIPE-deadlock from INFO spam.
    port = _free_port()
    template = FIXTURE_TEMPLATE.read_text()
    rendered = template.replace("__PORT__", str(port))

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        tmp.write(rendered)
        tmp.flush()
        tmp.close()

        # stderr -> STDOUT so a single drain handles both; stdout -> DEVNULL
        # so a long-running server (≥120s with chat completions) can't fill
        # the ~64KB pipe buffer and deadlock. WARNING level keeps this quiet
        # anyway, but DEVNULL is belt-and-suspenders.
        proc = subprocess.Popen(
            ["coldfire-inference-server", "launch", "--config", tmp.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # _wait_for_healthz polls /healthz and bails early if the
            # subprocess exits (proc.poll() != None), so a crash at second 2
            # doesn't burn the full 120s timeout.
            if not _wait_for_healthz(port, proc, timeout=120.0):
                _teardown_server(proc)
                pytest.fail(f"server did not become healthy on :{port} within 120s")

            yield f"http://127.0.0.1:{port}"
        finally:
            _teardown_server(proc)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def test_hot_add_then_chat_then_remove(server: str):
    r = httpx.get(f"{server}/v1/models")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 1

    body = {
        "model_path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "served_model_name": "qwen-hotadd",
        "on_demand": True,
        "on_demand_idle_timeout": 60,
    }
    r = httpx.post(f"{server}/admin/models/add", json=body, timeout=10.0)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "qwen-hotadd"

    r = httpx.get(f"{server}/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert "qwen-hotadd" in ids

    r = httpx.post(
        f"{server}/v1/chat/completions",
        json={
            "model": "qwen-hotadd",
            "messages": [{"role": "user", "content": "say hi in 3 words"}],
            "max_tokens": 10,
            "stream": False,
        },
        timeout=180.0,
    )
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "qwen-hotadd"

    r = httpx.delete(f"{server}/admin/models/qwen-hotadd", timeout=10.0)
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = httpx.get(f"{server}/v1/models")
    assert len(r.json()["data"]) == 1


def test_hot_add_slash_containing_default_id(server: str):
    """Verify the {model_id:path} converter on real FastAPI."""
    full_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    r = httpx.post(
        f"{server}/admin/models/add",
        json={"model_path": full_id, "on_demand": True},
        timeout=10.0,
    )
    assert r.status_code == 200, r.text
    assert r.json()["id"] == full_id

    r = httpx.delete(f"{server}/admin/models/{full_id}", timeout=10.0)
    assert r.status_code == 200, r.text


def test_delete_unknown_returns_404(server: str):
    r = httpx.delete(f"{server}/admin/models/never-registered", timeout=5.0)
    assert r.status_code == 404


def test_add_duplicate_returns_409(server: str):
    body = {
        "model_path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "served_model_name": "dup-test",
        "on_demand": True,
    }
    assert httpx.post(f"{server}/admin/models/add", json=body, timeout=10.0).status_code == 200
    assert httpx.post(f"{server}/admin/models/add", json=body, timeout=10.0).status_code == 409
    httpx.delete(f"{server}/admin/models/dup-test", timeout=5.0)


def test_add_on_demand_false_returns_400(server: str):
    r = httpx.post(
        f"{server}/admin/models/add",
        json={"model_path": "x", "on_demand": False},
        timeout=5.0,
    )
    assert r.status_code == 400
    assert "v0.1.2" in r.json()["detail"] or "resident" in r.json()["detail"].lower()
