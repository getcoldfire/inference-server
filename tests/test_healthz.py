"""Tests for the /healthz alias.

The cli-v2 lifecycle contract requires a ``GET /healthz`` endpoint that
always returns HTTP 200 when the server process is alive and responsive.
Zero-models is a valid lifecycle state (empty boot before hot-add via
``/admin/models/load``); the server is still healthy in that state.

The JSON body's ``model_status`` field conveys fine-grained readiness info
for callers that need it:
  - ``"no_models"``       — registry present, nothing loaded yet
  - ``"uninitialized"``   — single-handler mode, model not yet loaded
  - ``"initialized"``     — single-handler mode, model ready
  - ``"initialized (N model(s))"`` — registry mode, N models loaded

Upstream already exposes ``GET /health`` with the same behavior. ``/healthz``
must be a strict alias — same payload, same status semantics.

These tests construct a minimal FastAPI app and mount the production router,
then poke ``app.state`` directly to simulate load progression. This avoids
booting a real MLX handler subprocess in unit tests.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with the production router mounted."""
    app = FastAPI()
    app.include_router(router)
    # Default state: nothing loaded.
    app.state.handler = None
    app.state.registry = None
    return app


# ---------------------------------------------------------------------------
# Zero-models / not-yet-loaded: server must still report 200
# ---------------------------------------------------------------------------


def test_healthz_returns_200_with_zero_models():
    """Server is alive with no models loaded — HTTP layer must say so.

    Zero-models is the natural state immediately after boot when the operator
    intends to hot-add models via /admin/models/load. The backendlauncher in
    cli-v2 polls /healthz expecting 200 to gate IPC socket creation; returning
    503 here would permanently block that flow.
    """
    app = _make_app()

    class _FakeRegistry:
        def get_model_count(self) -> int:
            return 0

        def list_models(self) -> list:
            return []

    app.state.registry = _FakeRegistry()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/healthz")
    assert resp.status_code == 200, f"want 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("model_status") == "no_models", body
    assert body.get("status") == "ok", body


def test_healthz_returns_200_before_model_load():
    """Single-handler mode: handler not yet set → still 200."""
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/healthz")
    assert r.status_code == HTTPStatus.OK
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_status"] == "uninitialized"


# ---------------------------------------------------------------------------
# Model loaded: 200 with populated model_id
# ---------------------------------------------------------------------------


def test_healthz_returns_200_after_model_load():
    app = _make_app()

    # Fake a fully-initialized single-handler runtime.
    class _FakeHandler:
        model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    app.state.handler = _FakeHandler()
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == HTTPStatus.OK
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_id"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"


# ---------------------------------------------------------------------------
# /health alias parity
# ---------------------------------------------------------------------------


def test_health_alias_still_works():
    """The original /health endpoint must remain functional and identical."""
    app = _make_app()

    class _FakeHandler:
        model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    app.state.handler = _FakeHandler()
    client = TestClient(app)
    r_health = client.get("/health")
    r_healthz = client.get("/healthz")
    assert r_health.status_code == HTTPStatus.OK
    assert r_healthz.status_code == HTTPStatus.OK
    # The two endpoints must return the same body so any cli-v2 / Kubernetes
    # probe sees identical responses regardless of which it hits.
    assert r_health.json() == r_healthz.json()


def test_health_and_healthz_match_with_zero_models():
    """/health and /healthz must return identical payloads when no models are loaded."""
    app = _make_app()

    class _FakeRegistry:
        def get_model_count(self) -> int:
            return 0

        def list_models(self) -> list:
            return []

    app.state.registry = _FakeRegistry()
    client = TestClient(app, raise_server_exceptions=False)
    r_health = client.get("/health")
    r_healthz = client.get("/healthz")
    assert r_health.status_code == HTTPStatus.OK
    assert r_healthz.status_code == HTTPStatus.OK
    assert r_health.json() == r_healthz.json()
