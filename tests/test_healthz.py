"""Tests for the /healthz alias.

The cli-v2 lifecycle contract requires a `GET /healthz` endpoint that:
- Returns 503 while the handler is still loading the model.
- Returns 200 once the handler is initialized.

Upstream already exposes `GET /health` with this behavior (driven by
`app.state.handler` / `app.state.registry`). `/healthz` must be a strict
alias — same payload, same status semantics.

These tests construct a minimal FastAPI app and mount the production router,
then poke `app.state` directly to simulate load progression. This avoids
booting a real MLX handler subprocess in unit tests.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.endpoints import router


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with the production router mounted."""
    app = FastAPI()
    app.include_router(router)
    # Default state: nothing loaded.
    app.state.handler = None
    app.state.registry = None
    return app


def test_healthz_returns_503_before_model_load():
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/healthz")
    assert r.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    body = r.json()
    assert body["status"] == "unhealthy"


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


def test_healthz_503_matches_health_503_body():
    """When unhealthy, /healthz body must mirror /health body exactly."""
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)
    r_health = client.get("/health")
    r_healthz = client.get("/healthz")
    assert r_health.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert r_healthz.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert r_health.json() == r_healthz.json()
