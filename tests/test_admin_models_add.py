"""Tests for POST /admin/models/add.

Pattern follows tests/test_healthz.py — minimal FastAPI app, mount the
production admin router, ModelRegistry on app.state. Unit tests cover
the validation surface; integration smoke (Task 7) exercises a real
fork process.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_models import router as admin_router
from app.core.model_registry import ModelRegistry


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router)
    app.state.registry = ModelRegistry()
    return app


def test_add_on_demand_returns_200_with_metadata():
    app = _make_app()
    client = TestClient(app)

    r = client.post(
        "/admin/models/add",
        json={
            "model_path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "served_model_name": "qwen:0.5b",
            "on_demand": True,
            "on_demand_idle_timeout": 30,
        },
    )
    assert r.status_code == HTTPStatus.OK, r.text
    body = r.json()
    assert body["id"] == "qwen:0.5b"
    assert body["type"] == "lm"
    assert isinstance(body["created_at"], int)
    assert "qwen:0.5b" in app.state.registry.list_model_ids()


def test_add_default_served_model_name_falls_back_to_model_path():
    app = _make_app()
    client = TestClient(app)
    r = client.post(
        "/admin/models/add",
        json={
            "model_path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "on_demand": True,
        },
    )
    assert r.status_code == HTTPStatus.OK
    assert r.json()["id"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_add_on_demand_false_returns_400_with_v012_pointer():
    """Resident hot-add is deferred to v0.1.2 per spec §8."""
    app = _make_app()
    client = TestClient(app)
    r = client.post(
        "/admin/models/add",
        json={
            "model_path": "x",
            "on_demand": False,
        },
    )
    assert r.status_code == HTTPStatus.BAD_REQUEST
    assert "v0.1.2" in r.json()["detail"] or "resident" in r.json()["detail"].lower()


def test_add_on_demand_omitted_defaults_to_false_returns_400():
    """on_demand defaults to False per schema; this is the SAME error path."""
    app = _make_app()
    client = TestClient(app)
    r = client.post("/admin/models/add", json={"model_path": "x"})
    assert r.status_code == HTTPStatus.BAD_REQUEST


def test_add_duplicate_served_model_name_returns_409():
    app = _make_app()
    client = TestClient(app)
    body = {"model_path": "x", "served_model_name": "qwen:0.5b", "on_demand": True}
    assert client.post("/admin/models/add", json=body).status_code == HTTPStatus.OK
    r = client.post("/admin/models/add", json=body)
    assert r.status_code == HTTPStatus.CONFLICT
    assert "already" in r.json()["detail"].lower()


def test_add_missing_model_path_returns_422():
    """FastAPI Pydantic validation surface — 422, not 400."""
    app = _make_app()
    client = TestClient(app)
    r = client.post("/admin/models/add", json={"on_demand": True})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_add_invalid_model_type_returns_422():
    app = _make_app()
    client = TestClient(app)
    r = client.post(
        "/admin/models/add",
        json={
            "model_path": "x",
            "model_type": "audio",
            "on_demand": True,
        },
    )
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_admin_router_mounted_via_setup_server_in_multi_handler_mode():
    """End-to-end smoke that the conditional include in setup_server fires.

    Constructs the app via the real setup_server entry point with a minimal
    MultiModelServerConfig (no models = lifespan loop is a no-op). The admin
    router must be reachable; an invalid POST returns 422 (Pydantic validation),
    NOT 404 (router not mounted).

    Note: setup_server returns a uvicorn.Config and mutates the module-level
    ``app`` global — we reach into the module to grab the constructed FastAPI
    instance after calling setup_server.
    """
    from app import server as server_module
    from app.config import MultiModelServerConfig

    cfg = MultiModelServerConfig(models=[], host="127.0.0.1", port=0)
    server_module.setup_server(cfg)
    app = server_module.app
    assert app is not None, "setup_server did not populate module-level app"

    with TestClient(app) as client:
        r = client.post("/admin/models/add", json={})
        assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
            f"Admin router not mounted; got {r.status_code}: {r.text}"
        )
