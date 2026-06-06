"""Tests for DELETE /admin/models/{model_id:path}."""

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


def test_delete_simple_id_returns_200():
    app = _make_app()
    client = TestClient(app)
    client.post(
        "/admin/models/add",
        json={
            "model_path": "x",
            "served_model_name": "qwen:0.5b",
            "on_demand": True,
        },
    )
    r = client.delete("/admin/models/qwen:0.5b")
    assert r.status_code == HTTPStatus.OK, r.text
    assert r.json() == {"id": "qwen:0.5b", "deleted": True}
    assert "qwen:0.5b" not in app.state.registry.list_model_ids()


def test_delete_slash_containing_id_routes_correctly():
    """The {model_id:path} converter must accept slashes for HF IDs."""
    app = _make_app()
    client = TestClient(app)
    client.post(
        "/admin/models/add",
        json={
            "model_path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "on_demand": True,
        },
    )
    r = client.delete("/admin/models/mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert r.status_code == HTTPStatus.OK, r.text
    assert r.json()["id"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_delete_unknown_returns_404():
    app = _make_app()
    client = TestClient(app)
    r = client.delete("/admin/models/never-registered")
    assert r.status_code == HTTPStatus.NOT_FOUND


def test_delete_serving_active_returns_409_without_unregistering():
    """If the model is mid-request (refcount > 0), DELETE returns 409
    and the model stays registered. Task 1 enforces this inside
    unregister_model via ServingActiveError."""
    app = _make_app()
    client = TestClient(app)
    client.post(
        "/admin/models/add",
        json={
            "model_path": "x",
            "served_model_name": "qwen:0.5b",
            "on_demand": True,
        },
    )
    app.state.registry._on_demand_ref_count["qwen:0.5b"] = 1

    r = client.delete("/admin/models/qwen:0.5b")
    assert r.status_code == HTTPStatus.CONFLICT
    assert "serving" in r.json()["detail"].lower()
    assert "qwen:0.5b" in app.state.registry.list_model_ids(), "409 must not have removed the model"
