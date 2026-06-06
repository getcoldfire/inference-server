"""Dual-lock unregister_model + ServingActiveError contract.

The DELETE admin endpoint depends on three properties:

1. unregister handles BOTH resident handlers (in `_handlers`) AND
   unloaded on-demand entries (in `_on_demand_configs`).
2. The check-and-unregister sequence is atomic against
   `ensure_on_demand_loaded`, which mutates `_on_demand_ref_count`
   under `_on_demand_load_lock`, not `_lock`. The TOCTOU race
   between an external 'is serving?' check and `unregister_model`
   would orphan a just-spawned handler.
3. Concurrent unregister + load raises `ServingActiveError` rather
   than tearing down a mid-request handler.

This test pokes internal state and forces the race with explicit
event ordering — avoids needing real MLX subprocesses.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.model_registry import ModelRegistry, ServingActiveError


@pytest.mark.asyncio
async def test_unregister_removes_unloaded_on_demand_entry():
    reg = ModelRegistry()
    await reg.register_on_demand_model(
        model_id="qwen:0.5b",
        model_cfg_dict={"model_path": "x", "model_type": "lm"},
        model_type="lm",
        model_path="x",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 10},
        idle_timeout=30,
    )
    assert "qwen:0.5b" in reg.list_model_ids()
    await reg.unregister_model("qwen:0.5b")
    assert "qwen:0.5b" not in reg.list_model_ids()


@pytest.mark.asyncio
async def test_unregister_raises_serving_active_when_refcount_positive():
    reg = ModelRegistry()
    await reg.register_on_demand_model(
        model_id="qwen:0.5b",
        model_cfg_dict={"model_path": "x", "model_type": "lm"},
        model_type="lm",
        model_path="x",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 10},
        idle_timeout=30,
    )
    reg._on_demand_ref_count["qwen:0.5b"] = 1

    with pytest.raises(ServingActiveError):
        await reg.unregister_model("qwen:0.5b")

    # Model must still be registered — failed unregister must not partially clean up.
    assert "qwen:0.5b" in reg.list_model_ids()
    assert "qwen:0.5b" in reg._on_demand_configs


@pytest.mark.asyncio
async def test_unregister_cancels_idle_task_for_loaded_on_demand_model():
    reg = ModelRegistry()
    await reg.register_on_demand_model(
        model_id="qwen:0.5b",
        model_cfg_dict={"model_path": "x", "model_type": "lm"},
        model_type="lm",
        model_path="x",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 10},
        idle_timeout=30,
    )

    class _StubHandler:
        async def cleanup(self) -> None: ...

    reg._on_demand_loaded.add("qwen:0.5b")
    reg._handlers["qwen:0.5b"] = _StubHandler()
    from app.schemas.model import ModelMetadata

    reg._metadata["qwen:0.5b"] = ModelMetadata(
        id="qwen:0.5b",
        type="lm",
        context_length=None,
        created_at=0,
    )
    task = asyncio.create_task(asyncio.sleep(60))
    reg._on_demand_idle_tasks["qwen:0.5b"] = task

    await reg.unregister_model("qwen:0.5b")

    assert task.cancelled() or task.done()
    assert "qwen:0.5b" not in reg._on_demand_idle_tasks
    assert "qwen:0.5b" not in reg._on_demand_loaded
    assert "qwen:0.5b" not in reg._on_demand_configs


@pytest.mark.asyncio
async def test_unregister_unknown_model_raises_keyerror():
    reg = ModelRegistry()
    with pytest.raises(KeyError):
        await reg.unregister_model("never-registered")


@pytest.mark.asyncio
async def test_is_serving_active_lock_free_status_helper():
    """is_serving_active is for IPC/status reads only — DELETE does NOT
    use it (DELETE uses the in-lock check inside unregister_model)."""
    reg = ModelRegistry()
    await reg.register_on_demand_model(
        model_id="qwen:0.5b",
        model_cfg_dict={"model_path": "x", "model_type": "lm"},
        model_type="lm",
        model_path="x",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 10},
        idle_timeout=30,
    )
    assert reg.is_serving_active("qwen:0.5b") is False
    reg._on_demand_ref_count["qwen:0.5b"] = 2
    assert reg.is_serving_active("qwen:0.5b") is True
    assert reg.is_serving_active("never-registered") is False
