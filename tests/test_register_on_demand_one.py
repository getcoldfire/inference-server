"""Tests for register_on_demand_one — the extracted lifespan on-demand branch."""

from __future__ import annotations

import pytest

from app.core.model_registry import ModelRegistry
from app.core.registration import register_on_demand_one


@pytest.mark.asyncio
async def test_register_on_demand_one_registers_without_spawning():
    """An on-demand entry must register metadata only, not spawn a handler."""
    reg = ModelRegistry()

    await register_on_demand_one(
        reg,
        model_id="qwen:0.5b",
        model_cfg_dict={"model_path": "mlx-community/Q-4bit", "model_type": "lm"},
        model_type="lm",
        model_path="mlx-community/Q-4bit",
        context_length=None,
        queue_config={"timeout": 300, "queue_size": 100},
        idle_timeout=30,
    )

    assert "qwen:0.5b" in reg.list_model_ids()
    assert "qwen:0.5b" in reg._on_demand_configs
    assert "qwen:0.5b" not in reg._handlers


@pytest.mark.asyncio
async def test_register_on_demand_one_duplicate_raises():
    reg = ModelRegistry()
    args = dict(
        model_id="qwen:0.5b",
        model_cfg_dict={},
        model_type="lm",
        model_path="x",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 10},
        idle_timeout=30,
    )
    await register_on_demand_one(reg, **args)
    with pytest.raises(ValueError):
        await register_on_demand_one(reg, **args)
