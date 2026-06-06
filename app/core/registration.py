"""Reusable on-demand model registration.

Extracted from ``app/server.py``'s multi-handler lifespan loop
(on-demand branch only — lines ~326-339 at extraction time) so that
``POST /admin/models/add`` can register hot-added on-demand models
with identical semantics. A hot-added on-demand model is
indistinguishable from a config-declared on-demand model after this
function returns.

**Resident registration is NOT extracted.** v0.1.1 admin rejects
``on_demand: false`` because the spawn boundary requires the full
``ModelEntryConfig`` shape (~25 fields) that the admin request body
doesn't carry. Resident hot-add lands in v0.1.2 with a proper
config-passthrough surface.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .model_registry import ModelRegistry


async def register_on_demand_one(
    registry: ModelRegistry,
    *,
    model_id: str,
    model_cfg_dict: dict[str, Any],
    model_type: str,
    model_path: str,
    context_length: int | None,
    queue_config: dict[str, Any],
    idle_timeout: int,
) -> None:
    """Register a single on-demand model. Returns immediately;
    the handler subprocess is spawned by the registry's
    ``ensure_on_demand_loaded`` path on first request.

    Raises
    ------
    ValueError
        If ``model_id`` is already registered (re-raised from
        ``ModelRegistry.register_on_demand_model``).
    """
    await registry.register_on_demand_model(
        model_id=model_id,
        model_cfg_dict=model_cfg_dict,
        model_type=model_type,
        model_path=model_path,
        context_length=context_length,
        queue_config=queue_config,
        idle_timeout=idle_timeout,
    )
    logger.info(
        f"Model '{model_id}' registered as on-demand (will load when first requested)"
    )
