"""Admin endpoints for hot-add/hot-remove of models.

POST /admin/models/add               - register a new on-demand model
DELETE /admin/models/{model_id:path} - unregister a model

Mounted ONLY in multi-handler mode (when ``app.state.registry`` is set,
which is true only when the app was constructed via
``setup_server(MultiModelServerConfig)``). The fork binds 127.0.0.1
by default; loopback is the auth boundary at v0.1.1 (spec §8.4).

v0.1.1 scope: on_demand: true ONLY. Resident hot-add (on_demand: false)
returns 400 with a pointer to v0.1.2 — see spec §8 scope cut.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from ..core.model_registry import ServingActiveError
from ..core.registration import register_on_demand_one
from ..schemas.admin import AddModelRequest, AddModelResponse, DeleteModelResponse

router = APIRouter(prefix="/admin/models", tags=["admin"])


@router.post("/add", response_model=AddModelResponse)
async def add_model(req: AddModelRequest, request: Request) -> AddModelResponse:
    """Hot-add an on-demand model. Resident (on_demand=false) returns 400."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Multi-handler mode is not active (no ModelRegistry)",
        )

    if not req.on_demand:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(
                "Resident (on_demand: false) hot-add is deferred to coldfire-mlx-server v0.1.2. "
                "v0.1.1 accepts on_demand: true only. Set on_demand: true (model will load on first request)."
            ),
        )

    model_id = req.served_model_name or req.model_path
    if model_id in registry.list_model_ids():
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f"Model '{model_id}' is already registered",
        )

    # Build the model_cfg_dict the registry stores. For on-demand it's
    # consumed lazily by ensure_on_demand_loaded which tolerates missing
    # keys via defaults — so the minimum-field admin payload is safe.
    model_cfg_dict = {
        "model_path": req.model_path,
        "model_type": req.model_type,
        "served_model_name": model_id,
        "context_length": req.context_length,
        "on_demand": True,
        "on_demand_idle_timeout": req.on_demand_idle_timeout,
        "queue_timeout": req.queue_timeout,
        "queue_size": req.queue_size,
    }

    try:
        await register_on_demand_one(
            registry,
            model_id=model_id,
            model_cfg_dict=model_cfg_dict,
            model_type=req.model_type,
            model_path=req.model_path,
            context_length=req.context_length,
            queue_config={"timeout": req.queue_timeout, "queue_size": req.queue_size},
            idle_timeout=req.on_demand_idle_timeout,
        )
    except ValueError as e:
        # Race: registered between our check and the call.
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"add_model failed for '{model_id}'")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {e}",
        ) from e

    meta = registry.get_metadata(model_id)
    return AddModelResponse(id=meta.id, type=meta.type, created_at=meta.created_at)


@router.delete("/{model_id:path}", response_model=DeleteModelResponse)
async def delete_model(model_id: str, request: Request) -> DeleteModelResponse:
    """Hot-remove a model. 200 on success; 404 if not registered;
    409 if currently mid-request.

    The path converter ``{model_id:path}`` is required so HF-style
    IDs containing ``/`` route correctly.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Multi-handler mode is not active (no ModelRegistry)",
        )

    try:
        await registry.unregister_model(model_id)
    except ServingActiveError as e:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f"Model '{e.model_id}' is currently serving a request; retry shortly",
        ) from e
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Model '{model_id}' is not registered",
        ) from None
    except Exception as e:
        logger.exception(f"delete_model failed for '{model_id}'")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Unregister failed: {e}",
        ) from e

    return DeleteModelResponse(id=model_id)
