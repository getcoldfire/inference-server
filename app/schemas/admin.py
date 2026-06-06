"""Pydantic schemas for the admin endpoints.

POST /admin/models/add        - AddModelRequest -> AddModelResponse
DELETE /admin/models/{id}     - DeleteModelResponse on success

v0.1.1 scope: on_demand: true ONLY. The POST endpoint rejects
on_demand: false with 400 (deferred to v0.1.2 — resident hot-add
needs full ModelEntryConfig passthrough).

Pydantic v2 reserves the ``model_`` prefix; ``protected_namespaces=()``
suppresses the spurious warning about ``model_path`` / ``model_type``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AddModelRequest(BaseModel):
    """Request body for POST /admin/models/add."""

    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(..., min_length=1, description="HuggingFace repo ID or local path")
    model_type: Literal["lm", "embeddings"] = Field(default="lm")
    served_model_name: str | None = Field(
        default=None,
        description="Optional alias. Defaults to model_path if omitted.",
    )
    context_length: int | None = Field(default=None)
    on_demand: bool = Field(default=False, description="v0.1.1: must be true (resident deferred to v0.1.2)")
    on_demand_idle_timeout: int = Field(default=300, ge=1)
    queue_timeout: int = Field(default=300, ge=1)
    queue_size: int = Field(default=100, ge=1)


class AddModelResponse(BaseModel):
    """Response body for POST /admin/models/add (200)."""

    id: str
    type: str
    created_at: int


class DeleteModelResponse(BaseModel):
    """Response body for DELETE /admin/models/{id} (200)."""

    id: str
    deleted: Literal[True] = True
