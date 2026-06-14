"""Pydantic-level tests for the AddModelRequest llama-cpp branch.

Also covers Phase 4 wire-up: the dispatcher must return a handler whose
``handler_type`` is the FUNCTIONAL kind ("embeddings") when the config
``model_type`` is the runtime kind ("llama-cpp").
"""
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from app.schemas.admin import AddModelRequest


def test_minimum_llama_cpp_body_parses():
    body = AddModelRequest.model_validate({
        "model_path": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "model_type": "llama-cpp",
        "hf_file": "nomic-embed-text-v1.5.f16.gguf",
        "on_demand": True,
    })
    assert body.model_type == "llama-cpp"
    assert body.hf_file == "nomic-embed-text-v1.5.f16.gguf"


def test_llama_cpp_accepts_n_gpu_layers_negative_one():
    body = AddModelRequest.model_validate({
        "model_path": "org/repo-GGUF",
        "model_type": "llama-cpp",
        "hf_file": "x.gguf",
        "n_gpu_layers": -1,
        "on_demand": True,
    })
    assert body.n_gpu_layers == -1


def test_llama_cpp_rejects_n_gpu_layers_less_than_minus_one():
    with pytest.raises(ValidationError):
        AddModelRequest.model_validate({
            "model_path": "org/repo-GGUF",
            "model_type": "llama-cpp",
            "hf_file": "x.gguf",
            "n_gpu_layers": -2,
            "on_demand": True,
        })


def test_llama_cpp_rejects_n_ctx_zero():
    with pytest.raises(ValidationError):
        AddModelRequest.model_validate({
            "model_path": "org/repo-GGUF",
            "model_type": "llama-cpp",
            "hf_file": "x.gguf",
            "n_ctx": 0,
            "on_demand": True,
        })


def test_existing_lm_and_embeddings_still_accept():
    AddModelRequest.model_validate({
        "model_path": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "model_type": "lm",
        "on_demand": True,
    })
    AddModelRequest.model_validate({
        "model_path": "nomic-ai/nomic-embed-text-v1.5",
        "model_type": "embeddings",
        "on_demand": True,
    })


def test_invalid_model_type_rejected():
    with pytest.raises(ValidationError):
        AddModelRequest.model_validate({
            "model_path": "x/y",
            "model_type": "whisper",
            "on_demand": True,
        })


# ---------------------------------------------------------------------------
# Phase 4 — dispatcher + handler_type wire advertisement
# ---------------------------------------------------------------------------

def test_llama_cpp_dispatcher_returns_embeddings_handler_type():
    """create_handler_from_config returns a handler advertising handler_type='embeddings'.

    Internal dispatch key is model_type='llama-cpp'; the FUNCTIONAL kind
    advertised on the handler must be 'embeddings' so route guards and
    cli-v2 modelprobe classify correctly without learning a new value per
    runtime.

    Llama() construction is mocked so the test runs without a real GGUF
    file on disk.
    """
    with patch("app.handler.llama_cpp.loader.Llama") as MockLlama:
        MockLlama.from_pretrained.return_value = MagicMock()
        MockLlama.return_value = MagicMock()

        from app.config import ModelEntryConfig
        from app.server import create_handler_from_config

        cfg = ModelEntryConfig(
            model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
            model_type="llama-cpp",
            served_model_name="nomic-ai/nomic-embed-text-v1.5-GGUF",
            hf_file="nomic-embed-text-v1.5.f16.gguf",
            on_demand=True,
        )
        handler = create_handler_from_config(cfg)

        assert handler.handler_type == "embeddings", (
            f"Expected handler_type='embeddings' but got {handler.handler_type!r}. "
            "The dispatcher must advertise the FUNCTIONAL kind, not the runtime kind."
        )


def test_llama_cpp_handler_process_proxy_maps_to_embeddings():
    """HandlerProcessProxy.handler_type maps 'llama-cpp' → 'embeddings'.

    The proxy is the in-process representation of a handler subprocess.
    Its handler_type is read by _get_handler_type() in route guards.
    """
    from app.core.handler_process import HandlerProcessProxy

    proxy = HandlerProcessProxy(
        model_cfg_dict={},
        model_type="llama-cpp",
        model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
        served_model_name="nomic-ai/nomic-embed-text-v1.5-GGUF",
    )

    assert proxy.handler_type == "embeddings", (
        f"Expected HandlerProcessProxy.handler_type='embeddings' for "
        f"model_type='llama-cpp' but got {proxy.handler_type!r}."
    )


def test_valid_model_types_includes_llama_cpp():
    """VALID_MODEL_TYPES frozenset must contain 'llama-cpp'."""
    from app.config import VALID_MODEL_TYPES

    assert "llama-cpp" in VALID_MODEL_TYPES, (
        f"'llama-cpp' missing from VALID_MODEL_TYPES: {VALID_MODEL_TYPES}"
    )


# ---------------------------------------------------------------------------
# Regression: admin POST must pass all llama-cpp fields through model_cfg_dict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_post_llama_cpp_passes_hf_file_to_model_cfg_dict():
    """Regression for the defect where admin POST omitted llama-cpp fields
    from model_cfg_dict, causing on-demand subprocess load to fail with
    hf_file=None.

    The admin route handler calls register_on_demand_one with a
    model_cfg_dict. We mock register_on_demand_one and capture the dict
    it receives to assert all five llama-cpp fields are present.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from fastapi import Request

    from app.schemas.admin import AddModelRequest
    from app.api.admin_models import add_model

    req_body = AddModelRequest(
        model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
        model_type="llama-cpp",
        hf_file="nomic-embed-text-v1.5.f16.gguf",
        n_gpu_layers=-1,
        n_ctx=8192,
        n_batch=512,
        n_threads=4,
        on_demand=True,
        on_demand_idle_timeout=300,
    )

    captured: dict = {}

    async def fake_register_on_demand_one(registry, *, model_id, model_cfg_dict, **kwargs):
        captured["model_cfg_dict"] = model_cfg_dict

    # Build a minimal mock Request with a registry on app.state
    mock_registry = MagicMock()
    mock_registry.list_model_ids.return_value = []
    mock_meta = MagicMock()
    mock_meta.id = req_body.model_path
    mock_meta.type = "embeddings"
    mock_meta.created_at = 0
    mock_registry.get_metadata.return_value = mock_meta

    mock_app = MagicMock()
    mock_app.state.registry = mock_registry
    mock_request = MagicMock(spec=Request)
    mock_request.app = mock_app

    with patch("app.api.admin_models.register_on_demand_one", side_effect=fake_register_on_demand_one):
        await add_model(req_body, mock_request)

    cfg = captured.get("model_cfg_dict", {})
    assert cfg.get("hf_file") == "nomic-embed-text-v1.5.f16.gguf", (
        f"hf_file missing or wrong in model_cfg_dict: {cfg}"
    )
    assert cfg.get("n_gpu_layers") == -1, (
        f"n_gpu_layers missing or wrong in model_cfg_dict: {cfg}"
    )
    assert cfg.get("n_ctx") == 8192, (
        f"n_ctx missing or wrong in model_cfg_dict: {cfg}"
    )
    assert cfg.get("n_batch") == 512, (
        f"n_batch missing or wrong in model_cfg_dict: {cfg}"
    )
    assert cfg.get("n_threads") == 4, (
        f"n_threads missing or wrong in model_cfg_dict: {cfg}"
    )
