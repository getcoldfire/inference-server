"""Pydantic-level tests for the AddModelRequest llama-cpp branch."""
import pytest
from pydantic import ValidationError

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
