"""Unit tests for LlamaCppEmbeddingsLoader and LlamaCppEmbeddingsService.

Llama() is mocked — these tests assert the loader's CONFIG-vs-CONSTRUCTION
contract (lazy on construction; local-vs-HF dispatch; required-field
validation). Real model loading is covered by smoke tests on Apple Silicon.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from app.handler.llama_cpp.loader import LlamaCppConfig, LlamaCppEmbeddingsLoader


def test_construction_is_lazy_no_llama_called():
    cfg = LlamaCppConfig(
        model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
        hf_file="nomic-embed-text-v1.5.f16.gguf",
    )
    with patch("app.handler.llama_cpp.loader.Llama") as MockLlama:
        loader = LlamaCppEmbeddingsLoader(cfg)
        assert loader.llama is None
        MockLlama.assert_not_called()
        MockLlama.from_pretrained.assert_not_called()


def test_ensure_loaded_uses_from_pretrained_for_hf_repo():
    cfg = LlamaCppConfig(
        model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
        hf_file="nomic-embed-text-v1.5.f16.gguf",
        n_ctx=8192,
    )
    with patch("app.handler.llama_cpp.loader.Llama") as MockLlama:
        MockLlama.from_pretrained.return_value = MagicMock()
        loader = LlamaCppEmbeddingsLoader(cfg)
        loader.ensure_loaded()
        MockLlama.from_pretrained.assert_called_once_with(
            repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
            filename="nomic-embed-text-v1.5.f16.gguf",
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=8192,
        )
        MockLlama.assert_not_called()  # constructor path NOT taken for HF repo


def test_ensure_loaded_uses_constructor_for_local_path(tmp_path):
    local = tmp_path / "model.gguf"
    local.write_bytes(b"")  # presence check only
    cfg = LlamaCppConfig(
        model_path=str(local),
        hf_file=None,
    )
    with patch("app.handler.llama_cpp.loader.Llama") as MockLlama:
        MockLlama.return_value = MagicMock()
        loader = LlamaCppEmbeddingsLoader(cfg)
        loader.ensure_loaded()
        MockLlama.assert_called_once_with(
            model_path=str(local),
            embedding=True,
            n_gpu_layers=-1,
        )
        MockLlama.from_pretrained.assert_not_called()


def test_ensure_loaded_local_path_missing_raises():
    cfg = LlamaCppConfig(model_path="/does/not/exist.gguf", hf_file=None)
    with patch("app.handler.llama_cpp.loader.Llama"):
        loader = LlamaCppEmbeddingsLoader(cfg)
        with pytest.raises(FileNotFoundError, match="local but does not exist"):
            loader.ensure_loaded()


def test_ensure_loaded_hf_repo_missing_hf_file_raises():
    cfg = LlamaCppConfig(
        model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
        hf_file=None,
    )
    with patch("app.handler.llama_cpp.loader.Llama"):
        loader = LlamaCppEmbeddingsLoader(cfg)
        with pytest.raises(ValueError, match="hf_file is required"):
            loader.ensure_loaded()


def test_ensure_loaded_is_idempotent():
    cfg = LlamaCppConfig(
        model_path="nomic-ai/nomic-embed-text-v1.5-GGUF",
        hf_file="nomic-embed-text-v1.5.f16.gguf",
    )
    with patch("app.handler.llama_cpp.loader.Llama") as MockLlama:
        MockLlama.from_pretrained.return_value = MagicMock()
        loader = LlamaCppEmbeddingsLoader(cfg)
        a = loader.ensure_loaded()
        b = loader.ensure_loaded()
        assert a is b
        MockLlama.from_pretrained.assert_called_once()


# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------

from app.handler.llama_cpp.service import LlamaCppEmbeddingsService


def test_service_calls_loader_then_truncates():
    cfg = LlamaCppConfig(
        model_path="org/repo-GGUF",
        hf_file="model.f16.gguf",
    )
    with patch("app.handler.llama_cpp.loader.Llama") as MockLlama:
        mock_llama = MagicMock()
        # Llama.create_embedding returns OpenAI-shape dict
        mock_llama.create_embedding.return_value = {
            "data": [{"embedding": [0.5, 0.5, 0.5, 0.5]}],  # |v|=1
        }
        MockLlama.from_pretrained.return_value = mock_llama

        loader = LlamaCppEmbeddingsLoader(cfg)
        service = LlamaCppEmbeddingsService(loader)

        # Full-dim path
        full = service.embed(["hello"], dimensions=None)
        assert len(full[0]) == 4

        # Matryoshka path
        truncated = service.embed(["hello"], dimensions=2)
        assert len(truncated[0]) == 2
        # Verify it's L2-unit (within float tolerance)
        norm = sum(x * x for x in truncated[0]) ** 0.5
        assert abs(norm - 1.0) < 1e-6
