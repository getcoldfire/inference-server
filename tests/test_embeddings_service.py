"""Integration tests for app.handler.embeddings.service.EmbeddingService.

Loads the tiny_bert fixture as a complete embedding model, runs real text
inputs through it, and verifies the output respects the documented
contract: L2-normalized, deterministic, and tracks token usage.
"""

import math
from pathlib import Path

import pytest

from app.handler.embeddings.service import EmbeddingService

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_bert"


@pytest.fixture(scope="module")
def service():
    """Module-scoped service to avoid reloading the model per test."""
    return EmbeddingService(model_path=str(FIXTURE))


def test_single_input(service):
    """One text in -> one vector out with the fixture's hidden_size (32)."""
    result = service.embed(["hello world"])
    assert len(result.embeddings) == 1
    vec = result.embeddings[0]
    assert len(vec) == 32  # tiny_bert hidden_size

    # L2-normalized -> ||vec|| ~= 1.
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5


def test_batch_input(service):
    """Three texts in -> three vectors out, each unit-norm."""
    result = service.embed(["short", "a slightly longer input", "third"])
    assert len(result.embeddings) == 3
    for vec in result.embeddings:
        assert len(vec) == 32
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5


def test_batch_input_deterministic(service):
    """Running the same inputs twice yields bitwise-equal outputs.

    Embeddings models are pure (no sampling), so repeated calls on the
    same model + input must match. This catches any accidental hidden
    state leakage (e.g. caching the wrong thing across calls).
    """
    r1 = service.embed(["a", "b", "c"])
    r2 = service.embed(["a", "b", "c"])
    for v1, v2 in zip(r1.embeddings, r2.embeddings, strict=True):
        for a, b in zip(v1, v2, strict=True):
            assert abs(a - b) < 1e-6


def test_usage_counts(service):
    """prompt_tokens equals total_tokens and reflects the non-pad token count."""
    result = service.embed(["short", "a slightly longer input"])
    assert result.prompt_tokens > 0
    assert result.total_tokens == result.prompt_tokens
    # Two inputs always produce at least 2 tokens (one per sequence).
    assert result.prompt_tokens >= 2


def test_empty_batch_returns_empty(service):
    """No inputs -> empty embeddings list, zero tokens."""
    result = service.embed([])
    assert result.embeddings == []
    assert result.prompt_tokens == 0
    assert result.total_tokens == 0


def test_dimensions_arg_overrides_native_dim(service):
    """Per-request `dimensions` truncates the output vector to that size.

    Mirrors the OpenAI API's `dimensions` field on `/v1/embeddings`. Truncation
    happens before L2 normalize so the returned vector stays unit-norm.
    """
    result = service.embed(["hello"], dimensions=16)
    assert len(result.embeddings[0]) == 16
    norm = math.sqrt(sum(x * x for x in result.embeddings[0]))
    assert abs(norm - 1.0) < 1e-5


def test_dimensions_arg_rejects_out_of_range(service):
    """`dimensions=0` and `dimensions>hidden_size` both raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        service.embed(["hello"], dimensions=0)
    with pytest.raises(ValueError, match="out of range"):
        service.embed(["hello"], dimensions=999)


def test_dimensions_arg_overrides_model_matryoshka(tmp_path):
    """Per-request `dimensions` beats the model-config `matryoshka_dim`.

    OpenAI semantics: the request field always wins. Config says 16, request
    says 8 -> output is 8-dim.
    """
    import json
    import shutil

    fixture_copy = tmp_path / "tiny_bert_matryoshka_override"
    shutil.copytree(FIXTURE, fixture_copy)
    cfg = json.loads((fixture_copy / "config.json").read_text())
    cfg["matryoshka_dim"] = 16
    (fixture_copy / "config.json").write_text(json.dumps(cfg))

    svc = EmbeddingService(model_path=str(fixture_copy))
    result = svc.embed(["hello"], dimensions=8)
    assert len(result.embeddings[0]) == 8
    norm = math.sqrt(sum(x * x for x in result.embeddings[0]))
    assert abs(norm - 1.0) < 1e-5


def test_matryoshka_truncates_before_normalize(tmp_path):
    """When matryoshka_dim is set in config.json, embeddings are truncated
    to that dim BEFORE L2 normalization.

    Truncating after normalization would leave the truncated vector with
    norm < 1; truncating before guarantees the returned vector is still
    unit-norm.
    """
    import json
    import shutil

    fixture_copy = tmp_path / "tiny_bert_matryoshka"
    shutil.copytree(FIXTURE, fixture_copy)
    cfg = json.loads((fixture_copy / "config.json").read_text())
    cfg["matryoshka_dim"] = 16
    (fixture_copy / "config.json").write_text(json.dumps(cfg))

    svc = EmbeddingService(model_path=str(fixture_copy))
    result = svc.embed(["hello"])
    assert len(result.embeddings[0]) == 16
    norm = math.sqrt(sum(x * x for x in result.embeddings[0]))
    assert abs(norm - 1.0) < 1e-5
