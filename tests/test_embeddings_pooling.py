"""Unit tests for app.handler.embeddings.pooling.{apply_pooling, l2_normalize}.

Pooling correctness drives downstream embedding-vector quality, so each
mode is exercised on a hand-rolled tensor where the expected output is
trivially computable in this file.
"""

import math

import mlx.core as mx
import pytest

from app.handler.embeddings.pooling import apply_pooling, l2_normalize


def test_mean_pooling_respects_attention_mask():
    """Padded tokens (mask=0) must be excluded from the mean."""
    # (batch=1, seq=3, hidden=2)
    hidden_states = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = mx.array([[1, 1, 0]])
    result = apply_pooling(hidden_states, mask, "mean")
    assert result.shape == (1, 2)
    # mean of [1,2] and [3,4] = [2, 3]; the [5,6] position is masked out.
    assert mx.allclose(result, mx.array([[2.0, 3.0]]))


def test_mean_pooling_handles_all_masked():
    """When every token is padding the divisor clamps to 1 (avoid div-by-zero)."""
    hidden_states = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = mx.array([[0, 0]])
    result = apply_pooling(hidden_states, mask, "mean")
    # summed = [4, 6], counts clamped to 1 → result is [4, 6]
    # The actual numeric value doesn't matter much — what matters is "no NaN".
    assert mx.all(mx.isfinite(result)).item()


def test_cls_pooling():
    """CLS pooling returns the first-token hidden state unchanged."""
    hidden_states = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = mx.array([[1, 1]])
    result = apply_pooling(hidden_states, mask, "cls")
    assert mx.allclose(result, mx.array([[1.0, 2.0]]))


def test_last_token_pooling():
    """last_token pooling returns the hidden state at the LAST unmasked position."""
    # batch=1: seq has 3 tokens, last 1 is padding -> last real token at index 1
    hidden_states = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = mx.array([[1, 1, 0]])
    result = apply_pooling(hidden_states, mask, "last_token")
    assert mx.allclose(result, mx.array([[3.0, 4.0]]))


def test_max_pooling_respects_mask():
    """Padded positions must not win the per-channel max."""
    # The padded token has the largest values; max pooling must NOT pick them.
    hidden_states = mx.array([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = mx.array([[1, 1, 0]])
    result = apply_pooling(hidden_states, mask, "max")
    # Among unmasked positions per-channel: max([1,3]) = 3, max([2,4]) = 4
    assert mx.allclose(result, mx.array([[3.0, 4.0]]))


def test_unknown_pooling_raises():
    hidden_states = mx.array([[[1.0]]])
    mask = mx.array([[1]])
    with pytest.raises(ValueError, match="unknown pooling mode"):
        apply_pooling(hidden_states, mask, "weird")


def test_l2_normalize_unit_norm():
    """After l2_normalize each row has unit L2 norm."""
    x = mx.array([[3.0, 4.0], [1.0, 0.0]])
    out = l2_normalize(x)
    # ||[3,4]|| = 5 → [0.6, 0.8]
    # ||[1,0]|| = 1 → [1, 0]
    assert mx.allclose(out, mx.array([[0.6, 0.8], [1.0, 0.0]]), atol=1e-6)
    # Each row should now have norm 1.
    norms = mx.sqrt((out * out).sum(axis=-1))
    assert mx.allclose(norms, mx.array([1.0, 1.0]), atol=1e-6)


def test_l2_normalize_handles_zero_vector():
    """A pure zero vector should not produce NaN (eps clamp inside l2_normalize)."""
    x = mx.array([[0.0, 0.0]])
    out = l2_normalize(x)
    assert mx.all(mx.isfinite(out)).item()


def test_mean_pooling_no_padding_matches_simple_mean():
    """Sanity: with all-ones mask, mean pooling == np mean along seq."""
    hidden_states = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = mx.array([[1, 1, 1]])
    result = apply_pooling(hidden_states, mask, "mean")
    expected = mx.array([[(1.0 + 3.0 + 5.0) / 3, (2.0 + 4.0 + 6.0) / 3]])
    assert mx.allclose(result, expected, atol=1e-6)


def test_l2_normalize_value():
    """Cross-check the closed-form norm on a known input."""
    x = mx.array([[1.0, 2.0, 2.0]])
    out = l2_normalize(x)
    n = math.sqrt(1.0 + 4.0 + 4.0)  # = 3
    expected = mx.array([[1.0 / n, 2.0 / n, 2.0 / n]])
    assert mx.allclose(out, expected, atol=1e-6)
