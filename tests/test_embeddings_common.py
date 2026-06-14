# tests/test_embeddings_common.py
"""Unit tests for the shared embeddings post-processing helper.

The helper exists so the BERT-encoder embedding handler and the llama-cpp
embedding handler apply matryoshka truncation + L2 renormalization identically.
A regression here is a silent-correctness bug — both handlers' wire output
drift apart.
"""
import math
import pytest
import numpy as np

from app.handler.embeddings_common import apply_dimensions


def _l2_norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def test_target_equal_to_input_is_pass_through():
    """When target_dim == len(vec), helper still renormalizes; output equals
    input only because input is unit-norm to begin with."""
    vec = np.array([0.6, 0.8], dtype=np.float32)
    out = apply_dimensions(vec, 2)
    # Already L2-unit; should round-trip unchanged within float tolerance.
    assert list(out) == pytest.approx([0.6, 0.8], abs=1e-6)


def test_truncates_and_renormalizes():
    # Pre-L2-unit 4-vec. Truncate to 2 -> renormalize -> still L2-unit.
    vec = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32) / 2.0  # |v|=1
    out = apply_dimensions(vec, 2)
    assert len(out) == 2
    assert _l2_norm(out) == pytest.approx(1.0, abs=1e-6)
    # Each component should be 1/sqrt(2)
    assert out[0] == pytest.approx(1.0 / math.sqrt(2), abs=1e-6)


def test_target_greater_than_input_raises():
    vec = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds"):
        apply_dimensions(vec, 5)


def test_target_less_than_one_raises():
    vec = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        apply_dimensions(vec, 0)
    with pytest.raises(ValueError, match="positive"):
        apply_dimensions(vec, -1)


def test_zero_vector_truncated_returns_zeros_no_division_by_zero():
    # Edge case: caller passes a zero vector. We truncate but cannot
    # renormalize. Return zeros rather than blow up.
    vec = np.zeros(4, dtype=np.float32)
    out = apply_dimensions(vec, 2)
    assert list(out) == [0.0, 0.0]


def test_matches_pre_refactor_bert_output_on_fixed_input():
    """Refactor-safety guard: the helper produces the same output as the BERT
    path's pre-refactor inline truncation logic on a fixed input.

    Uses pytest.approx rather than byte equality because the pre-refactor path
    used MLX's l2_normalize with eps=1e-12 while the new helper uses numpy
    np.linalg.norm without eps; the values differ by up to ~1 ULP at fp32 on
    arbitrary inputs. On this specific input ([0.5, 0.5, 0.5, 0.5] → dim=2) the
    outputs happen to be byte-identical at fp32, but the assertion uses approx
    to remain semantically correct."""
    vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    out = apply_dimensions(vec, 2)
    assert list(out) == pytest.approx([0.70710677, 0.70710677], abs=1e-7)


def test_accepts_2d_batch_input():
    """2-D input of shape (batch, dim) is truncated + renormalized per row."""
    batch = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 2.0, 0.0],
    ], dtype=np.float32)
    out = apply_dimensions(batch, 2)
    assert out.shape == (2, 2)
    for row in out:
        n = math.sqrt(sum(x * x for x in row))
        assert n == pytest.approx(1.0, abs=1e-6)


def test_2d_zero_row_returns_zeros_in_place():
    """A 2-D batch where one row is all-zero: that row returns zeros; others normalize."""
    batch = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
    ], dtype=np.float32)
    out = apply_dimensions(batch, 2)
    assert list(out[0]) == [0.0, 0.0]
    n = math.sqrt(sum(x * x for x in out[1]))
    assert n == pytest.approx(1.0, abs=1e-6)
