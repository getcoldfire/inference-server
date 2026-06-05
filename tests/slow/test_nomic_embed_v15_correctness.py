"""Task 26a: nomic-embed-text-v1.5 correctness vs Sentence-Transformers.

Embeds a 100-sentence corpus with our MLX-served implementation and the
canonical PyTorch reference (``sentence-transformers`` via HuggingFace) and
asserts per-document cosine similarity >= 0.999.

Why 0.999 (not 1.000): float32 vs MLX dtype rounding accumulates across
attention + softmax + pooling. 4-decimal agreement is what comparable
correctness suites (e.g. mlx-examples bert tests) use as the threshold.

This test is ``@slow`` because:

* loading the reference model pulls ~280MB from HF and instantiates a
  torch model with ``trust_remote_code=True`` (nomic-bert's custom modeling
  code lives in the HF repo, not in upstream transformers).
* the comparison is meaningful only if it survives over the full corpus,
  not just a smoke single-input.

Never runs in CI. Invoked via ``make test-soak``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pytest

from tests.integration.conftest import EMBEDDING_MODEL_ID, requires_apple_silicon

_CORPUS_PATH = Path(__file__).parent / "reference_corpus.txt"


@requires_apple_silicon
@pytest.mark.slow
def test_nomic_embed_v15_matches_reference(
    embedding_server: tuple[str, int],
) -> None:
    """Per-doc cosine similarity vs Sentence-Transformers reference >= 0.999."""
    # Lazy import — sentence-transformers is a heavy dep (torch + transformers)
    # we only want loaded for slow tests.
    from sentence_transformers import SentenceTransformer

    base_url, _ = embedding_server
    corpus = [line for line in _CORPUS_PATH.read_text().splitlines() if line.strip()]
    assert len(corpus) >= 50, f"corpus too small: {len(corpus)} sentences"

    # Reference: torch-based nomic-bert on CPU. trust_remote_code=True is
    # required — nomic ships its modeling code in the HF repo.
    ref_model = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        trust_remote_code=True,
        device="cpu",
    )
    ref_vectors = ref_model.encode(corpus, normalize_embeddings=True)
    ref_vectors = np.asarray(ref_vectors)

    # Our server. POST in a single request — service does internal batching.
    r = httpx.post(
        f"{base_url}/v1/embeddings",
        json={"model": EMBEDDING_MODEL_ID, "input": corpus},
        timeout=120.0,
    )
    assert r.status_code == 200, f"embeddings request failed: {r.text[:500]}"
    our_vectors = np.array([d["embedding"] for d in r.json()["data"]])

    assert our_vectors.shape == ref_vectors.shape, f"shape mismatch: ours {our_vectors.shape} ref {ref_vectors.shape}"
    # Both vectors L2-normalized; cosine = dot product
    cosines = (our_vectors * ref_vectors).sum(axis=1)
    min_cos = float(cosines.min())
    mean_cos = float(cosines.mean())
    print(f"\nnomic-embed-text-v1.5 vs reference: {len(corpus)} docs, min_cos={min_cos:.5f} mean_cos={mean_cos:.5f}")
    bad = [(i, float(c), corpus[i][:60]) for i, c in enumerate(cosines) if c < 0.999]
    assert not bad, f"{len(bad)} docs below 0.999 cosine threshold; first 5: {bad[:5]}"
