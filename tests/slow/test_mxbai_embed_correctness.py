"""Task 26b: mxbai-embed-large-v1 correctness vs Sentence-Transformers.

Companion to ``test_nomic_embed_v15_correctness.py``. Same shape: 100-doc
corpus, same per-doc cosine >= 0.999 assertion. The difference is the model
family — mxbai-embed-large-v1 is a vanilla BERT (absolute positions + GeLU
MLP), not RoPE+SwiGLU like nomic. Together the two tests cover both encoder
variants implemented in ``app/handler/embeddings/encoder.py``.

Model id preference: try ``mlx-community/mxbai-embed-large-v1`` first; if
not present on HF, fall back to ``mixedbread-ai/mxbai-embed-large-v1`` (the
canonical upstream). Both serve the same weights.

Never runs in CI. Invoked via ``make test-soak``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest

from tests.integration.conftest import (
    _boot_server,
    _teardown_server,
    _wait_for_healthz,
    requires_apple_silicon,
)

_CORPUS_PATH = Path(__file__).parent / "reference_corpus.txt"

# Try mlx-community first (matches Coldfire convention); fall back to upstream.
_MXBAI_CANDIDATES = [
    "mlx-community/mxbai-embed-large-v1",
    "mixedbread-ai/mxbai-embed-large-v1",
]


def _resolve_mxbai_model() -> str:
    """Pick the first mxbai repo id available on HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()
    for candidate in _MXBAI_CANDIDATES:
        try:
            api.model_info(candidate)
            return candidate
        except Exception:  # noqa: BLE001 — any HF failure → try next
            continue
    pytest.skip(f"none of {_MXBAI_CANDIDATES} reachable on HuggingFace; cannot run mxbai correctness")


@pytest.fixture(scope="module")
def mxbai_server() -> Iterator[tuple[str, int, str]]:
    """Boot one mxbai-embed server per module; yield ``(url, pid, model_id)``."""
    model_id = _resolve_mxbai_model()
    proc, port, log_path = _boot_server(model_id, model_type="embeddings")
    try:
        ready = _wait_for_healthz(port, proc, timeout=300.0)
        if not ready:
            out = log_path.read_text(errors="replace") if log_path.exists() else "(no log)"
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            pytest.fail(f"mxbai server failed to become healthy on :{port}.\nOutput:\n{out}")
        yield (f"http://127.0.0.1:{port}", proc.pid, model_id)
    finally:
        _teardown_server(proc)
        log_path.unlink(missing_ok=True)


@requires_apple_silicon
@pytest.mark.slow
def test_mxbai_embed_matches_reference(
    mxbai_server: tuple[str, int, str],
) -> None:
    """Per-doc cosine similarity vs Sentence-Transformers reference >= 0.999."""
    from sentence_transformers import SentenceTransformer

    base_url, _, model_id = mxbai_server
    corpus = [line for line in _CORPUS_PATH.read_text().splitlines() if line.strip()]
    assert len(corpus) >= 50, f"corpus too small: {len(corpus)} sentences"

    # Reference is always the canonical upstream regardless of which mlx repo
    # we used for the server (they share the same underlying weights).
    ref_model = SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1",
        device="cpu",
    )
    ref_vectors = np.asarray(ref_model.encode(corpus, normalize_embeddings=True))

    r = httpx.post(
        f"{base_url}/v1/embeddings",
        json={"model": model_id, "input": corpus},
        timeout=120.0,
    )
    assert r.status_code == 200, f"embeddings request failed: {r.text[:500]}"
    our_vectors = np.array([d["embedding"] for d in r.json()["data"]])

    assert our_vectors.shape == ref_vectors.shape, f"shape mismatch: ours {our_vectors.shape} ref {ref_vectors.shape}"
    cosines = (our_vectors * ref_vectors).sum(axis=1)
    min_cos = float(cosines.min())
    mean_cos = float(cosines.mean())
    print(
        f"\nmxbai-embed-large-v1 ({model_id}) vs reference: "
        f"{len(corpus)} docs, min_cos={min_cos:.5f} mean_cos={mean_cos:.5f}"
    )
    bad = [(i, float(c), corpus[i][:60]) for i, c in enumerate(cosines) if c < 0.999]
    assert not bad, f"{len(bad)} docs below 0.999 cosine threshold; first 5: {bad[:5]}"
