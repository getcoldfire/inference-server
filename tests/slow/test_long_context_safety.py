"""Task 27a: long-context safety.

Guards against upstream mlx-lm issue #883 — a kernel panic observed at
~58k context length on M3 Ultra. The assertion is "the test process and
server subprocess survive" (HTTP 200 or 503 are both acceptable; OOM with a
clean error is fine; what we won't tolerate is silent process death or a
macOS kernel panic).

Why we use ~16k instead of 50k tokens:

* The smoke fixture model is Llama-3.2-1B-Instruct-4bit, which has a
  declared context window of 128k. Even 16k exercises the long-context
  attention + RoPE-scaling code paths well past the typical "short
  prompt" fast path.
* Pushing to 50k on a 1B model on a typical M-series chip risks OOM on
  the test machine itself (not a regression — just a hostile test).
* The bug we're chasing manifests when the kernel-level Metal scheduler
  gets large allocations queued; 16k allocations are large enough to
  trigger the same code path. If a future test machine has more
  headroom, parametrize the test to push higher.

Token estimation: ~10 chars/token for English. ~16k tokens ≈ 160k chars.

Never runs in CI. Invoked via ``make test-soak``.
"""

from __future__ import annotations

import httpx
import psutil
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon


@requires_apple_silicon
@pytest.mark.slow
def test_long_context_does_not_kernel_panic(
    chat_server: tuple[str, int],
) -> None:
    """Long context returns 200 or 503 (clean) — never process death."""
    base_url, server_pid = chat_server
    proc = psutil.Process(server_pid)

    # ~160k chars ≈ ~16k tokens for English. Well past the short-prompt fast
    # path; still safe for a 1B model on M-series chips.
    long_chunk = "The quick brown fox jumps over the lazy dog. "
    # 3500 reps × ~45 chars ≈ 157,500 chars ≈ ~16k tokens.
    long_input = long_chunk * 3500

    assert proc.is_running(), "server died before test even started"

    try:
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": CHAT_MODEL_ID,
                "messages": [{"role": "user", "content": long_input + "\n\nSummarize."}],
                "max_tokens": 50,
            },
            timeout=600.0,
        )
    except httpx.TransportError as e:
        # Connection-level failure (server died, broken pipe, etc.) is a
        # process-death signal — fail with diagnostic.
        pytest.fail(f"transport error during long-context request (server alive? {proc.is_running()}): {e}")

    # The strict requirement: server is still alive. 200 (handled) or 503
    # (rejected as too large) are both acceptable outcomes; what's not
    # acceptable is the server subprocess dying.
    assert proc.is_running(), f"server subprocess died handling long-context request (HTTP status={r.status_code})"
    assert r.status_code in (200, 503), (
        f"unexpected status code from long-context request: {r.status_code} {r.text[:300]}"
    )
    print(f"\nLong-context safety: ~16k-token request returned {r.status_code}, server still alive.")
