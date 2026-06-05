"""Task 25: 100 sequential completion latency soak.

Verifies that latency doesn't degrade over a sustained sequence of 100
non-streaming completions. Symptom we're guarding against:

* p99 climbs over the run (KV cache growth, lock contention, scheduler
  starvation) — past ~20% degradation is a real problem.

Assertion: ``p99 <= 2 * p50 + 1.0s``. The constant tolerance handles small
absolute jitter on fast machines where p50 may be ~0.3s; the 2x multiplier
catches degradation patterns regardless of absolute speed.

Also asserts RSS growth stays bounded (no slow leak across the 100-request
window). Uses the session-scoped chat_server fixture so the model is
already warm when the test starts.
"""

from __future__ import annotations

import time

import httpx
import psutil
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon


@requires_apple_silicon
@pytest.mark.slow
def test_100_sequential_completions(chat_server: tuple[str, int]) -> None:
    """Run 100 sequential non-streaming completions; assert latency stays stable."""
    base_url, server_pid = chat_server
    proc = psutil.Process(server_pid)
    baseline_rss = proc.memory_info().rss

    latencies: list[float] = []
    with httpx.Client(timeout=120.0) as c:
        for i in range(100):
            assert proc.is_running(), f"server died at request {i}"
            start = time.monotonic()
            r = c.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": CHAT_MODEL_ID,
                    "messages": [
                        {"role": "user", "content": f"Count to {i % 10 + 1}."}
                    ],
                    "max_tokens": 40,
                },
            )
            assert r.status_code == 200, (
                f"non-200 at request {i}: {r.status_code} {r.text[:200]}"
            )
            latencies.append(time.monotonic() - start)

    final_rss = proc.memory_info().rss
    rss_growth_mb = (final_rss - baseline_rss) / (1024 * 1024)

    sorted_latencies = sorted(latencies)
    # 100 samples — index 50 is roughly p50 (median-ish), index 99 is p99
    p50 = sorted_latencies[50]
    p99 = sorted_latencies[99]
    p_min = sorted_latencies[0]
    p_max = sorted_latencies[-1]

    print(
        f"\nSequential soak: 100 completions. "
        f"latency p50={p50:.3f}s p99={p99:.3f}s "
        f"min={p_min:.3f}s max={p_max:.3f}s. "
        f"RSS growth: {rss_growth_mb:.1f} MB."
    )

    # Latency stability: p99 should not be more than 2x the median + 1s of jitter
    assert p99 <= 2 * p50 + 1.0, (
        f"latency degradation: p99 ({p99:.3f}s) > 2 * p50 ({p50:.3f}s) + 1.0s"
    )
    # Memory: no obvious leak across 100 requests
    assert rss_growth_mb < 100, (
        f"memory growth across 100 requests: {rss_growth_mb:.1f} MB"
    )
