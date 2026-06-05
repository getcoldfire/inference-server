"""Task 24: 1-hour sustained streaming soak.

Verifies that streaming chat completions can run for an hour without:

* crashing the server subprocess (kernel panic, OOM kill, segfault)
* leaking memory beyond a calibrated RSS-growth ceiling

The RSS ceiling starts at 200 MB above baseline — this is an initial,
intentionally generous target. Once we have a real baseline number from the
first successful run, tighten to baseline + 50 MB.

Runs against the session-scoped ``chat_server`` fixture (re-exported from
``tests/integration/conftest.py``). PID is read from the fixture's tuple so
we attach ``psutil`` directly to the server subprocess, not the test
process.

This test is ``@slow`` — never runs in CI. Invoked via ``make test-soak``.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import psutil
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon


@requires_apple_silicon
@pytest.mark.slow
@pytest.mark.asyncio
async def test_one_hour_streaming_soak(chat_server: tuple[str, int]) -> None:
    """Stream chat completions for 1 hour; assert process alive + RSS bounded."""
    base_url, server_pid = chat_server
    proc = psutil.Process(server_pid)

    # Warm-up to settle RSS — first few completions allocate buffers etc.
    async with httpx.AsyncClient(timeout=120.0) as c:
        for _ in range(3):
            async with c.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json={
                    "model": CHAT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 32,
                    "stream": True,
                },
            ) as r:
                async for _ in r.aiter_lines():
                    pass

    baseline_rss = proc.memory_info().rss
    max_rss = baseline_rss
    requests_made = 0
    soak_duration_seconds = 3600  # 1 hour
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=300.0) as c:
        while time.monotonic() - start < soak_duration_seconds:
            assert proc.is_running(), "server subprocess died during soak"
            async with c.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json={
                    "model": CHAT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Tell a 100-word story."}],
                    "max_tokens": 150,
                    "stream": True,
                },
            ) as r:
                assert r.status_code == 200, f"non-200 during soak after {requests_made} requests: {r.status_code}"
                async for _ in r.aiter_lines():
                    pass
            requests_made += 1
            try:
                max_rss = max(max_rss, proc.memory_info().rss)
            except psutil.NoSuchProcess:
                pytest.fail(f"server died after {requests_made} requests")
            # Avoid hammering — yield to event loop, keep soak realistic
            await asyncio.sleep(0)

    rss_growth_mb = (max_rss - baseline_rss) / (1024 * 1024)
    print(
        f"\nStreaming soak: {requests_made} requests in "
        f"{soak_duration_seconds}s. "
        f"Baseline RSS: {baseline_rss / 1024 / 1024:.1f} MB. "
        f"Peak RSS: {max_rss / 1024 / 1024:.1f} MB. "
        f"Growth: {rss_growth_mb:.1f} MB."
    )
    # Initial calibration ceiling — tighten to ~50 MB once a real baseline run
    # has been observed. The point of this test is to catch unbounded growth,
    # not to enforce tight memory budgets.
    assert rss_growth_mb < 200, f"memory leak suspected: RSS grew {rss_growth_mb:.1f} MB during 1h soak"
