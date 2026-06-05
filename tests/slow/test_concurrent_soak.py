"""Task 27b: 15-minute 8-stream concurrent soak.

Drives 8 concurrent streaming chat completions for 15 minutes, each with a
unique marker. Asserts no KV cache cross-contamination: each stream's
response must contain its own marker and no marker from another stream.

This is the long-duration counterpart to the 4-request smoke version in
``tests/integration/test_concurrent_smoke.py``. The smoke proves the
property momentarily; this soak proves it stays true under sustained
concurrent load, where KV-cache reuse, scheduler races, and
event-loop starvation are most likely to surface.

Concurrency note: the upstream server's ``--max-concurrency`` defaults to
something ≥ 8 (see Phase 5 CLI flag work). If a future build lowers it,
this test will queue rather than parallelize — still correct, just slower.

Never runs in CI. Invoked via ``make test-soak``.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon


@requires_apple_silicon
@pytest.mark.slow
@pytest.mark.asyncio
async def test_8_concurrent_streams_15_minutes(
    chat_server: tuple[str, int],
) -> None:
    """8 streams × 15 min; each response carries only its own marker."""
    base_url, _ = chat_server
    soak_duration_seconds = 15 * 60
    n_streams = 8
    markers = [f"MARKER{i}ZZZ" for i in range(n_streams)]
    end_at = time.monotonic() + soak_duration_seconds

    contamination_events: list[tuple[str, str, str]] = []
    completed_counts = [0] * n_streams

    async def stream_loop(idx: int) -> None:
        own_marker = markers[idx]
        foreign_markers = [m for m in markers if m != own_marker]
        async with httpx.AsyncClient(timeout=120.0) as c:
            while time.monotonic() < end_at:
                text = ""
                async with c.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": CHAT_MODEL_ID,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Repeat exactly this token and nothing else: {own_marker}",
                            }
                        ],
                        "max_tokens": 24,
                        "stream": True,
                    },
                ) as r:
                    assert r.status_code == 200, f"stream {idx} got status {r.status_code}"
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            text += line
                # Own marker must appear
                if own_marker not in text:
                    contamination_events.append((own_marker, "missing-own", text[:200]))
                # No foreign marker may appear
                for foreign in foreign_markers:
                    if foreign in text:
                        contamination_events.append((own_marker, f"leaked:{foreign}", text[:200]))
                completed_counts[idx] += 1

    await asyncio.gather(*[stream_loop(i) for i in range(n_streams)])

    total = sum(completed_counts)
    print(
        f"\nConcurrent soak: {total} completions across {n_streams} streams "
        f"in {soak_duration_seconds}s. "
        f"Per-stream counts: {completed_counts}. "
        f"Contamination events: {len(contamination_events)}."
    )
    assert not contamination_events, (
        f"{len(contamination_events)} KV-isolation failures; first 5: {contamination_events[:5]}"
    )
