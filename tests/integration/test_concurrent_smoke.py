"""Concurrent chat smoke: 4 simultaneous streams must not cross-contaminate.

Sends 4 concurrent ``/v1/chat/completions`` requests, each asking the model
to repeat a distinct marker string. Each response MUST contain its own
marker and NONE of the others — any leakage indicates the BatchScheduler's
KV cache is bleeding generated tokens between sequences sharing a batch.

Markers use a distinctive base32-ish alphabet (no spaces, no model-vocab
sub-tokens) so substring matching is unambiguous. Budget is generous
(120s) because 4 concurrent streams on the 1B model on a single GPU
serializes through one BatchGenerator and runs ~2-3x slower per stream
than a solo run.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon

# Use a distinct marker per request. Capital-letter + digit groups avoid
# accidental in-text collisions and aren't subwords of common English
# tokens. Each one is unique by construction.
#
# Markers intentionally have no underscores or other punctuation Llama tends
# to elide when reproducing tokens verbatim — the KV-contamination check is
# what we care about, and a brittle "exact substring" test would noise it.
_MARKERS = ["MARKER42XYZABC123", "TOKEN77ABCDEF456", "BEACON15LMNGHI789", "SIGIL33OPQJKL012"]


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_no_kv_contamination(chat_server: tuple[str, int]) -> None:
    """4 concurrent chat completions: each response carries only its own marker."""
    base_url, _ = chat_server

    async def _one(client: httpx.AsyncClient, marker: str) -> str:
        r = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": CHAT_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": (f"Repeat the following token exactly once, with no other words: {marker}"),
                    },
                ],
                "max_tokens": 32,
                "temperature": 0.0,
            },
            timeout=120.0,
        )
        assert r.status_code == 200, f"marker={marker}: status {r.status_code}: {r.text}"
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        return content or ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        responses = await asyncio.gather(*(_one(client, m) for m in _MARKERS))

    for marker, text in zip(_MARKERS, responses, strict=True):
        assert marker in text, f"missing own marker {marker!r} in response: {text!r}"
        for other in _MARKERS:
            if other == marker:
                continue
            assert other not in text, (
                f"KV contamination: marker {other!r} from a sibling request leaked into "
                f"response for {marker!r}: {text!r}"
            )
