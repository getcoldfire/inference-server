"""Chat-completion smoke tests against a real Llama-3.2-1B-Instruct-4bit server.

Two tests share the session-scoped `chat_server` fixture from conftest:

* `test_non_streaming_completion`: POST `/v1/chat/completions` with `stream=False`,
  assert HTTP 200 and a non-empty assistant message.
* `test_streaming_completion`: POST with `stream=True`, parse the SSE chunks
  and assert at least one valid `choices[0].delta` payload arrives before
  the `[DONE]` sentinel.

The fixture boots the server once and reuses it; first run can take 5+ minutes
on a cold HuggingFace cache.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_non_streaming_completion(chat_server: tuple[str, int]) -> None:
    """Non-streaming chat completion returns a non-empty assistant message."""
    base_url, _ = chat_server
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": CHAT_MODEL_ID,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text}"
    body = r.json()
    assert "choices" in body, f"response missing 'choices': {body}"
    assert body["choices"], "empty choices list"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"].strip(), f"empty assistant content: {msg!r}"


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_streaming_completion(chat_server: tuple[str, int]) -> None:
    """Streaming chat completion emits valid SSE chunks ending with [DONE]."""
    base_url, _ = chat_server
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": CHAT_MODEL_ID,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        },
        timeout=120.0,
    ) as r:
        assert r.status_code == 200, f"unexpected status {r.status_code}"
        chunks = []
        saw_done = False
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))

    assert saw_done, "stream did not terminate with [DONE]"
    assert chunks, "no SSE chunks received before [DONE]"
    assert "choices" in chunks[0], f"first chunk missing 'choices': {chunks[0]!r}"
    # At least one chunk must carry actual text content in delta.content.
    seen_text = any((c.get("choices") or [{}])[0].get("delta", {}).get("content") for c in chunks)
    assert seen_text, f"no delta.content in any of {len(chunks)} chunks"
