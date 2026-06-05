"""``/v1/models`` smoke test: the loaded model must appear in the list.

Uses the session-scoped ``chat_server`` fixture so this test reuses the same
server boot as the chat smoke tests. ``/v1/models`` doesn't drive inference,
so it is unaffected by the MLX stream-affinity bug that xfails the chat
completions tests — this test is expected to pass on its own.
"""

from __future__ import annotations

import httpx
import pytest

from tests.integration.conftest import CHAT_MODEL_ID, requires_apple_silicon


@requires_apple_silicon
@pytest.mark.smoke
@pytest.mark.integration
def test_models_lists_loaded_model(chat_server: tuple[str, int]) -> None:
    """GET /v1/models returns the loaded model id under ``data[].id``."""
    base_url, _ = chat_server
    r = httpx.get(f"{base_url}/v1/models", timeout=10.0)
    assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("object") == "list", f"unexpected top-level object: {body}"
    model_ids = [m.get("id") for m in body.get("data", [])]
    assert CHAT_MODEL_ID in model_ids, f"expected {CHAT_MODEL_ID!r} in /v1/models response, got {model_ids!r}"
