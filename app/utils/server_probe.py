"""Best-effort 'is the fork serving this model?' probe.

Used by:
  - `coldfire-mlx-server models list` (STATUS column, via serving_model_ids)
  - `coldfire-mlx-server models rm` (safety check, via is_model_serving)

Returns False (or empty set) on any error (connection refused, timeout,
non-200, malformed JSON). Defaults to 500ms timeout so callers stay
responsive when no server is running.

Built on stdlib urllib.request to avoid pulling httpx as a top-level
CLI dependency (it's transitive via FastAPI but we don't want to bind
tightly).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def is_model_serving(hf_id: str, port: int = 8000, timeout: float = 0.5) -> bool:
    """Return True iff a running coldfire-mlx-server on 127.0.0.1:<port>
    advertises hf_id in its GET /v1/models response.

    Comparison is against the `id` field of every entry in `data`. If
    the operator gave a model a `served_model_name` alias, the fork's
    /v1/models returns the alias as `id` — callers should pass whichever
    string they want to check against the served set.

    Default port matches the fork's own `launch --port` default (8000).
    cli-v2 operators (whose daemon launches the fork on 11435) should pass
    --port 11435 to `models list`/`models rm`.

    Any error (connection refused, timeout, non-200, malformed JSON)
    returns False without exception.
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False
    for entry in data.get("data", []):
        if entry.get("id") == hf_id:
            return True
    return False


def serving_model_ids(port: int = 8000, timeout: float = 0.5) -> set[str]:
    """Single-shot probe: return the set of model IDs currently advertised
    by a fork on 127.0.0.1:<port>. Empty set on any error.

    Used by `models list` to annotate the STATUS column without making one
    HTTP call per cached row. For per-id checks (e.g. `models rm`'s safety
    check), `is_model_serving` above is the convenience wrapper.

    Default port matches the fork's own `launch --port` default (8000).
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return set()
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return set()
    return {e.get("id") for e in data.get("data", []) if e.get("id")}
