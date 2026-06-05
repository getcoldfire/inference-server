"""Regression tests for upstream issue #307: handler subprocess robust to TimeoutError.

The dispatch path in ``handler_process`` runs each inbound request as an
``asyncio.Task`` under ``_handle_request``. Before this patch, an
``asyncio.TimeoutError`` raised by the handler method was caught by the generic
``except Exception`` branch and reported with ``status_code=500``. That's
wrong in two ways:

  - It conflates a known-recoverable condition (the upstream request took too
    long) with an unknown server crash.
  - The OpenAI API convention is to use 504 Gateway Timeout for upstream
    timeouts, which lets clients (cli-v2, gateways, etc.) treat them as a
    retriable failure mode rather than a "panic and crash" signal.

Issue #307 was triggered when the handler subprocess saw a ``TimeoutError``
and the error path's logging / response shape was thin enough that downstream
saw "handler crashed" symptoms (no readable status, often connection close).

The patch adds a dedicated ``except TimeoutError`` arm that:
  - logs at WARNING (not ERROR) — the request timed out, the process is fine,
  - reports ``status_code=504``,
  - keeps the subprocess alive and ready for the next request.

These tests verify the behavior of ``_handle_request`` against a stubbed
handler+queue. They do NOT spawn a real subprocess (too expensive for a
unit test) but exercise the exact code path on the handler side.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.handler_process import _handle_request_for_test


class _CaptureQueue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


@pytest.mark.asyncio
async def test_timeout_error_is_reported_as_504_not_500():
    """When a handler method raises ``TimeoutError``, the response queue must
    receive a 504 — not 500 — and the dispatcher must not let the subprocess
    crash."""
    response_queue = _CaptureQueue()

    class _SlowHandler:
        async def chat(self, *_args, **_kwargs):
            raise TimeoutError("simulated upstream timeout")

    request = {
        "id": "req-timeout-1",
        "method": "chat",
        "args": (),
        "kwargs": {},
        "stream": False,
    }

    # Should not raise — the patch must keep the handler-loop alive.
    await _handle_request_for_test(_SlowHandler(), response_queue, request)

    # Exactly one error message on the queue
    error_messages = [m for m in response_queue.items if m.get("type") == "error"]
    assert len(error_messages) == 1, response_queue.items
    err = error_messages[0]
    assert err["id"] == "req-timeout-1"
    assert err["status_code"] == 504, (
        f"TimeoutError should map to 504, got {err['status_code']}; "
        "the handler dispatcher is treating timeout as a generic crash."
    )
    # error_type identifies it specifically
    assert "Timeout" in err["error_type"]


@pytest.mark.asyncio
async def test_timeout_error_does_not_kill_dispatcher():
    """After a TimeoutError is handled, the dispatcher must accept and serve
    the next request normally — i.e. the exception was caught, not allowed to
    propagate."""
    response_queue = _CaptureQueue()

    class _SometimesSlowHandler:
        def __init__(self):
            self.call_count = 0

        async def chat(self, *_args, **_kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise TimeoutError("first call times out")
            return {"ok": True, "call": self.call_count}

    handler = _SometimesSlowHandler()

    req_1 = {"id": "req-1", "method": "chat", "args": (), "kwargs": {}, "stream": False}
    await _handle_request_for_test(handler, response_queue, req_1)

    req_2 = {"id": "req-2", "method": "chat", "args": (), "kwargs": {}, "stream": False}
    await _handle_request_for_test(handler, response_queue, req_2)

    # First request: 504 error
    first = response_queue.items[0]
    assert first["type"] == "error"
    assert first["status_code"] == 504
    # Second request: normal result
    second = response_queue.items[1]
    assert second["type"] == "result"
    assert second["value"] == {"ok": True, "call": 2}


@pytest.mark.asyncio
async def test_non_timeout_exception_still_500():
    """Non-timeout errors must still report ``status_code=500`` so we don't
    silently downgrade other crashes to 504."""
    response_queue = _CaptureQueue()

    class _BrokenHandler:
        async def chat(self, *_args, **_kwargs):
            raise RuntimeError("totally broken")

    req = {"id": "req-boom", "method": "chat", "args": (), "kwargs": {}, "stream": False}
    await _handle_request_for_test(_BrokenHandler(), response_queue, req)

    err = response_queue.items[0]
    assert err["type"] == "error"
    assert err["status_code"] == 500
    assert err["error_type"] == "RuntimeError"
