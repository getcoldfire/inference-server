"""Regression test for upstream issue #302: mid-stream client disconnect.

`handle_stream_response` is the chat-completions SSE wrapper that consumes the
handler's `generate_text_stream` async generator and yields OpenAI-formatted
SSE chunks. Before this patch it iterated the inner generator until exhaustion,
ignoring whether the HTTP client had hung up. The handler subprocess kept
generating tokens (eating GPU time, holding KV-cache slots) until the model
naturally finished.

This file verifies the new contract:

  - If the request's `is_disconnected()` returns True between chunks,
    `handle_stream_response` stops iterating the source generator and closes
    it (`.aclose()`).

The close call is the trigger for the existing cancellation propagation in
`HandlerProcessProxy._call_stream`: its `finally` block sends a `_CANCEL`
control message to the handler subprocess, which sets the per-request
`cancel_event` on the `_PendingRequest` in `BatchScheduler`. So verifying that
`handle_stream_response` closes the source generator is the load-bearing
assertion — the rest of the cancel chain is exercised by existing tests
(`tests/test_handler_process_stream_cancellation.py`, `tests/test_batch_scheduler.py`).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.endpoints import handle_stream_response


class _TrackingGenerator:
    """Async generator stub that records whether it was closed early."""

    def __init__(self, num_chunks: int = 20) -> None:
        self.num_chunks = num_chunks
        self.yielded = 0
        self.aclose_called = False
        self.exhausted_naturally = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.yielded >= self.num_chunks:
            self.exhausted_naturally = True
            raise StopAsyncIteration
        self.yielded += 1
        # Cooperative yield so the consumer can poll is_disconnected between
        # chunks.
        await asyncio.sleep(0)
        return f"token{self.yielded}"

    async def aclose(self) -> None:
        self.aclose_called = True


def _make_disconnected_request(disconnects_after_n_polls: int = 1) -> MagicMock:
    """Build a Request stub whose is_disconnected flips to True on the Nth poll."""
    request = MagicMock()
    sequence = [False] * disconnects_after_n_polls + [True] * 50
    request.is_disconnected = AsyncMock(side_effect=sequence)
    return request


@pytest.mark.asyncio
async def test_disconnect_stops_iterating_source_generator():
    """If the client disconnects mid-stream, the wrapper must stop consuming
    the source generator long before it exhausts naturally."""
    src = _TrackingGenerator(num_chunks=200)
    request = _make_disconnected_request(disconnects_after_n_polls=2)

    output_chunks: list[str] = []
    async for sse_chunk in handle_stream_response(
        src,  # type: ignore[arg-type]
        model="test-model",
        request_id="req-disconnect",
        raw_request=request,
    ):
        output_chunks.append(sse_chunk)

    # We must NOT have read all 200 chunks before bailing out.
    assert src.yielded < 200, (
        f"wrapper read {src.yielded} chunks despite client disconnect — is_disconnected polling is not wired"
    )
    # is_disconnected MUST have been called.
    assert request.is_disconnected.await_count >= 1
    # The source generator must NOT have completed naturally.
    assert not src.exhausted_naturally


@pytest.mark.asyncio
async def test_disconnect_closes_source_generator():
    """The source generator must be `.aclose()`d when the client disconnects.

    `aclose()` is what triggers `_call_stream`'s `finally` block to send
    `_CANCEL` to the handler subprocess, which sets the per-request
    `cancel_event` in `BatchScheduler`.
    """
    src = _TrackingGenerator(num_chunks=200)
    request = _make_disconnected_request(disconnects_after_n_polls=1)

    async for _ in handle_stream_response(
        src,  # type: ignore[arg-type]
        model="test-model",
        request_id="req-aclose",
        raw_request=request,
    ):
        pass

    assert src.aclose_called, (
        "source generator was not closed — _CANCEL control message will not "
        "be sent to the handler subprocess; cancel_event will never be set"
    )


@pytest.mark.asyncio
async def test_no_disconnect_drains_source_naturally():
    """When the client never disconnects, the wrapper must drain the full
    stream — i.e. the new is_disconnected hook must not introduce spurious
    early-termination."""
    src = _TrackingGenerator(num_chunks=5)
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=False)

    output_chunks = []
    async for sse_chunk in handle_stream_response(
        src,  # type: ignore[arg-type]
        model="test-model",
        request_id="req-clean",
        raw_request=request,
    ):
        output_chunks.append(sse_chunk)

    assert src.yielded == 5
    assert src.exhausted_naturally


@pytest.mark.asyncio
async def test_handle_stream_response_works_without_raw_request():
    """Backwards compatibility: callers that omit raw_request must still work.
    The Responses API and existing internal callers may pass None."""
    src = _TrackingGenerator(num_chunks=3)
    output_chunks = []
    async for sse_chunk in handle_stream_response(
        src,  # type: ignore[arg-type]
        model="test-model",
        request_id="req-no-raw",
    ):
        output_chunks.append(sse_chunk)

    assert src.yielded == 3
    assert src.exhausted_naturally
