"""Tests for ``_stream_until_cancelled`` in the handler subprocess.

A client that disconnects mid-stream must stop generation. The helper only
checked for cancellation when the next chunk took longer than
``poll_interval`` to arrive, so a healthy stream -- a token every few
milliseconds -- was never cancelled and ran on to ``max_tokens`` with
nobody reading, holding its batch slot the whole time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from app.core.handler_process import _stream_until_cancelled

pytestmark = pytest.mark.asyncio


class _Source:
    """An endless token stream that records whether it was closed."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.produced = 0
        self.closed = False

    async def stream(self) -> AsyncGenerator[int, None]:
        """Yield a chunk every ``delay`` seconds until closed."""
        try:
            while True:
                await asyncio.sleep(self.delay)
                self.produced += 1
                yield self.produced
        finally:
            self.closed = True


async def _consume(source: _Source, cancel: asyncio.Event, cancel_after: int, poll_interval: float) -> list[int]:
    """Forward chunks, requesting cancellation after ``cancel_after`` of them."""
    received: list[int] = []
    async for chunk in _stream_until_cancelled(
        stream=source.stream(), should_cancel=cancel.is_set, poll_interval=poll_interval
    ):
        received.append(chunk)
        if len(received) == cancel_after:
            cancel.set()
        if len(received) > 1000:
            break
    return received


async def test_fast_stream_stops_promptly_when_cancelled() -> None:
    """Chunks arriving faster than the poll interval must not hide a cancel."""
    source = _Source(delay=0.001)
    cancel = asyncio.Event()

    received = await asyncio.wait_for(_consume(source, cancel, cancel_after=5, poll_interval=0.1), timeout=5)

    assert len(received) <= 6
    assert source.closed


async def test_slow_stream_stops_when_cancelled_between_chunks() -> None:
    """A cancel noticed while waiting for a slow chunk still ends the stream."""
    source = _Source(delay=0.5)
    cancel = asyncio.Event()

    received = await asyncio.wait_for(_consume(source, cancel, cancel_after=1, poll_interval=0.05), timeout=5)

    assert received == [1]
    assert source.closed


async def test_stream_runs_to_completion_without_cancel() -> None:
    """An uncancelled stream forwards every chunk."""

    async def finite() -> AsyncGenerator[int, None]:
        for i in range(20):
            yield i

    received = [chunk async for chunk in _stream_until_cancelled(stream=finite(), should_cancel=lambda: False)]

    assert received == list(range(20))
