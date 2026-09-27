"""Tests for scheduler recovery after ``BatchGenerator.next()`` raises.

A long generation hit ``[metal::malloc] Resource limit (499000) exceeded``
inside ``BatchGenerator.next()``. The scheduler failed the active requests
but left their sequences inside the generator, so every later ``next()``
raised the same error and the model returned 500 for every request until
the server was restarted.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_batch_scheduler import (  # noqa: F401 -- fixture re-export
    FakeBatchGenerator,
    FakeTokenizer,
    _FakeScript,
    patched_scheduler,
)


class _PoisonedBatchGenerator(FakeBatchGenerator):
    """Raises from ``next()`` for as long as the first sequence is in the batch."""

    def next(self) -> tuple[list[Any], list[Any]]:
        """Fail while uid 0 is still held, as the real generator did."""
        if any(uid == 0 for uid, _script, _idx in self._pending):
            raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
        return super().next()


@pytest.fixture
def bsm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The ``batch_scheduler`` module over a generator that fails on uid 0."""
    module = request.getfixturevalue("patched_scheduler")
    monkeypatch.setattr(module, "BatchGenerator", _PoisonedBatchGenerator)
    return module


@pytest.mark.asyncio
async def test_request_after_generator_failure_still_succeeds(bsm: Any) -> None:
    """One failed step must not leave the model failing every later request."""
    FakeBatchGenerator.script_queue = [
        _FakeScript(tokens=[10, 11], finish_reason="stop"),
        _FakeScript(tokens=[20, 21], finish_reason="stop"),
    ]
    scheduler = bsm.BatchScheduler(model=object(), tokenizer=FakeTokenizer(), idle_poll_timeout=0.01)
    scheduler.start()
    try:
        with pytest.raises(RuntimeError, match="Resource limit"):
            [chunk async for chunk in scheduler.submit_stream(input_ids=[1], max_tokens=8)]

        chunks = [chunk async for chunk in scheduler.submit_stream(input_ids=[2], max_tokens=8)]
    finally:
        scheduler.stop()

    assert [c.token for c in chunks] == [20, 21]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_failed_sequences_are_removed_from_the_generator(bsm: Any) -> None:
    """The failed sequence must be taken out of the batch, not just forgotten."""
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[10, 11], finish_reason="stop")]
    scheduler = bsm.BatchScheduler(model=object(), tokenizer=FakeTokenizer(), idle_poll_timeout=0.01)
    scheduler.start()
    try:
        with pytest.raises(RuntimeError, match="Resource limit"):
            [chunk async for chunk in scheduler.submit_stream(input_ids=[1], max_tokens=8)]
        generator = scheduler._batch_generator
    finally:
        scheduler.stop()

    assert generator.removed == [0]
