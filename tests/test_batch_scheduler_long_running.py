"""Tests for the scheduler's long-running request log.

Requests have been seen to be admitted and then never finish, with no
chunk reaching the client until the proxy's RPC timeout. The scheduler
only logged admission and completion, so there was no way to tell a
sequence still generating (whose output never reaches the client) from
one that had stalled. These tests pin down the periodic warning that
reports how a long-running request is progressing.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger

from tests.test_batch_scheduler import (  # noqa: F401 -- fixture re-export
    FakeBatchGenerator,
    FakeTokenizer,
    _FakeScript,
    patched_scheduler,
)


@pytest.fixture
def bsm(request: pytest.FixtureRequest) -> Any:
    """The ``batch_scheduler`` module with its MLX pieces stubbed."""
    return request.getfixturevalue("patched_scheduler")


@pytest.fixture
def warnings_logged() -> Iterator[list[str]]:
    """Capture loguru WARNING-level messages emitted during a test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


def _long_running(messages: list[str]) -> list[str]:
    """Return only the long-running request warnings."""
    return [m for m in messages if "still running" in m]


def _make_scheduler(bsm: Any, **kwargs: Any) -> Any:
    """Build a scheduler over the fake generator without starting it."""
    return bsm.BatchScheduler(model=object(), tokenizer=FakeTokenizer(), idle_poll_timeout=0.01, **kwargs)


def _add_active(bsm: Any, scheduler: Any, uid: int, admitted_time: float) -> Any:
    """Register a live request directly, as admission would."""
    state = bsm._ActiveRequest(
        loop=None,
        out_queue=None,
        detokenizer=FakeTokenizer().detokenizer,
        cancel_event=threading.Event(),
        prompt_tokens=7574,
        cached_prompt_tokens=6552,
        pending_segment_types=[],
        admitted_time=admitted_time,
    )
    scheduler._active[uid] = state
    return state


def test_request_under_threshold_is_not_reported(bsm: Any, warnings_logged: list[str]) -> None:
    """A request younger than the threshold is normal and stays quiet."""
    scheduler = _make_scheduler(bsm, long_running_log_after=60.0)
    _add_active(bsm, scheduler, uid=3, admitted_time=1000.0)

    scheduler._log_long_running(now=1059.0)

    assert _long_running(warnings_logged) == []


def test_request_with_no_tokens_reports_that_none_arrived(bsm: Any, warnings_logged: list[str]) -> None:
    """A stalled request must say it has produced nothing at all."""
    scheduler = _make_scheduler(bsm, long_running_log_after=60.0)
    _add_active(bsm, scheduler, uid=3, admitted_time=1000.0)

    scheduler._log_long_running(now=1090.0)

    (message,) = _long_running(warnings_logged)
    assert "uid=3" in message
    assert "90s" in message
    assert "generated=0" in message
    assert "no token yet" in message
    assert "prompt_tokens=7574" in message
    assert "cached_prefix=6552" in message


def test_generating_request_reports_progress_and_recent_output(bsm: Any, warnings_logged: list[str]) -> None:
    """A request that is generating must show its count, last token age and tail."""
    scheduler = _make_scheduler(bsm, long_running_log_after=60.0)
    state = _add_active(bsm, scheduler, uid=0, admitted_time=1000.0)
    state.generation_tokens = 5000
    state.last_token_time = 1098.0
    state.recent_tokens.extend([151643, 151643, 151643])
    state.recent_text = "…still thinking"

    scheduler._log_long_running(now=1100.0)

    (message,) = _long_running(warnings_logged)
    assert "generated=5000" in message
    assert "last token 2.0s ago" in message
    assert "recent_tokens=[151643, 151643, 151643]" in message
    assert "recent_text='…still thinking'" in message


def test_long_running_report_is_throttled(bsm: Any, warnings_logged: list[str]) -> None:
    """Each request is reported at most once per interval, not on every step."""
    scheduler = _make_scheduler(bsm, long_running_log_after=60.0, long_running_log_interval=60.0)
    _add_active(bsm, scheduler, uid=3, admitted_time=1000.0)

    scheduler._log_long_running(now=1060.0)
    scheduler._log_long_running(now=1061.0)
    scheduler._log_long_running(now=1119.0)
    assert len(_long_running(warnings_logged)) == 1

    scheduler._log_long_running(now=1120.0)
    assert len(_long_running(warnings_logged)) == 2


@pytest.mark.asyncio
async def test_scheduler_loop_reports_long_running_request_with_its_tokens(
    bsm: Any, warnings_logged: list[str]
) -> None:
    """The live loop must track each token and report a slow request."""
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[10, 11, 12, 13], finish_reason="length")]
    FakeBatchGenerator.step_delay = 0.02
    scheduler = _make_scheduler(bsm, long_running_log_after=0.0, long_running_log_interval=0.0)
    scheduler.start()
    try:
        chunks = [chunk async for chunk in scheduler.submit_stream(input_ids=[7, 8], max_tokens=16)]
    finally:
        scheduler.stop()

    assert [c.token for c in chunks] == [10, 11, 12, 13]
    reports = _long_running(warnings_logged)
    assert reports, "expected the loop to report the running request"
    assert "recent_tokens=[10, 11, 12]" in reports[-1]
    assert "recent_text='<10><11><12>'" in reports[-1]
