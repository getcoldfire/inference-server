"""Regression tests for on-demand model reference counting and idle unload.

Two defects let the idle timer unload an on-demand model while requests
were still being served:

* Requests to an already-loaded on-demand model took the plain
  ``registry.get_handler`` path, so they never took a reference and never
  cancelled a pending idle timer.
* The route released its reference in ``finally`` as soon as it returned
  a ``StreamingResponse`` -- before a single token had been generated.

A third defect made the aftermath worse: unloading the model left any
open streams waiting out the full RPC timeout instead of failing them.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from fastapi.responses import JSONResponse, StreamingResponse

from app.core import handler_process
from app.core.handler_process import HandlerProcessProxy
from app.core.model_registry import ModelRegistry
from app.schemas.openai import ChatCompletionRequest, Message, ResponsesRequest

pytestmark = pytest.mark.asyncio

MODEL_ID = "fast"


def _load_endpoints_module() -> Any:
    """Import ``app.api.endpoints`` with a lightweight LM handler stub.

    Returns
    -------
    Any
        A freshly imported ``app.api.endpoints`` module.
    """
    fake_lm_module = types.ModuleType("app.handler.mlx_lm")
    fake_lm_module.MLXLMHandler = object

    module_names = ["app.handler.mlx_lm", "app.api.endpoints"]
    original_modules: dict[str, types.ModuleType | None] = {name: sys.modules.get(name) for name in module_names}
    try:
        sys.modules["app.handler.mlx_lm"] = fake_lm_module
        sys.modules.pop("app.api.endpoints", None)
        return importlib.import_module("app.api.endpoints")
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _StubProxy:
    """Stand-in for ``HandlerProcessProxy`` that never spawns a process."""

    handler_type = "lm"
    _uses_model_sampling_defaults = False

    def __init__(self, **kwargs: Any) -> None:
        self.served_model_name = kwargs["served_model_name"]
        self.cleaned_up = False

    async def start(self, queue_config: dict[str, Any]) -> None:
        """Pretend to start the handler subprocess."""

    async def cleanup(self) -> None:
        """Record that the registry unloaded this handler."""
        self.cleaned_up = True


@pytest_asyncio.fixture
async def registry(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ModelRegistry, None]:
    """A real registry with one on-demand model whose loads are stubbed."""
    monkeypatch.setattr(handler_process, "HandlerProcessProxy", _StubProxy)
    reg = ModelRegistry()
    await reg.register_on_demand_model(
        model_id=MODEL_ID,
        model_cfg_dict={"model_path": "x", "model_type": "lm"},
        model_type="lm",
        model_path="x",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 10},
        idle_timeout=600,
    )
    yield reg
    for task in reg._on_demand_idle_tasks.values():
        task.cancel()


def _make_raw_request(registry: ModelRegistry) -> Any:
    """Build a fresh request-like object bound to ``registry``."""
    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(registry=registry, handler=None)),
        state=types.SimpleNamespace(request_id="req-test"),
    )


class _GatedStream:
    """A streaming body that yields one chunk, then waits to be released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def body(self) -> AsyncGenerator[str, None]:
        """Yield a first chunk, then block until ``release`` is set."""
        yield "data: first\n\n"
        await self.release.wait()
        yield "data: [DONE]\n\n"


def _chat_request(stream: bool) -> ChatCompletionRequest:
    """Build a minimal chat request for ``MODEL_ID``."""
    return ChatCompletionRequest(model=MODEL_ID, messages=[Message(role="user", content="hi")], stream=stream)


def _responses_request(stream: bool) -> ResponsesRequest:
    """Build a minimal Responses API request for ``MODEL_ID``."""
    return ResponsesRequest(model=MODEL_ID, input="hi", stream=stream)


async def _call_route(
    endpoints: Any,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    registry: ModelRegistry,
    response: JSONResponse | StreamingResponse,
) -> JSONResponse | StreamingResponse:
    """Call a text route with its processing step replaced by ``response``.

    Parameters
    ----------
    endpoints : Any
        The imported endpoints module.
    monkeypatch : pytest.MonkeyPatch
        Used to replace the route's processing function.
    route : str
        ``"chat"`` or ``"responses"``.
    registry : ModelRegistry
        The registry the request resolves against.
    response : JSONResponse | StreamingResponse
        What the processing step returns.

    Returns
    -------
    JSONResponse | StreamingResponse
        Whatever the route returned.
    """

    async def _fake_process(*args: Any, **kwargs: Any) -> JSONResponse | StreamingResponse:
        return response

    if route == "chat":
        monkeypatch.setattr(endpoints, "process_text_request", _fake_process)
        stream = isinstance(response, StreamingResponse)
        return await endpoints.chat_completions(_chat_request(stream), _make_raw_request(registry))
    monkeypatch.setattr(endpoints, "process_text_responses_request", _fake_process)
    stream = isinstance(response, StreamingResponse)
    return await endpoints.responses_endpoint(_responses_request(stream), _make_raw_request(registry))


@pytest.mark.parametrize("route", ["chat", "responses"])
async def test_streaming_request_holds_reference_until_body_finishes(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """A stream must keep the model referenced until its last chunk is sent."""
    endpoints = _load_endpoints_module()
    gate = _GatedStream()

    response = await _call_route(endpoints, monkeypatch, route, registry, StreamingResponse(gate.body()))

    assert registry._on_demand_ref_count[MODEL_ID] == 1
    assert MODEL_ID not in registry._on_demand_idle_tasks

    chunks = [await anext(response.body_iterator)]
    gate.release.set()
    chunks.extend([chunk async for chunk in response.body_iterator])

    assert chunks == ["data: first\n\n", "data: [DONE]\n\n"]
    assert registry._on_demand_ref_count[MODEL_ID] == 0
    assert MODEL_ID in registry._on_demand_idle_tasks


async def test_stream_releases_reference_when_client_disconnects(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing the body early (client disconnect) must still release the model."""
    endpoints = _load_endpoints_module()
    gate = _GatedStream()

    response = await _call_route(endpoints, monkeypatch, "chat", registry, StreamingResponse(gate.body()))
    await anext(response.body_iterator)
    await response.body_iterator.aclose()

    assert registry._on_demand_ref_count[MODEL_ID] == 0
    assert MODEL_ID in registry._on_demand_idle_tasks


async def test_non_streaming_request_releases_reference_on_return(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain JSON response is complete when the route returns."""
    endpoints = _load_endpoints_module()

    await _call_route(endpoints, monkeypatch, "chat", registry, JSONResponse(content={"ok": True}))

    assert registry._on_demand_ref_count[MODEL_ID] == 0
    assert MODEL_ID in registry._on_demand_idle_tasks


async def test_request_to_loaded_model_takes_reference_and_cancels_idle_timer(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests after the one that loaded the model must count as activity."""
    endpoints = _load_endpoints_module()

    # First request loads the model and, once finished, arms the idle timer.
    await _call_route(endpoints, monkeypatch, "chat", registry, JSONResponse(content={"ok": True}))
    idle_task = registry._on_demand_idle_tasks[MODEL_ID]

    gate = _GatedStream()
    response = await _call_route(endpoints, monkeypatch, "chat", registry, StreamingResponse(gate.body()))
    await asyncio.sleep(0)

    assert idle_task.cancelled()
    assert MODEL_ID not in registry._on_demand_idle_tasks
    assert registry._on_demand_ref_count[MODEL_ID] == 1

    gate.release.set()
    [chunk async for chunk in response.body_iterator]
    assert registry._on_demand_ref_count[MODEL_ID] == 0


async def test_concurrent_streams_keep_model_loaded_until_all_finish(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The idle timer must not start while any stream is still open."""
    endpoints = _load_endpoints_module()
    gate_a, gate_b = _GatedStream(), _GatedStream()

    response_a = await _call_route(endpoints, monkeypatch, "chat", registry, StreamingResponse(gate_a.body()))
    response_b = await _call_route(endpoints, monkeypatch, "chat", registry, StreamingResponse(gate_b.body()))
    assert registry._on_demand_ref_count[MODEL_ID] == 2

    gate_a.release.set()
    [chunk async for chunk in response_a.body_iterator]
    assert registry._on_demand_ref_count[MODEL_ID] == 1
    assert MODEL_ID not in registry._on_demand_idle_tasks

    gate_b.release.set()
    [chunk async for chunk in response_b.body_iterator]
    assert registry._on_demand_ref_count[MODEL_ID] == 0
    assert MODEL_ID in registry._on_demand_idle_tasks


class _FakeProcess:
    """A ``multiprocessing.Process`` stand-in with a controllable lifetime."""

    def __init__(self, alive: bool) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        """Report whether the fake child is still running."""
        return self.alive

    def terminate(self) -> None:
        """Mark the fake child as stopped."""
        self.alive = False

    def kill(self) -> None:
        """Mark the fake child as stopped."""
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        """Return immediately; the fake child has no real process."""


class _BrokenQueue:
    """A request queue whose pipe has already broken."""

    def put(self, item: dict[str, Any]) -> None:
        """Fail the way a queue to a dead child fails."""
        raise BrokenPipeError("child is gone")


@pytest.mark.parametrize("alive", [True, False], ids=["child-alive", "child-already-dead"])
async def test_proxy_cleanup_fails_open_streams_instead_of_leaving_them_to_time_out(alive: bool) -> None:
    """Unloading a model must end its open streams with an error straight away."""
    proxy = HandlerProcessProxy(
        model_cfg_dict={},
        model_type="lm",
        model_path="x",
        served_model_name=MODEL_ID,
    )
    proxy._process = _FakeProcess(alive=alive)
    proxy._request_queue = _BrokenQueue()
    stream_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    proxy._pending["stream-1"] = stream_queue

    await proxy.cleanup()

    message = stream_queue.get_nowait()
    assert message["type"] == "error"
    assert message["status_code"] == 503
    assert proxy._pending == {}
