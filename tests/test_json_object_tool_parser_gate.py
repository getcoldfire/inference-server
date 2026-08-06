"""The safety property of Task 6: the bare-JSON parser must only ever run
against a response to a request that actually sent ``tools``.

Every other tool parser keys off a wire-format delimiter (``<tool_call>`` and
friends) -- a model answering in plain content simply never emits that
delimiter, so nothing gets misparsed. The bare-JSON parser
(``app/parsers/json_object.py``) decides purely from shape, so without a
gate a model legitimately answering a question with a JSON object would be
misread as calling a function. The gate lives in
``app/handler/mlx_lm.py:_gate_tool_parser_on_tools``, called from
``_build_inference_context`` before either the streaming or non-streaming
path ever touches the parser -- this exercises it through the real handler,
not just the helper in isolation.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import threading
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path


def _load_mlx_lm_handler_class() -> type:
    """Import ``MLXLMHandler`` with lightweight stubs for MLX-heavy modules.

    Mirrors the loader in
    tests/test_mixed_think_tool_handoff_stream_handler_integration.py.
    """
    repo_root = Path(__file__).resolve().parents[1]

    fake_handler_package = types.ModuleType("app.handler")
    fake_handler_package.__path__ = [str(repo_root / "app" / "handler")]

    fake_model_module = types.ModuleType("app.models.mlx_lm")
    fake_model_module.MLX_LM = object

    fake_prompt_cache_module = types.ModuleType("app.utils.prompt_cache")
    fake_prompt_cache_module.LRUPromptCache = object

    module_names = [
        "app.handler",
        "app.models.mlx_lm",
        "app.utils.prompt_cache",
        "app.handler.mlx_lm",
    ]
    original_modules: dict[str, types.ModuleType | None] = {name: sys.modules.get(name) for name in module_names}

    try:
        sys.modules["app.handler"] = fake_handler_package
        sys.modules["app.models.mlx_lm"] = fake_model_module
        sys.modules["app.utils.prompt_cache"] = fake_prompt_cache_module
        sys.modules.pop("app.handler.mlx_lm", None)

        module = importlib.import_module("app.handler.mlx_lm")
        return module.MLXLMHandler
    finally:
        sys.modules.pop("app.handler.mlx_lm", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@dataclass
class _FakeNonStreamResponse:
    """Minimal non-stream response object consumed by ``generate_text_response``."""

    text: str
    tokens: list[int]
    prompt_tokens: int = 11
    generation_tokens: int = 7
    generation_tps: float = 1.0
    peak_memory: float = 0.0


@dataclass
class _FakeRequest:
    """Minimal ``ChatCompletionRequest`` stand-in: only ``.tools`` matters here."""

    tools: list[object] | None = field(default=None)


class _FakeModel:
    """Tiny model stub used by ``generate_text_response``."""

    has_draft_model = False
    cache_is_batchable = False
    cache_is_trimmable = True  # skip checkpoint logic in the non-batched path

    def create_input_prompt(self, messages: list[dict[str, str]], kwargs: dict[str, object]) -> str:
        return "prompt"

    def encode_prompt(self, prompt: str) -> list[int]:
        return [1, 2, 3]

    def create_prompt_cache(self) -> dict[str, bool]:
        return {"cache": True}


class _FakePromptCache:
    """Prompt cache stub matching the handler interface."""

    def fetch_nearest_cache(self, input_ids: list[int], *, allowed_sources=None) -> tuple[None, list[int]]:
        return None, input_ids

    def insert_cache(self, cache_key: list[int], cache: object) -> None:
        pass

    def log_cache_stats(self) -> None:
        pass


class _FakeInferenceWorker:
    """Inference worker stub that returns a fixed non-stream response."""

    def __init__(self, non_stream_response: _FakeNonStreamResponse) -> None:
        self._non_stream_response = non_stream_response

    async def submit(self, *args: object, **kwargs: object) -> _FakeNonStreamResponse:
        return self._non_stream_response


def _configure_handler_stubs(handler: object) -> None:
    """Add the minimal handler-instance attributes production ``__init__`` sets
    but ``object.__new__`` skips."""
    handler._generation_lock = threading.RLock()  # type: ignore[attr-defined]
    handler._batch_scheduler = None  # type: ignore[attr-defined]
    handler._batch_scheduler_lock = asyncio.Lock()  # type: ignore[attr-defined]
    handler._disable_batching = False  # type: ignore[attr-defined]


def _build_handler(handler_cls: type, response_text: str) -> object:
    handler = object.__new__(handler_cls)
    handler.debug = False
    handler.message_converter = None
    handler.enable_auto_tool_choice = False
    handler.reasoning_parser_name = None
    handler.tool_parser_name = "json"
    handler.model = _FakeModel()
    handler.prompt_cache = _FakePromptCache()
    _configure_handler_stubs(handler)
    handler.inference_worker = _FakeInferenceWorker(
        _FakeNonStreamResponse(text=response_text, tokens=[1, 2, 3], prompt_tokens=100, generation_tokens=20)
    )

    async def _fake_prepare_text_request(
        self: object, request: object
    ) -> tuple[list[dict[str, str]], dict[str, object]]:
        return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

    handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)
    return handler


class JsonObjectToolParserGateTests(unittest.TestCase):
    """Exercise the ``requires_tools`` gate through the real handler."""

    BARE_JSON_OUTPUT = '{"name": "create_rule", "arguments": {"name": "Junk bob"}}'

    def test_tools_absent_returns_json_looking_output_as_content(self) -> None:
        """No ``tools`` on the request: the bare JSON must NOT be parsed as a call."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = _build_handler(handler_cls, self.BARE_JSON_OUTPUT)

        result = asyncio.run(handler.generate_text_response(request=_FakeRequest(tools=None)))
        parsed = result["response"]

        assert parsed["tool_calls"] is None, "a tools-less request must never yield tool_calls"
        assert parsed["content"] == self.BARE_JSON_OUTPUT

    def test_tools_present_parses_the_same_output_as_a_call(self) -> None:
        """Same model output, but the request offered tools: this IS a call."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = _build_handler(handler_cls, self.BARE_JSON_OUTPUT)

        result = asyncio.run(handler.generate_text_response(request=_FakeRequest(tools=[object()])))
        parsed = result["response"]

        assert isinstance(parsed["tool_calls"], list)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "create_rule"
        assert json.loads(parsed["tool_calls"][0]["arguments"]) == {"name": "Junk bob"}


if __name__ == "__main__":
    unittest.main()
