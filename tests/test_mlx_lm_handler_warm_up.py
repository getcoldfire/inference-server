"""Tests that ``MLXLMHandler.initialize()`` warms the model on the caller thread.

Same root cause as ``BatchScheduler._warm_up_model``. Requests on the
``--disable-batching`` path run on the ``InferenceWorker`` thread, which
owns its own thread-local stream. Without a loader-thread warm-up the
model's deferred MLX state binds to the worker thread on first use and
later cross-thread evaluations raise
``RuntimeError: There is no Stream(gpu, N) in current thread``.
"""

from __future__ import annotations

import threading
from types import ModuleType
from typing import Any

import pytest


def _install_stub_handler_deps(monkeypatch: pytest.MonkeyPatch, captured: list[int]) -> ModuleType:
    """Stub heavy handler dependencies so the handler builds without a real model.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to install the stubs.
    captured : list[int]
        Receives the thread id of every forward pass on the stub model.

    Returns
    -------
    ModuleType
        The imported ``app.handler.mlx_lm`` module.
    """

    class _StubRawModel:
        # Presence of ``.layers`` distinguishes a real model from unit-test
        # mocks; the warm-up must skip when absent.
        layers = (object(),)

        def __call__(self, _ids: Any, cache: Any = None) -> Any:
            captured.append(threading.get_ident())
            return object()  # opaque logits

    class _StubTokenizer:
        bos_token_id = 1
        pad_token_id = 0
        eos_token_ids: list[int] = [1]
        eos_token_id = 1
        vocab_size = 32000

    class _StubMLX_LM:
        def __init__(self, model_path: str, **_kwargs: Any) -> None:
            self.model = _StubRawModel()
            self.tokenizer = _StubTokenizer()
            self.draft_model = None
            self.draft_tokenizer = None
            self.model_type = "stub"
            self.pad_token_id = 0
            self.bos_token = "<s>"

        def get_model_type(self) -> str:
            return self.model_type

    import app.models.mlx_lm as model_module

    monkeypatch.setattr(model_module, "MLX_LM", _StubMLX_LM)

    import app.handler.mlx_lm as handler_module
    from app.core.inference_worker import InferenceWorker as _RealInferenceWorker

    # ``test_batch_scheduler`` pops ``app.handler.mlx_lm`` from ``sys.modules``
    # and re-imports it against a fake ``app.core`` whose ``InferenceWorker``
    # is a no-arg stub; that pop is not undone by ``monkeypatch``. Re-bind to
    # the real class defensively.
    monkeypatch.setattr(handler_module, "InferenceWorker", _RealInferenceWorker)

    # Bypass parser loading (``None`` is safe; the handler tolerates it).
    monkeypatch.setattr(
        handler_module.MessageConverterManager,
        "create_converter",
        staticmethod(lambda **_kwargs: None),
    )

    # Stub the warm-up's MLX-side calls so it runs end-to-end without real MLX.
    import mlx.core as mx
    import mlx_lm.models.cache as cache_mod

    monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda _model: [])
    monkeypatch.setattr(mx, "array", lambda _data: object())
    monkeypatch.setattr(mx, "eval", lambda *_args, **_kwargs: None)

    return handler_module


@pytest.mark.asyncio
async def test_initialize_warms_up_model_on_caller_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """``initialize()`` must invoke the model on the caller thread before the worker spawns."""
    main_tid = threading.get_ident()
    call_threads: list[int] = []
    handler_module = _install_stub_handler_deps(monkeypatch, call_threads)

    handler = handler_module.MLXLMHandler(model_path="stub/path")
    try:
        await handler.initialize({})
    finally:
        handler.inference_worker.stop()

    assert main_tid in call_threads, (
        "MLXLMHandler.initialize() must run a warm-up forward pass on the "
        f"caller thread (id={main_tid}); observed call threads={call_threads}"
    )
