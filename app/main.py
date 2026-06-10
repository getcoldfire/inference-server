"""CLI entrypoint shim for the MLX OpenAI Server package.

This lightweight module allows running the CLI via ``python -m app.main``
while preserving the same behavior as the installed console script. It
normalizes ``sys.argv`` so a missing subcommand implicitly becomes
``launch`` (backwards compatibility) and delegates to the Click-based
``cli`` command group defined in :mod:`app.cli`.

Examples
--------
Run the default launch flow:

    python -m app.main

Forward explicit arguments to the CLI:

    python -m app.main launch --port 8000

Run multi-handler mode from YAML config:

    python -m app.main launch --config config.yaml
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from dataclasses import MISSING, fields

import uvicorn
from loguru import logger

from .config import MLXServerConfig, ModelEntryConfig, MultiModelServerConfig
from .server import setup_server
from .version import __version__

# Total wall-clock budget (seconds) from first SIGTERM/SIGINT until process exit.
# Per the cli-v2 contract: SIGTERM must produce exit code 0 within 5 seconds.
SHUTDOWN_DEADLINE_SECONDS: float = 5.0

_MODEL_ENTRY_DEFAULTS = {
    field.name: (field.default_factory() if field.default_factory is not MISSING else field.default)
    for field in fields(ModelEntryConfig)
    if field.default is not MISSING or field.default_factory is not MISSING
}


def _format_bytes(n: int) -> str:
    """Render a byte count in a human-friendly unit.

    The prompt-cache byte budget defaults to ``1 << 63`` as a sentinel for
    "unbounded", which reads as a 19-digit integer in the startup banner.
    This helper turns that (and more ordinary values) into something a human
    can read at a glance.
    """
    if n >= (1 << 60):
        return "unbounded"
    gib = n / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = n / (1024**2)
    if mib >= 1:
        return f"{mib:.2f} MiB"
    return f"{n} B"


def print_startup_banner(config_args: MLXServerConfig) -> None:
    """Log a compact startup banner describing the selected config.

    The function emits human-friendly log messages that summarize the
    runtime configuration (model path/type, host/port, concurrency,
    LoRA settings, and logging options). Intended for the user-facing
    startup output only.

    Parameters
    ----------
    config_args : MLXServerConfig
        Single-model server configuration.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🔮 Model Path: {config_args.model_path}")
    if config_args.served_model_name:
        logger.info(f"🔮 Served Model Name: {config_args.served_model_name}")
    logger.info(f"🔮 Model Type: {config_args.model_type}")
    if config_args.context_length:
        logger.info(f"🔮 Context Length: {config_args.context_length}")
    logger.info(f"🌐 Host: {config_args.host}")
    logger.info(f"🔌 Port: {config_args.port}")
    logger.info(f"⏱️ Queue Timeout: {config_args.queue_timeout} seconds")
    logger.info(f"📊 Queue Size: {config_args.queue_size}")
    if config_args.model_type == "lm":
        if config_args.enable_auto_tool_choice:
            logger.info("🔧 Auto Tool Choice: Enabled")
        if config_args.tool_call_parser:
            logger.info(f"🔧 Tool Call Parser: {config_args.tool_call_parser}")
        if config_args.reasoning_parser:
            logger.info(f"🔧 Reasoning Parser: {config_args.reasoning_parser}")
        if config_args.message_converter:
            logger.info(f"🔧 Message Converter: {config_args.message_converter}")
    if config_args.model_type == "lm":
        logger.info(f"💾 Prompt Cache Size: {config_args.prompt_cache_size} entries")
        logger.info(f"💾 Prompt Cache Max Bytes: {_format_bytes(config_args.prompt_cache_max_bytes)}")
        if getattr(config_args, "prompt_cache_dir", None):
            logger.info(f"💾 Prompt Cache Dir: {config_args.prompt_cache_dir}")
        if config_args.disable_batching:
            logger.info("🧵 Batch Scheduler: Disabled")
        else:
            logger.info(
                f"🧵 Batch Scheduler: decode={config_args.batch_completion_size}, "
                f"prefill={config_args.batch_prefill_size}, "
                f"prefill_step={config_args.batch_prefill_step_size}"
            )
    logger.info(f"📝 Log Level: {config_args.log_level}")
    logger.info("📝 Log Output: stderr (file logging disabled)")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _model_entry_extras(m: ModelEntryConfig) -> list[tuple[str, object]]:
    """Return non-default per-model settings worth surfacing in the banner.

    Only fields that differ from their dataclass default are included —
    this keeps the banner quiet for minimal configs and lights up every
    knob that was actually touched (parsers, tool choice, LoRA, draft
    model, KV quant, on-demand, etc.).
    """
    extras: list[tuple[str, object]] = []

    if m.served_model_name and m.served_model_name != m.model_path:
        extras.append(("served_model_name", m.served_model_name))
    if m.context_length is not None:
        extras.append(("context_length", m.context_length))
    if m.enable_auto_tool_choice:
        extras.append(("auto_tool_choice", True))
    if m.tool_call_parser:
        extras.append(("tool_call_parser", m.tool_call_parser))
    if m.reasoning_parser:
        extras.append(("reasoning_parser", m.reasoning_parser))
    if m.message_converter:
        extras.append(("message_converter", m.message_converter))
    if m.chat_template_file:
        extras.append(("chat_template_file", m.chat_template_file))
    if m.trust_remote_code:
        extras.append(("trust_remote_code", True))
    if m.draft_model_path:
        extras.append(("draft_model_path", m.draft_model_path))
        extras.append(("num_draft_tokens", m.num_draft_tokens))
    if m.kv_bits is not None:
        extras.append(("kv_bits", f"{m.kv_bits} (group={m.kv_group_size}, start={m.quantized_kv_start})"))
    if m.model_type == "lm":
        batch_settings: list[str] = []
        if m.disable_batching:
            batch_settings.append("disabled")
        if not m.disable_batching and m.batch_completion_size != _MODEL_ENTRY_DEFAULTS["batch_completion_size"]:
            batch_settings.append(f"decode={m.batch_completion_size}")
        if not m.disable_batching and m.batch_prefill_size != _MODEL_ENTRY_DEFAULTS["batch_prefill_size"]:
            batch_settings.append(f"prefill={m.batch_prefill_size}")
        if not m.disable_batching and m.batch_prefill_step_size != _MODEL_ENTRY_DEFAULTS["batch_prefill_step_size"]:
            batch_settings.append(f"prefill_step={m.batch_prefill_step_size}")
        if batch_settings:
            extras.append(("batch_scheduler", ", ".join(batch_settings)))
    if m.on_demand:
        extras.append(("on_demand", f"idle_timeout={m.on_demand_idle_timeout}s"))
    if m.debug:
        extras.append(("debug", True))
    return extras


def print_multi_startup_banner(config: MultiModelServerConfig) -> None:
    """Log a startup banner for multi-handler mode.

    Parameters
    ----------
    config : MultiModelServerConfig
        Multi-model server configuration.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting (Multi-Handler Mode) ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🌐 Host: {config.host}")
    logger.info(f"🔌 Port: {config.port}")
    logger.info(f"📝 Log Level: {config.log_level}")
    logger.info(f"🔢 Models to load: {len(config.models)}")
    for idx, m in enumerate(config.models, start=1):
        logger.info(f"  [{idx}] {m.served_model_name} (type={m.model_type}, path={m.model_path})")
        for key, value in _model_entry_extras(m):
            logger.info(f"       • {key}: {value}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _apply_sampling_env(config: MLXServerConfig) -> None:
    """Set DEFAULT_* env vars from config so model layers use CLI sampling defaults."""
    if config.default_max_tokens is not None:
        os.environ["DEFAULT_MAX_TOKENS"] = str(config.default_max_tokens)
    if config.default_temperature is not None:
        os.environ["DEFAULT_TEMPERATURE"] = str(config.default_temperature)
    if config.default_top_p is not None:
        os.environ["DEFAULT_TOP_P"] = str(config.default_top_p)
    if config.default_top_k is not None:
        os.environ["DEFAULT_TOP_K"] = str(config.default_top_k)
    if config.default_min_p is not None:
        os.environ["DEFAULT_MIN_P"] = str(config.default_min_p)
    if config.default_repetition_penalty is not None:
        os.environ["DEFAULT_REPETITION_PENALTY"] = str(config.default_repetition_penalty)
    if config.default_presence_penalty is not None:
        os.environ["DEFAULT_PRESENCE_PENALTY"] = str(config.default_presence_penalty)
    if config.default_xtc_probability is not None:
        os.environ["DEFAULT_XTC_PROBABILITY"] = str(config.default_xtc_probability)
    if config.default_xtc_threshold is not None:
        os.environ["DEFAULT_XTC_THRESHOLD"] = str(config.default_xtc_threshold)
    if config.default_seed is not None:
        os.environ["DEFAULT_SEED"] = str(config.default_seed)
    if config.default_repetition_context_size is not None:
        os.environ["DEFAULT_REPETITION_CONTEXT_SIZE"] = str(config.default_repetition_context_size)


def _force_kill_children() -> None:
    """SIGKILL any surviving multiprocessing-spawned handler children.

    Called from the shutdown watchdog when normal cleanup exceeds the
    5 s budget. Uses ``mp.active_children()`` rather than chasing
    ``HandlerProcessProxy`` references because the latter may be
    half-torn-down or unreachable from the main thread.
    """
    for child in mp.active_children():
        try:
            child.kill()
        except (OSError, ProcessLookupError, ValueError):
            pass


def _arm_shutdown_watchdog(server: uvicorn.Server) -> threading.Event:
    """Install SIGTERM/SIGINT handlers that enforce the 5 s exit deadline.

    On the first signal:
      1. ``server.should_exit`` is set so uvicorn returns from
         ``serve()`` and the FastAPI lifespan starts its cleanup.
      2. A daemon thread starts a 5 s wall-clock timer; if the process
         is still running when it expires, surviving handler
         subprocesses are SIGKILLed and the process exits with code 0
         via ``os._exit``. Exit code 0 is the cli-v2 contract: a
         caller-initiated SIGTERM is a clean shutdown, not a failure.

    On a second signal (double Ctrl-C), exit immediately with code 0.

    Returns
    -------
    threading.Event
        An event set when the first signal is observed. Tests can wait
        on this to confirm the handler fired.
    """
    triggered = threading.Event()

    def _watchdog() -> None:
        deadline = time.monotonic() + SHUTDOWN_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            time.sleep(0.05)
        # Budget exceeded: kill any lingering MLX handler subprocesses, then
        # hard-exit the parent. We exit 0 because the operator (or launchd)
        # asked us to stop; exceeding the soft drain budget is not a failure
        # mode the caller cares about.
        logger.warning(
            f"Shutdown exceeded {SHUTDOWN_DEADLINE_SECONDS}s budget; force-killing handler subprocesses and exiting."
        )
        _force_kill_children()
        os._exit(0)

    def _on_signal(signum: int, _frame: object) -> None:
        if triggered.is_set():
            # Second signal: caller wants out immediately.
            logger.warning(f"Received signal {signum} during shutdown; exiting now.")
            _force_kill_children()
            os._exit(0)
        triggered.set()
        logger.info(f"Received signal {signum}; initiating graceful shutdown.")
        server.should_exit = True
        threading.Thread(target=_watchdog, name="shutdown-watchdog", daemon=True).start()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    return triggered


async def _serve_with_watchdog(server: uvicorn.Server) -> None:
    """Run ``server.serve()`` with our 5 s shutdown contract installed.

    The signal handlers must be installed *before* ``serve()`` is
    awaited so that they pre-empt uvicorn's own ``capture_signals``.
    Uvicorn re-installs its handlers inside ``serve()``; we re-arm ours
    after a short delay to win the race for any subsequent signal.
    """
    _arm_shutdown_watchdog(server)

    async def _rearm_after_uvicorn() -> None:
        # uvicorn's capture_signals overrides our handlers as soon as
        # serve() starts; wait briefly then re-install ours.
        await asyncio.sleep(0.2)
        _arm_shutdown_watchdog(server)

    rearm_task = asyncio.create_task(_rearm_after_uvicorn())
    try:
        await server.serve()
    finally:
        rearm_task.cancel()


async def start(config: MLXServerConfig) -> None:
    """Run the ASGI server using the provided configuration.

    This coroutine wires the configuration into the server setup
    routine, logs progress, and starts the Uvicorn server. It handles
    KeyboardInterrupt and logs any startup failures before exiting the
    process with a non-zero code.

    Parameters
    ----------
    config : MLXServerConfig
        Single-model server configuration.
    """
    try:
        _apply_sampling_env(config)
        # Display startup information
        print_startup_banner(config)

        # Set up and start the server
        uvconfig = setup_server(config)
        logger.info("Server configuration complete.")
        logger.info("Starting Uvicorn server...")
        server = uvicorn.Server(uvconfig)
        await _serve_with_watchdog(server)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Server startup failed: {e!s}")
        sys.exit(1)


async def start_multi(config: MultiModelServerConfig) -> None:
    """Run the ASGI server in multi-handler mode.

    Similar to ``start`` but works with a ``MultiModelServerConfig``
    which defines multiple models to be loaded concurrently.

    Parameters
    ----------
    config : MultiModelServerConfig
        Multi-model YAML-based configuration.
    """
    try:
        print_multi_startup_banner(config)

        uvconfig = setup_server(config)
        logger.info("Multi-handler server configuration complete.")
        logger.info("Starting Uvicorn server...")
        server = uvicorn.Server(uvconfig)
        await _serve_with_watchdog(server)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Multi-handler server startup failed: {e!s}")
        sys.exit(1)


def main():
    """Normalize process args and dispatch to the Click CLI.

    This helper gathers command-line arguments, inserts the "launch"
    subcommand when a subcommand is omitted for backwards compatibility,
    and delegates execution to :func:`app.cli.cli` through
    ``cli.main``.

    Top-level flags exposed on the ``cli`` group itself (``--version``,
    ``--help``, ``--licenses``) are passed through unmodified so that
    ``python -m app.main --licenses`` prints attributions and exits
    instead of being silently rewritten to a ``launch`` invocation.
    """
    from .cli import cli

    # Top-level group flags must not be rewritten into a ``launch`` invocation.
    _GROUP_FLAGS = {"--version", "--help", "-h", "--licenses"}
    args = [str(x) for x in sys.argv[1:]]
    # Keep backwards compatibility: Add 'launch' subcommand if none is provided
    if not args:
        args.insert(0, "launch")
    elif args[0].startswith("-") and args[0] not in _GROUP_FLAGS:
        args.insert(0, "launch")
    cli.main(args=args)


if __name__ == "__main__":
    main()
