"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server. When a ``--config`` YAML file is supplied the server
runs in multi-handler mode, loading multiple models at once.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from loguru import logger

from .config import MLXServerConfig, load_config_from_yaml
from .main import start_multi
from .parsers import REASONING_PARSER_MAP, TOOL_PARSER_MAP, UNIFIED_PARSER_MAP
from .version import __version__

# cli-v2 spec contract (§4): accepts info|debug|warn|error in any case.
# We translate to the internal loguru levels used by the rest of the app.
# Aliases keep the older verbose names ("WARNING", "CRITICAL") working so
# existing scripts/Makefiles don't break.
_LOG_LEVEL_CHOICES: tuple[str, ...] = (
    "DEBUG",
    "INFO",
    "WARN",
    "WARNING",
    "ERROR",
    "CRITICAL",
)
_LOG_LEVEL_ALIASES: dict[str, str] = {"WARN": "WARNING"}


def _resolve_log_level(value: str | None) -> str:
    """Canonicalize ``--log-level`` input to a loguru-compatible name."""
    if value is None:
        return "INFO"
    upper = value.upper()
    return _LOG_LEVEL_ALIASES.get(upper, upper)


class UpperChoice(click.Choice):
    """Case-insensitive choice type that returns uppercase values.

    This small convenience subclass normalizes user input in a
    case-insensitive way but returns the canonical uppercase option
    value to callers. It is useful for flags like ``--log-level``
    where the internal representation is uppercased.
    """

    def normalize_choice(self, choice, ctx):
        """Return the canonical uppercase choice or raise BadParameter.

        Parameters
        ----------
        choice:
            Raw value supplied by the user (may be ``None``).
        ctx:
            Click context object (unused here but part of the API).

        Returns
        -------
        str | None
            Uppercased canonical choice, or ``None`` if ``choice`` is
            ``None``.
        """
        if choice is None:
            return None
        upperchoice = choice.upper()
        for opt in self.choices:
            if opt.upper() == upperchoice:
                return upperchoice
        raise click.BadParameter(f"invalid choice: {choice!r}. (choose from {', '.join(map(repr, self.choices))})")


# Configure basic logging for CLI (will be overridden by main.py)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "✦ <level>{message}</level>",
    colorize=True,
    level="INFO",
)


def _print_licenses_and_exit(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print the bundled third-party NOTICES.txt (if present) and exit.

    The cli-v2 spec (§4) requires a ``--licenses`` flag that prints all
    third-party attributions and exits without starting the server. The
    NOTICES.txt file is generated per release in Phase 8b; when the file
    is absent (e.g. running from a source checkout outside a release
    tarball), a fallback message is printed so the flag never crashes
    the CLI.
    """
    if not value or ctx.resilient_parsing:
        return
    notices = Path(__file__).resolve().parent.parent / "NOTICES.txt"
    if notices.exists():
        click.echo(notices.read_text())
    else:
        click.echo(
            "NOTICES.txt not bundled in this build — run from a release tarball "
            "to see full third-party license attributions. The top-level NOTICE "
            "file describes the upstream fork relationship."
        )
    ctx.exit(0)


@click.group()
@click.version_option(
    version=__version__,
    message="""
✨ %(prog)s - OpenAI Compatible API Server for MLX models ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Version: %(version)s
""",
)
@click.option(
    "--licenses",
    is_flag=True,
    callback=_print_licenses_and_exit,
    expose_value=False,
    is_eager=True,
    help="Print bundled third-party license attributions and exit.",
)
def cli():
    """Top-level Click command group for the MLX server CLI.

    Subcommands (such as ``launch``) are registered on this group and
    invoked by the console entry point.
    """


@cli.command()
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to a YAML config file for multi-handler mode. "
    "When provided, --model-path and other per-model flags are ignored.",
)
@click.option(
    "--model-path",
    "--model",
    "model_path",
    required=False,
    default=None,
    help=(
        "HuggingFace repo ID or local path to the model. "
        "``--model`` is the cli-v2 spec name; ``--model-path`` is the legacy alias."
    ),
)
@click.option(
    "--model-type",
    default="lm",
    type=click.Choice(["lm", "embeddings"]),
    help="Type of model to run (lm: text-only, embeddings: text embeddings)",
)
@click.option(
    "--context-length",
    default=None,
    type=int,
    help="Context length for language models. If not specified, uses model default. Only works with the `lm` model type.",
)
@click.option(
    "--served-model-name",
    default=None,
    type=str,
    help="Override the model name returned by /v1/models and accepted in request 'model' field. Defaults to model_path if not set.",
)
@click.option("--port", default=8000, type=int, help="Port to run the server on")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help=("Host to bind. Defaults to loopback per cli-v2 contract; set to 0.0.0.0 to expose on all interfaces."),
)
@click.option("--queue-timeout", default=300, type=int, help="Request timeout in seconds")
@click.option("--queue-size", default=100, type=int, help="Maximum queue size for pending requests")
@click.option(
    "--idle-unload-seconds",
    default=0,
    type=int,
    help=(
        "When > 0, run the model in on-demand mode: load on first request and "
        "unload after this many seconds of inactivity. 0 (default) keeps the "
        "model resident for the life of the process. cli-v2 contract."
    ),
)
@click.option(
    "--disable-auto-resize",
    is_flag=True,
    help="Disable automatic model resizing. Only work for Vision Language Models.",
)
@click.option(
    "--log-file",
    default=None,
    type=str,
    help="Path to log file. If not specified, logs will be written to 'logs/app.log' by default.",
)
@click.option(
    "--no-log-file",
    is_flag=True,
    help="Disable file logging entirely. Only console output will be shown.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=UpperChoice(list(_LOG_LEVEL_CHOICES)),
    help=(
        "Logging level. cli-v2 spec accepts info|debug|warn|error (any case); "
        "warning and critical are also accepted for backward compatibility."
    ),
)
@click.option(
    "--enable-auto-tool-choice",
    is_flag=True,
    help="Enable automatic tool choice. Only works with language models.",
)
@click.option(
    "--tool-call-parser",
    default=None,
    type=click.Choice(sorted(set(TOOL_PARSER_MAP.keys()) | set(UNIFIED_PARSER_MAP.keys()))),
    help="Specify tool call parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--reasoning-parser",
    default=None,
    type=click.Choice(sorted(set(REASONING_PARSER_MAP.keys()) | set(UNIFIED_PARSER_MAP.keys()))),
    help="Specify reasoning parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Enable trust_remote_code when loading models. This allows loading custom code from model repositories.",
)
@click.option(
    "--chat-template-file",
    default=None,
    type=str,
    help="Path to a custom chat template file. Only works with language models (lm).",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for language models. Only works with language models (lm).",
)
@click.option(
    "--prompt-cache-size",
    default=10,
    type=int,
    help="Maximum number of prompt KV cache entries to store. Only works with language models (lm). Default is 10.",
)
@click.option(
    "--max-bytes",
    "prompt_cache_max_bytes",
    default=1 << 63,
    type=int,
    help="Maximum total bytes retained by prompt KV caches before eviction. Only works with language models (lm).",
)
@click.option(
    "--prompt-cache-dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    help=("Directory for disk-backed prompt KV cache payloads. Defaults to a process-local temporary directory."),
)
@click.option(
    "--draft-model-path",
    default=None,
    type=str,
    help="Path to the draft model for speculative decoding. Only supported with model type 'lm'. When set, --num-draft-tokens controls how many tokens the draft model generates per step.",
)
@click.option(
    "--num-draft-tokens",
    default=2,
    type=int,
    help="Number of draft tokens per step when using speculative decoding (--draft-model-path). Only supported with model type 'lm'. Default is 2.",
)
@click.option(
    "--kv-bits",
    default=None,
    type=int,
    help="Number of bits for KV cache quantization (e.g. 4, 8). Reduces memory usage at the cost of some quality. Only works with the 'lm' model type.",
)
@click.option(
    "--kv-group-size",
    default=64,
    type=int,
    help="Group size for KV cache quantization. Default is 64.",
)
@click.option(
    "--quantized-kv-start",
    default=0,
    type=int,
    help="Step to begin using a quantized KV cache when --kv-bits is set. Default is 0.",
)
# Continuous-batching concurrency (forwarded to mlx_lm.generate.BatchGenerator).
# Names mirror the flags exposed by mlx-lm's own HTTP server so operators who
# already know that tool can tune this one the same way.
@click.option(
    "--decode-concurrency",
    "--max-concurrency",
    "batch_completion_size",
    default=32,
    type=int,
    help=(
        "When a request is batchable, decode that many requests in parallel. "
        "``--max-concurrency`` is the cli-v2 spec name; ``--decode-concurrency`` "
        "is the legacy alias. Applies to the 'lm' model type. Default is 32."
    ),
)
@click.option(
    "--prompt-concurrency",
    "batch_prefill_size",
    default=8,
    type=int,
    help=(
        "When a request is batchable, prefill that many prompts in parallel. "
        "Applies to the 'lm' model type. Default is 8."
    ),
)
@click.option(
    "--prefill-step-size",
    "batch_prefill_step_size",
    default=2048,
    type=int,
    help=(
        "Maximum tokens processed per prefill step during batched generation. "
        "Applies to the 'lm' model type. Default is 2048."
    ),
)
@click.option(
    "--disable-batching",
    is_flag=True,
    default=False,
    help=("Disable continuous batching for LM models. Use this when per-request positive seeds must be honored."),
)
# Sampling parameters (defaults used when API request omits them)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="Default maximum number of tokens to generate. If omitted, uses model generation_config.json when available.",
)
@click.option("--temperature", default=None, type=float, help="Default sampling temperature.")
@click.option("--top-p", default=None, type=float, help="Default nucleus sampling (top-p) probability.")
@click.option("--top-k", default=None, type=int, help="Default top-k sampling parameter.")
@click.option("--min-p", default=None, type=float, help="Default min-p sampling parameter.")
@click.option(
    "--repetition-penalty",
    default=None,
    type=float,
    help="Default repetition penalty for token generation.",
)
@click.option(
    "--presence-penalty",
    default=None,
    type=float,
    help="Default presence penalty for token generation.",
)
@click.option(
    "--xtc-probability",
    default=None,
    type=float,
    help="Default XTC probability sampling parameter.",
)
@click.option(
    "--xtc-threshold",
    default=None,
    type=float,
    help="Default XTC threshold sampling parameter.",
)
@click.option("--seed", default=None, type=int, help="Default random seed for generation.")
@click.option(
    "--repetition-context-size",
    default=None,
    type=int,
    help="Default repetition context size parameter.",
)
def launch(
    config_file,
    model_path,
    model_type,
    context_length,
    served_model_name,
    port,
    host,
    queue_timeout,
    queue_size,
    idle_unload_seconds,
    disable_auto_resize,
    log_file,
    no_log_file,
    log_level,
    enable_auto_tool_choice,
    tool_call_parser,
    reasoning_parser,
    trust_remote_code,
    chat_template_file,
    debug,
    prompt_cache_size,
    prompt_cache_max_bytes,
    prompt_cache_dir,
    draft_model_path,
    num_draft_tokens,
    kv_bits,
    kv_group_size,
    quantized_kv_start,
    batch_completion_size,
    batch_prefill_size,
    batch_prefill_step_size,
    disable_batching,
    max_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    repetition_penalty,
    presence_penalty,
    xtc_probability,
    xtc_threshold,
    seed,
    repetition_context_size,
) -> None:
    """Start the FastAPI/Uvicorn server with the supplied flags.

    When ``--config`` is provided the server launches in multi-handler
    mode, loading all models defined in the YAML file. In this mode
    per-model CLI flags (``--model-path``, ``--model-type``, etc.) are
    ignored.

    Otherwise the command builds a single-model ``MLXServerConfig``
    and calls the async ``start`` routine.
    """
    # ---- Multi-handler mode ----
    if config_file is not None:
        logger.info(f"Loading multi-handler config from: {config_file}")
        try:
            multi_config = load_config_from_yaml(config_file)
        except (FileNotFoundError, ValueError) as e:
            raise click.BadParameter(str(e), param_hint="'--config'") from e
        asyncio.run(start_multi(multi_config))
        return

    # ---- Single-handler mode (original behavior) ----
    if model_path is None:
        raise click.UsageError(
            "Either --config (multi-handler YAML) or --model / --model-path (single model) is required."
        )

    args = MLXServerConfig(
        model_path=model_path,
        model_type=model_type,
        context_length=context_length,
        served_model_name=served_model_name,
        port=port,
        host=host,
        queue_timeout=queue_timeout,
        queue_size=queue_size,
        # cli-v2 --idle-unload-seconds: > 0 flips on-demand load/unload.
        on_demand=idle_unload_seconds > 0,
        on_demand_idle_timeout=idle_unload_seconds or 60,
        disable_auto_resize=disable_auto_resize,
        log_file=log_file,
        no_log_file=no_log_file,
        log_level=_resolve_log_level(log_level),
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        trust_remote_code=trust_remote_code,
        chat_template_file=chat_template_file,
        debug=debug,
        prompt_cache_size=prompt_cache_size,
        prompt_cache_max_bytes=prompt_cache_max_bytes,
        prompt_cache_dir=prompt_cache_dir,
        draft_model_path=draft_model_path,
        num_draft_tokens=num_draft_tokens,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        batch_completion_size=batch_completion_size,
        batch_prefill_size=batch_prefill_size,
        batch_prefill_step_size=batch_prefill_step_size,
        disable_batching=disable_batching,
        default_max_tokens=max_tokens,
        default_temperature=temperature,
        default_top_p=top_p,
        default_top_k=top_k,
        default_min_p=min_p,
        default_repetition_penalty=repetition_penalty,
        default_presence_penalty=presence_penalty,
        default_xtc_probability=xtc_probability,
        default_xtc_threshold=xtc_threshold,
        default_seed=seed,
        default_repetition_context_size=repetition_context_size,
    )

    # Single-model launches always run through the HandlerProcessProxy
    # subprocess path — same isolation used by multi-model YAML mode — to
    # avoid the MLX Metal command-buffer race (``Completed handler
    # provided after commit call``) that crashes the in-process path when
    # the continuous batcher runs on a thread other than the one that
    # loaded the model. See https://github.com/ml-explore/mlx/issues/2457.
    asyncio.run(start_multi(args.to_multi_model_server_config()))
