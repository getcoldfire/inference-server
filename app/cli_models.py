"""Click subgroup `coldfire-inference-server models` — local HuggingFace cache management.

Subcommands:
  - models list   (Task 4)
  - models pull   (Task 5)
  - models rm     (Task 6)

Wired into the main `cli` group via `cli.add_command(models)` in app/cli.py.
"""

from __future__ import annotations

import json as _json
import os
import shutil
from datetime import UTC, datetime

import click
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from app.utils.hf_cache import MLX_SUPPORTED_MODEL_TYPES, cache_path_for, list_cached_models
from app.utils.server_probe import is_model_serving, serving_model_ids


@click.group(name="models")
def models() -> None:
    """Manage the local HuggingFace cache (list, pull, rm).

    Operates entirely on the local filesystem — no interaction with a
    running coldfire-inference-server. The three commands are CLI utilities,
    not service operations.
    """


def _human_bytes(n: int) -> str:
    """Render byte count as 'NUMBER UNIT' in SI (base-10) units.

    macOS Finder uses base-10; storage labels do too (a "1 TB drive" =
    10^12 bytes). HF's size_on_disk is raw bytes. Whole-number values
    render without a trailing '.0' so 712_000_000 -> '712 MB' not '712.0 MB'.
    """
    if n < 1000:
        return f"{n} B"
    f = float(n)
    # PB-scale caches are out of scope for v0.2.0 — the TB branch catches
    # any size beyond 1000 GB and stays in TB units.
    for unit in ("KB", "MB", "GB", "TB"):
        f /= 1000.0
        if f < 1000 or unit == "TB":
            if f == int(f):
                return f"{int(f)} {unit}"
            return f"{f:.1f} {unit}"
    # Unreachable: the TB branch above always exits the loop. Belt-and-
    # suspenders only — Python type-checkers are happier with an
    # explicit terminal return.
    return f"{f:.1f} TB"  # pragma: no cover


def _relative_time(when: datetime | None) -> str:
    if when is None:
        return "never"
    delta = datetime.now(tz=UTC) - when
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} hour{'s' if secs // 3600 != 1 else ''} ago"
    days = secs // 86400
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 60:
        return f"{days // 7} week{'s' if days // 7 != 1 else ''} ago"
    return f"{days // 30} month{'s' if days // 30 != 1 else ''} ago"


@models.command("list")
@click.option("--all", "show_all", is_flag=True, help="Show every cached HF repo, not just MLX-shaped ones.")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON instead of the table.")
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="Port to probe for the 'serving' STATUS column. "
    "Default 8000 matches `coldfire-inference-server launch --port`. "
    "cli-v2 daemon-launched forks listen on 11435 — pass --port 11435 there.",
)
def models_list(show_all: bool, as_json: bool, port: int) -> None:
    """List models in the local HuggingFace cache.

    By default shows only MLX-shaped models. Use --all to include
    every cached repo (Sentence-Transformers BERTs etc.). The STATUS
    column shows 'serving' if a coldfire-inference-server on 127.0.0.1:<port>
    advertises the model via /v1/models — defaults to port 8000.

    Note: STATUS matches against the fork's /v1/models `id` field. If a
    model was registered with a served_model_name alias, the table may
    show '-' for the cache row even when that model is being served
    under the alias — the cache name and the alias are different strings.
    """
    rows = list_cached_models(mlx_only=not show_all)
    # Single probe for STATUS — fetch /v1/models once, check each row
    # against the returned set. Avoids N * 500ms wait for empty caches.
    served = serving_model_ids(port=port)
    annotated = [(r, r.name in served) for r in rows]

    if as_json:
        out = [
            {
                "name": r.name,
                "size_bytes": r.size_bytes,
                "last_used": r.last_used.isoformat() if r.last_used else None,
                "is_mlx": r.is_mlx,
                "serving": serving,
            }
            for r, serving in annotated
        ]
        click.echo(_json.dumps(out, indent=2))
        return

    # Human table.
    if not annotated:
        total_str = "0 B"
    else:
        total_str = _human_bytes(sum(r.size_bytes for r, _ in annotated))

    header = f"{'NAME':<52} {'SIZE':>10}  {'LAST USED':<14} STATUS"
    click.echo(header)
    for r, serving in sorted(annotated, key=lambda x: x[0].name):
        status = "serving" if serving else "-"
        click.echo(f"{r.name:<52} {_human_bytes(r.size_bytes):>10}  {_relative_time(r.last_used):<14} {status}")
    click.echo()
    click.echo(f"Total: {total_str} across {len(annotated)} models in ~/.cache/huggingface/hub")


_DEFAULT_PULL_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.json",
    "tokenizer*",
    "*.txt",
)


def _looks_mlx_from_hub(hf_id: str) -> bool:
    """Best-effort pre-download MLX-shape check via the Hub's config.json.

    Used by `models pull` to decide whether to warn. Skipped when the
    repo is already fully cached locally (we'd be doing a redundant
    HEAD/etag round-trip; the cached `is_mlx_shaped` is just as good).

    Falls back to True (skip warning) on any error so we don't block
    legitimate downloads when the Hub is briefly unreachable.
    """
    if hf_id.startswith("mlx-community/"):
        return True
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id=hf_id, filename="config.json")
        with open(cfg_path) as fh:
            cfg = _json.loads(fh.read())
    except Exception:
        return True  # be charitable on error — let download proceed
    if "quantization" in cfg:
        return True
    return cfg.get("model_type", "") in MLX_SUPPORTED_MODEL_TYPES


@models.command("pull")
@click.argument("hf_id")
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress progress banners AND HF's own tqdm progress bars "
    "(by setting HF_HUB_DISABLE_PROGRESS_BARS=1 for this command).",
)
@click.option(
    "--include",
    "include",
    multiple=True,
    help="Additional glob pattern to include in the download. Repeatable.",
)
@click.option(
    "--exclude",
    "exclude",
    multiple=True,
    help="Glob pattern to remove from the allowlist. Repeatable.",
)
def models_pull(hf_id: str, quiet: bool, include: tuple[str, ...], exclude: tuple[str, ...]) -> None:
    """Download a model from HuggingFace into the local cache.

    Doesn't register the model with any running fork — use
    `coldfire-ctl models install` for that. This command is the
    "pre-warm the cache, don't disturb the server" companion.

    Default download allowlist: *.safetensors, *.json, tokenizer*, *.txt.
    Add patterns with --include; remove with --exclude.
    """
    # --quiet also disables huggingface_hub's tqdm progress bars
    # (not controllable via snapshot_download kwarg).
    if quiet:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Skip the pre-flight MLX-shape check if the repo is already in
    # the local cache — we'd otherwise be doing a redundant HF round-
    # trip. Cached repos are already known to the operator; if they
    # weren't MLX-shaped they would have seen the warning on first pull.
    already_cached = cache_path_for(hf_id) is not None
    if already_cached:
        is_mlx = True  # don't warn redundantly
    else:
        is_mlx = _looks_mlx_from_hub(hf_id)

    if not is_mlx:
        click.echo(
            f"⚠ {hf_id} doesn't look MLX-quantized — coldfire-inference-server "
            f"will fail to load it. Consider mlx-community/* alternatives.",
            err=True,
        )

    patterns = list(_DEFAULT_PULL_PATTERNS) + list(include)
    patterns = [p for p in patterns if p not in set(exclude)]

    if not quiet:
        click.echo(f"Downloading {hf_id} ...")

    try:
        path = snapshot_download(repo_id=hf_id, allow_patterns=patterns)
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        click.echo(f"error: {e}", err=True)
        raise SystemExit(1) from e
    except OSError as e:
        # No space left on device etc.
        click.echo(f"error: {e}", err=True)
        raise SystemExit(2) from e

    if not quiet:
        click.echo(f"✓ {hf_id} cached at {path}")
    if not is_mlx:
        click.echo(
            "⚠ Reminder: this repo is not MLX-quantized; `models install` of this id will fail.",
            err=True,
        )


def _lookup_size_bytes(hf_id: str) -> int:
    """Return huggingface_hub.scan_cache_dir's size_on_disk for hf_id,
    or 0 if not found. Used for the 'freed N GB' message in `rm`.

    Avoids a manual tree walk (and the Python 3.13-only
    `Path.is_file(follow_symlinks=)` we'd otherwise want for HF's
    blob-symlinks layout). The list is typically <100 entries.
    """
    for row in list_cached_models(mlx_only=False):
        if row.name == hf_id:
            return row.size_bytes
    return 0


@models.command("rm")
@click.argument("hf_id")
@click.option(
    "--force",
    is_flag=True,
    help="Delete even if the model is currently being served by a running fork.",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="Port to probe for the serving safety check. "
    "Default matches `coldfire-inference-server launch --port` (8000); "
    "cli-v2-daemon-launched forks listen on 11435.",
)
def models_rm(hf_id: str, force: bool, port: int) -> None:
    """Delete a model from the local HuggingFace cache.

    Refuses by default if a coldfire-inference-server on 127.0.0.1:<port>
    is currently advertising the model — stop the server first or
    pass --force. This is a cache-only operation; it does NOT
    unregister the model from a running fork. To remove the
    registration AND the cache entry, use `coldfire-ctl models remove`.
    """
    path = cache_path_for(hf_id)
    if path is None:
        click.echo(f"model {hf_id!r} is not in the local cache", err=True)
        raise SystemExit(1)

    if not force and is_model_serving(hf_id, port=port):
        click.echo(
            f"refusing to remove {hf_id!r}: currently being served by "
            f"coldfire-inference-server on port {port}. Stop the server or "
            f"pass --force to delete anyway.",
            err=True,
        )
        raise SystemExit(1)

    # Use HF's precomputed size_on_disk instead of walking the tree
    # ourselves — same scan_cache_dir() call already used by cache_path_for.
    size_bytes = _lookup_size_bytes(hf_id)
    shutil.rmtree(path)
    click.echo(f"✓ Removed {hf_id} (freed {_human_bytes(size_bytes)})")
