"""Click subgroup `coldfire-mlx-server models` — local HuggingFace cache management.

Subcommands (added in subsequent tasks):
  - models list   (Task 4)
  - models pull   (Task 5)
  - models rm     (Task 6)

Wired into the main `cli` group via `cli.add_command(models)` in app/cli.py.
"""

from __future__ import annotations

import json as _json
from datetime import UTC, datetime

import click

from app.utils.hf_cache import list_cached_models
from app.utils.server_probe import serving_model_ids


@click.group(name="models")
def models() -> None:
    """Manage the local HuggingFace cache (list, pull, rm).

    Operates entirely on the local filesystem — no interaction with a
    running coldfire-mlx-server. The three commands are CLI utilities,
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
    for unit in ("KB", "MB", "GB", "TB"):
        f /= 1000.0
        if f < 1000 or unit == "TB":
            if f == int(f):
                return f"{int(f)} {unit}"
            return f"{f:.1f} {unit}"
    return f"{f:.1f} TB"


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
    "Default 8000 matches `coldfire-mlx-server launch --port`. "
    "cli-v2 daemon-launched forks listen on 11435 — pass --port 11435 there.",
)
def models_list(show_all: bool, as_json: bool, port: int) -> None:
    """List models in the local HuggingFace cache.

    By default shows only MLX-shaped models. Use --all to include
    every cached repo (Sentence-Transformers BERTs etc.). The STATUS
    column shows 'serving' if a coldfire-mlx-server on 127.0.0.1:<port>
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
