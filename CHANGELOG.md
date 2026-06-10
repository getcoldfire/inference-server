# Changelog

## v0.2.1 — 2026-06-10

### Bug fixes

- **Stderr-only logging** (`app/server.py`, `app/cli.py`, `app/main.py`): Dropped the
  rotating file handler that wrote to a relative `logs/app.log` path. Under macOS
  launchd the inherited cwd is `/`, so the path resolved to `/logs/app.log` on a
  read-only filesystem, crashing the server at startup with `[Errno 30]`. The parent
  daemon already captures stderr and routes lines through its own structured logger;
  no separate file is needed. Removed the `--log-file` and `--no-log-file` CLI flags
  accordingly. `configure_logging()` still accepts those kwargs for call-site
  compatibility but ignores them.

- **Zero-models config accepted** (`app/config.py`): Removed the `len(models_raw) == 0`
  guard that rejected `models: []` with "must be a non-empty list". The
  `/admin/models/load` endpoint (added in v0.1.1) makes empty-then-hot-add a
  first-class lifecycle: operators can start the server with no models and hot-add
  via the admin API. An empty list now boots cleanly; `/v1/models` returns
  `{"object":"list","data":[]}` and `/v1/chat/completions` returns 404 for any
  unknown model.

## v0.2.0 — 2026-06-07

- Cache management CLI: `models list`, `models pull`, `models rm` subcommands.
  `models list` probes a running server on `--port` for live STATUS column.
  `models pull` downloads model weights without registering. `models rm` deletes
  a local cache directory (refuses if the model is currently served, unless `--force`).
