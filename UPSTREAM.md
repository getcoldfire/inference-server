# Upstream Tracking

This repository is a fork of [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server).

**Forked at:** `4b7d4b61586dd08b5ca1149b73003e85272ff652` (upstream version 1.8.1, 2026-05-31)
**Deviation policy:** No automatic upstream sync. Patches reviewed manually on Coldfire's schedule.
**Why this fork exists:** See `docs/LICENSING.md` and the Coldfire spec at `docs/superpowers/specs/2026-06-03-mlx-openai-server-fork-spec.md` in the private monorepo.

## Inherited upstream failures (resolved)

At fork point, `pytest tests/ --tb=line` on the audited commit returned **125 passed, 14 failed,
4 skipped, 19 warnings**. All 14 failures were test-side mock/fake objects whose signatures drifted
from production code — production behavior was correct, but the in-test fakes were not updated to
match.

All 14 inherited failures were fixed in this fork. See the commit
`fix(tests): update inherited upstream fakes to match production signatures` for full details.
The three categories of fixes applied were:

1. **`_FakeLRU.fetch_nearest_cache`** — 4 classes in `test_batch_scheduler.py` updated to accept
   the `allowed_sources` keyword-only parameter added to `LRUPromptCache.fetch_nearest_cache`.
2. **`_FakeModel` and `_FakePromptCache`** — `test_mixed_think_tool_handoff_stream_handler_integration.py`
   updated: `_FakeModel` gained `has_draft_model`, `cache_is_batchable`, and `cache_is_trimmable`
   class attributes; `_FakePromptCache.fetch_nearest_cache` gained the `allowed_sources` kwarg;
   a `_configure_handler_stubs` helper added `_generation_lock`, `_batch_scheduler`, and
   `_batch_scheduler_lock` to the bypass-`__init__` handler instances.
3. **`_prepare_text_request` attribute access** — 2 tests in `test_chat_completions_prompt_history.py`
   that use `object.__new__` to bypass `__init__` now set `kv_bits`, `kv_group_size`,
   `quantized_kv_start`, and `_disable_batching` on the handler before calling the method under test.

Post-fix baseline: **282 passed, 4 skipped, 0 failed, 19 warnings** (excluding `test_lifecycle.py`
which terminates the process via SIGTERM — a pre-existing test isolation issue unrelated to these
changes).

### Install-command note

The plan's command `pip install -e ".[dev]"` does not install the dev dependencies, because upstream
declares them under PEP 735 `[dependency-groups]` (not `[project.optional-dependencies]`). The
working invocation is:

```bash
pip install -e .
pip install --group dev   # requires pip >= 25.1
```

The Coldfire Phase 8a `Makefile` work should encode this so contributors do not hit the same
trap.

## Coldfire-specific endpoints (not in upstream cubist38/mlx-openai-server)

- `POST /admin/models/add` (since v0.1.1) — hot-add an on-demand model.
  v0.1.1 accepts `on_demand: true` only; resident hot-add deferred to v0.1.2.
- `DELETE /admin/models/{model_id:path}` (since v0.1.1) — hot-remove. Path
  converter accepts slash-containing IDs. 409 if model is mid-request.

Loopback-only at v0.1.1; no admin auth. Required by cli-v2 Phase 8 for
`coldfire-ctl models install/remove` without restarting the fork subprocess.
Not yet submitted to upstream cubist38.

## Coldfire-specific CLI commands (not in upstream cubist38/mlx-openai-server)

- `coldfire-inference-server models list` (since v0.2.0) — list local HF cache contents; MLX-shape filter by default, `--all` shows everything; `--json` for machine output; STATUS column probes a running fork on `--port`.
- `coldfire-inference-server models pull <hf-id>` (since v0.2.0) — download a model without registering. Conservative allowlist (`*.safetensors`, `*.json`, `tokenizer*`, `*.txt`) with `--include`/`--exclude` overrides. Warns on non-MLX-shaped repos but completes the download.
- `coldfire-inference-server models rm <hf-id>` (since v0.2.0) — delete a cache directory. Refuses by default if the model is currently being served; `--force` overrides but stays cache-only (does not unregister).

These are local-filesystem operations only — no HTTP surface, no admin endpoints. They complement the v0.1.1 hot-add admin endpoints (which manage the running server's registration state) by exposing the orthogonal cache dimension.
