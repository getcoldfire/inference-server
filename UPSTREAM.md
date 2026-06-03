# Upstream Tracking

This repository is a fork of [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server).

**Forked at:** `4b7d4b61586dd08b5ca1149b73003e85272ff652` (upstream version 1.8.1, 2026-05-31)
**Deviation policy:** No automatic upstream sync. Patches reviewed manually on Coldfire's schedule.
**Why this fork exists:** See `docs/LICENSING.md` and the Coldfire spec at `docs/superpowers/specs/2026-06-03-mlx-openai-server-fork-spec.md` in the private monorepo.

## Known upstream failures at fork point

Baseline run of `pytest tests/ --tb=line` on the audited commit (Python 3.12.13, macOS arm64, MLX
installed) returned **125 passed, 14 failed, 4 skipped, 19 warnings**. All 14 failures are
test-side mock/fake objects whose signatures have drifted from production code — production behavior
appears correct, but the in-test fakes were not updated to match. None touch the code paths Coldfire
intends to modify (audio / VLM / image-gen stripping; embedding-service replacement). Candidates for
an upstream PR; **not** in scope for this fork.

Failing tests:

- `tests/test_batch_scheduler.py::test_exact_cache_hit_is_backed_off_before_kickoff_token`
- `tests/test_batch_scheduler.py::test_exact_non_trimmable_cache_hit_logs_info`
  - Both fail with: `_FakeLRU.fetch_nearest_cache() got an unexpected keyword argument 'allowed_sources'`
- `tests/test_chat_completions_prompt_history.py::test_prepare_text_request_strips_reasoning_content_from_prior_assistant_messages`
- `tests/test_chat_completions_prompt_history.py::test_prepare_text_request_strips_reasoning_content_from_tool_call_assistant_messages`
  - Both fail with: `'_FakeModel' object has no attribute 'has_draft_model'`
- `tests/test_mixed_think_tool_handoff_stream_handler_integration.py` (10 tests in
  `MixedThinkToolHandoffStreamHandlerIntegrationTests`):
  - `test_mixed_think_tool_handoff_inside_thinking_reenters_reasoning_after_tool_parse`
  - `test_mixed_think_tool_handoff_terminal_tool_call_before_thinking_close`
  - `test_nonstream_qwen3_moe_tool_fallback_does_not_leak_synthetic_reasoning_prefix`
  - `test_nonstream_step35_hides_reasoning_when_open_tag_missing_but_close_tag_present`
  - `test_nonstream_step35_parses_tool_call_when_response_starts_with_stray_think_close`
  - `test_stream_mixed_think_parser_preserves_literal_text_when_open_tag_is_missing`
  - `test_stream_step35_hides_reasoning_when_open_tag_missing_but_close_tag_present`
  - `test_stream_step35_parses_tool_call_when_open_marker_is_split_as_too_plus_l_call`
  - `test_stream_step35_parses_tool_call_when_output_starts_with_stray_think_close`
  - `test_stream_step35_preserves_split_parameter_close_marker_inside_tool_call`

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
