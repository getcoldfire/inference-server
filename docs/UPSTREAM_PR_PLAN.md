# Upstream Contribution Plan: MLX Stream-Affinity Bug

**Status:** Drafted — not yet filed. **Plan revised 2026-06-05 after prior-art search; see "Prior Art" section below before acting on the original three-path proposal.**
**Date drafted:** 2026-06-05
**Coldfire fork commits implementing the fix:**
- `8547c67` — BatchScheduler warm-up + main-loop stream wrap (chat path)
- `c90e01d` — EmbeddingService warm-up (embeddings path), plus nomic-bert weight remap (separate concern; NOT being upstreamed — that's Coldfire-specific)

---

## Prior Art (added 2026-06-05) — READ FIRST

A search of `ml-explore/mlx`, `ml-explore/mlx-lm`, and `cubist38/mlx-openai-server` turned up substantial existing reporting on this bug. The original three-path plan below was drafted before this search; **most of it should not be executed as written.** The revised recommendation lives in the new [Revised Status & Next Steps](#revised-status--next-steps-supersedes-original-status--next-steps-section) section at the bottom.

### What already exists

**`ml-explore/mlx-lm#1256` (OPEN, filed 2026-05-07)** — duplicate of what we would file.
Same `RuntimeError: There is no Stream(gpu, 1) in current thread`, same `generate.py:1161` line, same stack trace. Author's root-cause diagnosis matches ours (module-level `generation_stream = mx.new_thread_local_stream(...)` binds to the importing thread). A vllm-mlx contributor confirmed in comments that it's not sliding-window-specific — any code path running generation on a non-loader thread hits it.
Link: https://github.com/ml-explore/mlx-lm/issues/1256

**`ml-explore/mlx-lm#1275` (OPEN, filed 2026-05-14, awaiting review, last touched 2026-06-04)** — proposed fix for #1256.
Makes `generation_stream` a `threading.local()`-backed factory. 6 call sites updated. Includes `test_batch_sliding_window_threaded` covering the exact scenario. **Not yet merged.** If this lands, the `mlx_lm.server`-style symptom goes away — but our path (cubist38's `BatchScheduler` with its own per-worker stream) may still need the warm-up because the failure mode is lazy *model state* binding to a thread, not just the module-level stream.
Link: https://github.com/ml-explore/mlx-lm/pull/1275

**Closed history (so we understand maintainer cadence):**
- `mlx-lm#1088` (closed Apr 2) — same `threading.local()` fix, closed in favor of #1090
- `mlx-lm#1090` (MERGED Apr 22) — went with `ThreadLocalStream` instead. This is the "partial fix in v0.31.3" that #1256 explains is insufficient
- `mlx-lm#1182` (closed Apr 22) — BatchGenerator-specific variant, also closed in favor of #1090

**`ml-explore/mlx#3529` (CLOSED, 2026-05-11)** — Apple's stance on the root cause.
Identical cross-thread lazy-eval failure. Maintainer **@zcbenz** replied:

> "This is expected behavior because we don't support evaluating an array created in another thread. Two solutions: (1) inherit from `nn.Module` and call `mx.eval(model.parameters())` before passing to the worker, (2) create the class in the worker."

**Our warm-up is essentially solution (1)** plus a forward pass to also force prompt-cache scaffolding to materialize. So Apple won't accept a new bug report — they consider this a documented contract, not a bug. Any framing that calls it a "bug in mlx" will be rejected.
Link: https://github.com/ml-explore/mlx/issues/3529

Also notable: `mlx#3391` (closed Apr 13) — auto-register-stream PR rejected by zcbenz with rationale about future queue semantics.

**`cubist38/mlx-openai-server#290` (CLOSED, 2026-04-28)** — same bug reported against cubist38 directly.
Multiple users (Qwen3.6-27B, Qwen3.6-35B-A3B, Darwin-36B, Gemma 4). Closed by **PR #295** (MERGED 2026-05-03) "Fix/lm thread affinity and batching policy" — added `--disable-batching`, gave `InferenceWorker` its own `new_thread_local_stream`, partitioned prompt-cache reuse by batch/nonbatch path, moved cache persistence onto the worker thread.

**But #290 was closed prematurely.** Comment from `therealmobasha` on **2026-05-14**: *"Still running into this issue with gemma4 models even when using `--disable-batching`."* This matches our finding exactly — stream-management alone isn't enough; the lazy *model state* (notably downstream of `set_wired_limit`) also binds to whichever thread first triggers it, and only a forward-pass warm-up resolves that.
Links: https://github.com/cubist38/mlx-openai-server/issues/290 , https://github.com/cubist38/mlx-openai-server/pull/295

### What this changes about our plan

1. **Do not file a new mlx-lm issue.** It duplicates #1256, which already has the same diagnosis and a community-confirmed scope. Filing a duplicate would split the conversation and likely get closed-as-duplicate.

2. **Do not file a new mlx (core) issue.** Apple's position is documented in #3529. Reframing what is, per the maintainer, expected behavior as a bug will be rejected and may damage future credibility with the team.

3. **Path B (cubist38 issue) should be reframed as a follow-up to closed #290**, not a fresh report. We have specific new information: (a) #295's fix is incomplete (point at the May 14 comment + our reproduction), (b) the structural fix is loader-thread warm-up of the model's lazy state, not just stream-management.

4. **Path C (cubist38 PR) is the only path that ships net-new code.** Reframe the PR description to "completes the work begun in #295" — credit the prior PR, explain why per-thread streams alone don't cover the lazy-state path, position the warm-up as the structural complement.

5. **Comment on mlx-lm#1256 and PR #1275** with our additional repro: that cubist38's `BatchScheduler` (with its own per-worker `new_thread_local_stream`) still fails *even if* PR #1275 lands, because the bug is in lazy *model* state, not the module-level stream. This is genuinely additive context for the mlx-lm maintainers reviewing #1275.

The full prior-art-aware recommendation is in [Revised Status & Next Steps](#revised-status--next-steps-supersedes-original-status--next-steps-section) at the bottom. Everything between here and there is the original draft, preserved as reference content (the technical analysis and diagnostic content are still correct and reusable in the new framing).

---

## Executive Summary

While building the `getcoldfire/inference-server` fork's integration test suite, we discovered that **chat completions through `BatchScheduler` raise `RuntimeError: There is no Stream(gpu, N) in current thread.` on every request** in pristine upstream `cubist38/mlx-openai-server@4b7d4b6`. The bug is reproducible against bare upstream with zero Coldfire modifications.

Root cause is in `ml-explore/mlx-lm`: model forward passes trigger lazy MLX allocations that bind to whichever thread first touches them. When the scheduler thread is that first thread, later cross-thread evaluations fail because MLX requires the stream to be resident in the current thread.

We have a working fix (warm up the model on the loader thread before spawning workers) that's surgical (~30 LOC in BatchScheduler), well-validated (all 7 integration smoke tests pass; unit baseline preserved), and generally applicable (the same fix unblocked our embedding service after the same bug surfaced there too).

This document captures everything needed to submit:
- An issue to `ml-explore/mlx-lm` describing the root cause
- An issue + PR to `cubist38/mlx-openai-server` shipping the workaround

We are NOT submitting today — this is a plan and a content cache.

---

## Background — How We Found It

The Coldfire fork added an integration test suite (Phase 6 of our buildout plan) that boots a real server with a real `mlx-community/Llama-3.2-1B-Instruct-4bit` model and exercises chat completions end-to-end. Three tests failed identically on first run:

- `test_non_streaming_completion`
- `test_streaming_completion`
- `test_concurrent_no_kv_contamination` (4 concurrent streaming requests)

Server-side traceback (identical across all three):

```
File ".../app/core/batch_scheduler.py", line 494, in _run
    prompt_responses, gen_responses = self._batch_generator.next()
File ".../mlx_lm/generate.py", line 1855, in next
    return self._next()
File ".../mlx_lm/generate.py", line 1841, in _next
    self._prompt_batch.prompt(prompts)
File ".../mlx_lm/generate.py", line 1161, in prompt
    mx.eval([c.state for c in self.prompt_cache])
RuntimeError: There is no Stream(gpu, 1) in current thread.
```

We pinned the fork to upstream SHA `4b7d4b6` and built a clean test environment against pristine upstream (no Coldfire patches applied) by pointing `PYTHONPATH` at the upstream worktree. The bug reproduces 1:1 — same error, same line, same stack trace.

**Conclusion:** the bug exists in upstream and Coldfire did not introduce it. Upstream simply has no integration tests for chat that hit the BatchScheduler under a real model, so it was never caught.

---

## The Bug

### Symptom
Every chat completion request through `BatchScheduler` returns HTTP 500 with:

```
{"detail":{"error":{"message":"Failed to generate text response: There is no Stream(gpu, N) in current thread.","type":"server_error","param":null,"code":"500"}}}
```

Where `N` is the device index (typically `1` on Apple Silicon when the scheduler thread's default stream differs from the main thread's).

### Reproduction (minimal, no server required)

```python
# Requires: pip install mlx mlx-lm
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx
import threading

# 1. Load a model on the main thread.
model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit", lazy=False)

# 2. WITHOUT warm-up: kick off a worker thread that allocates a prompt cache
#    and runs a forward pass. This is what BatchScheduler does on its dedicated
#    scheduler thread.
def worker():
    s = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(s):
        c = make_prompt_cache(model)
        ids = mx.array([tok.encode("hi")])
        mx.eval(model(ids, cache=c), [x.state for x in c])

t = threading.Thread(target=worker)
t.start()
t.join()
# -> RuntimeError: There is no Stream(gpu, 1) in current thread.
```

### Workaround proof — main-thread warm-up resolves it

```python
# Same setup as above.
model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit", lazy=False)

# 3. WITH warm-up on the loader thread BEFORE spawning the worker:
def warm_up():
    c = make_prompt_cache(model)
    ids = mx.array([tok.encode("hi")])
    mx.eval(model(ids, cache=c), [x.state for x in c])

warm_up()  # forces lazy allocations to resolve on the main/loader thread

# Now the worker thread succeeds.
def worker():
    s = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(s):
        c = make_prompt_cache(model)
        ids = mx.array([tok.encode("hi")])
        mx.eval(model(ids, cache=c), [x.state for x in c])

threading.Thread(target=worker).start()  # OK
```

### Impact

Anyone running `cubist38/mlx-openai-server` for chat completions hits this immediately under any threaded scheduler. The only reason it isn't a known issue is upstream's lack of chat integration tests — manual smoke testing in upstream's CI happens via direct mlx-lm scripts that don't exercise the `BatchScheduler` path.

Severity: **critical** for anyone deploying `mlx-openai-server` for chat workloads. Not affecting embedding-only workloads on the same server because the embedding path also hits this (we found out the hard way when fixing the chat path unblocked the embedding path's same bug — see [`EmbeddingService` warm-up](#embeddingservice-warm-up-cubist38-also-applies)).

---

## Root Cause Analysis

The bug is in `ml-explore/mlx-lm`, not in `cubist38/mlx-openai-server`. cubist38's `BatchScheduler` is just where the symptom surfaces.

**What mlx-lm does that creates the problem:**

`mlx_lm.utils.load(...)` returns a model with weights materialized, but downstream MLX state — specifically state related to `set_wired_limit` and various cache scaffolding — is **lazy**. The first forward pass on the model triggers MLX to allocate and configure these deferred resources. Critically, MLX binds these allocations to **whichever stream is active on whichever thread first triggers them**.

**Where things go wrong:**

In cubist38's `BatchScheduler`:
1. The main/loader thread calls `load(...)` to construct the model.
2. The scheduler thread is spawned. The scheduler creates a *thread-local* MLX stream (`mx.new_thread_local_stream(mx.default_device())`).
3. The scheduler thread's first inference attempt triggers the model's deferred allocations under that thread-local stream.
4. `mlx_lm.BatchGenerator._next()` correctly wraps `_next()` in `mx.stream(self._stream)` — that handles the forward pass itself.
5. **But** `BatchGenerator.insert_segments(...)` (called outside `_next()`) allocates new prompt-cache scaffolding, and various `make_prompt_cache(model)` paths inherit the lazy state from earlier. The stream binding on those arrays is whatever stream was active during the first lazy resolution.
6. When `mx.eval([c.state for c in self.prompt_cache])` runs at `mlx_lm/generate.py:1161`, MLX checks: "Is this array's stream resident in the current thread?" If the array got bound earlier under a different thread's stream — for example, if the scheduler thread starts handling a *different* prompt with a *fresh* `mx.stream(self._stream):` context — that resident check fails.

**Why obvious fixes don't work:**

We tried three approaches before landing on the warm-up:

| Attempt | What we did | Why it failed |
|---|---|---|
| Monkey-patch `generation_stream` | Set `sys.modules['mlx_lm.generate'].generation_stream = self._stream` in the scheduler | By the time we monkey-patch the module attribute, arrays have already been allocated with the old stream binding. The module attribute is read at *allocation time*, not at *evaluation time*. |
| Wrap `insert_segments` in `mx.stream(self._stream):` | Add a single-line wrap around the `BatchGenerator.insert_segments(...)` call in `_admit_pending` | Some lazy allocations are triggered *before* `insert_segments` is ever called (during model initial state resolution). The wrap moves the problem earlier but doesn't eliminate it. |
| Wrap the entire `_run` main loop in `mx.stream(self._stream):` | Broader wrap around the whole scheduler loop body | Same issue — the deferred allocations happen during the first forward pass, not at loop entry. The scheduler thread's stream context doesn't propagate back through model initialization. |

**Why warm-up works:**

If we run a one-token forward pass + `mx.eval` on the **loader thread** before the scheduler thread is spawned, MLX resolves every lazy allocation under the loader thread's default stream. After warm-up, only static, fully-resolved tensors exist; no deferred state remains. Every subsequent forward pass (on the scheduler thread, on the embedding worker thread, anywhere) operates against materialized tensors, and stream binding is no longer an issue.

The warm-up is essentially "force MLX to eagerly resolve at load time what it would otherwise lazily resolve at first use." Since "first use" is the threading problem, removing lazy resolution removes the problem.

---

## The Fix (Two Changes)

### 1. `BatchScheduler.start()` warm-up

Inserted at the top of `start()`, before the scheduler thread is spawned:

```python
def start(self) -> None:
    """Spawn the scheduler thread. Idempotent."""
    if self._thread is not None and self._thread.is_alive():
        return
    if not hasattr(mlx_generate, "BatchGenerator"):
        raise RuntimeError(
            "mlx_lm.generate.BatchGenerator is not available in the"
            " installed mlx-lm; upgrade mlx-lm to enable batching."
        )
    # Materialize the model's lazy MLX state on the *caller* thread
    # before spawning the scheduler thread. Without this warm-up, the
    # first forward pass on the scheduler thread raises
    # ``RuntimeError: There is no Stream(gpu, N) in current thread.``
    # because some of the model's deferred allocations (notably
    # downstream of ``set_wired_limit``) capture a stream reference
    # from the loading thread that the scheduler thread cannot
    # resolve. Running a one-token forward + ``mx.eval`` here forces
    # every lazy allocation to resolve on the loader thread, leaving
    # only static weights for the scheduler thread to read.
    self._warm_up_model()

    # ... existing code: spawn the scheduler thread ...
```

And the new method:

```python
def _warm_up_model(self) -> None:
    """One-token forward pass on the caller thread to materialize lazy state.

    See the rationale in ``start()``. Silently skipped for unit-test mocks
    that lack a real model graph (no ``.layers`` attribute).
    """
    if not hasattr(self._model, "layers"):
        return  # unit-test mock; skip
    try:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self._model)
        # Single BOS-or-pad token is enough to trigger every lazy alloc.
        token_id = getattr(self._tokenizer, "bos_token_id", None)
        if token_id is None:
            token_id = getattr(self._tokenizer, "pad_token_id", None) or 0
        ids = mx.array([[token_id]])
        logits = self._model(ids, cache=cache)
        mx.eval(logits, [c.state for c in cache])
    except Exception:
        # Warm-up failure shouldn't kill the scheduler — the real
        # forward pass will surface any actual model problem with a
        # cleaner traceback.
        pass
```

### 2. Defensive `mx.stream(self._stream)` wrap on the main loop

Belt-and-suspenders for any future MLX operations the scheduler triggers that aren't already wrapped by `BatchGenerator`. Matches the existing precedent in `_process_cancellations`:

```python
def _run(self) -> None:
    while not self._stop_event.is_set():
        with mx.stream(self._stream):
            # ... existing main loop body ...
```

The warm-up alone is sufficient to fix the bug; the wrap is insurance.

### 3. `EmbeddingService` warm-up (cubist38 also applies)

The same root-cause issue surfaces in the embedding service after the chat fix is applied. Same fix shape — call a one-input embed on the loader thread:

```python
class EmbeddingService:
    def __init__(self, model_dir: str | Path):
        # ... existing load logic ...
        self._warm_up()

    def _warm_up(self) -> None:
        """Materialize lazy MLX state on the caller thread.

        See BatchScheduler._warm_up_model for full rationale. Without this,
        the first embed() call on a worker thread fails with the same
        ``RuntimeError: There is no Stream(gpu, N) in current thread``.
        """
        try:
            self.embed(["a"])
        except Exception:
            pass
```

This applies to cubist38's embedding handler too. We're upstreaming both changes as one fix because they're the same bug.

---

## Three Upstream Contribution Paths

### Path A: File issue + PR to `ml-explore/mlx-lm` (root cause)

The "right" layer to fix this is in mlx-lm itself. Two options:

1. **Thread-localize the entire lazy-allocation pipeline.** Every deferred state in the model graph should resolve under the consumer's stream, not the loader's. This is a meaningful refactor and Apple's team would need to do it.
2. **Eagerly materialize all lazy state at `load(lazy=False)` time.** When `lazy=False` is passed, no deferred allocations should remain. This is conceptually simpler but changes the load-time vs. first-inference cost profile.

We can file a clear minimal-repro issue and offer to discuss approach with Apple. We probably can't submit a PR ourselves without deep mlx internals knowledge — too easy to get the architectural call wrong.

### Path B: File issue at `cubist38/mlx-openai-server` (symptom)

A separate issue at the layer where the symptom surfaces. Describes that any chat completion in their server fails on threaded scheduling. Links the mlx-lm issue. Lets cubist38 decide whether to merge a workaround now or wait for an upstream mlx-lm fix.

### Path C: PR to `cubist38/mlx-openai-server` (workaround)

Submit our actual fix as a PR. Two parts:
1. `_warm_up_model()` in `BatchScheduler`
2. `_warm_up()` in the embedding service handler
3. (Optional) the `mx.stream(self._stream):` main loop wrap, for defense in depth

**Recommended sequence: A → B → C.** File the root-cause issue at mlx-lm first (signal to Apple). Then file the symptom issue at cubist38 (lets them know). Then offer the workaround PR. This sequence shows good citizenship — we're not asking cubist38 to merge a workaround until they (and Apple) know the root cause is upstream of them.

---

## Ready-To-Go Content

### A1. `mlx-lm` issue body — DRAFT

**Title:** `Model forward passes leak loader-thread stream binding into lazy allocations`

**Body:**

```markdown
## Summary

When a model loaded via `mlx_lm.utils.load(...)` is first used from a thread other than the loader thread, certain lazy MLX allocations fail with `RuntimeError: There is no Stream(gpu, N) in current thread.`

The issue is that some deferred state (downstream of `set_wired_limit` and related initialization) captures a stream reference from whichever thread first triggers it. If that's the loader thread, later cross-thread evaluations fail.

## Reproduction

```python
# pip install mlx mlx-lm
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx
import threading

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit", lazy=False)

def worker():
    s = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(s):
        c = make_prompt_cache(model)
        ids = mx.array([tok.encode("hi")])
        mx.eval(model(ids, cache=c), [x.state for x in c])

t = threading.Thread(target=worker)
t.start()
t.join()  # -> RuntimeError: There is no Stream(gpu, 1) in current thread.
```

## Workaround (current)

Run a one-token forward + `mx.eval` on the loader thread before any consumer thread touches the model:

```python
model, tok = load("...", lazy=False)

# warm-up
c = make_prompt_cache(model)
mx.eval(model(mx.array([[tok.bos_token_id]]), cache=c), [x.state for x in c])

# now worker threads can use the model freely
```

## Impact

This affects any consumer that does inference on a thread other than the loader thread. The pattern is common in inference servers (e.g., `cubist38/mlx-openai-server`'s `BatchScheduler` runs on a dedicated worker thread). Without an out-of-the-box workaround, every threaded scheduler must include explicit warm-up logic.

## Suggested fixes

Two approaches:

1. **Thread-localize lazy allocation.** Defer stream binding to consumer-thread use rather than loader-thread first-touch.
2. **Eagerly resolve at `load(lazy=False)`.** When the caller explicitly opts into eager loading, no deferred state should remain post-`load()`.

Option 2 is simpler conceptually and matches the existing `lazy=False` semantics most users expect. Happy to discuss further or submit a PR if there's preferred direction.

## Environment

- mlx: <fill in version>
- mlx-lm: <fill in version>
- macOS: Apple Silicon (M-series), Sonoma/Sequoia
- Python: 3.12

## Related

Discovered while building integration tests for a fork of `cubist38/mlx-openai-server`. The same bug manifests in their `BatchScheduler` (chat completion path) and in our embedding-server path. Both paths require the warm-up workaround pending an upstream fix.
```

### B1. `cubist38/mlx-openai-server` issue body — DRAFT

**Title:** `Chat completions through BatchScheduler fail with "There is no Stream(gpu, N) in current thread"`

**Body:**

```markdown
## Summary

Every chat completion request through the `BatchScheduler` returns HTTP 500 with:

```
RuntimeError: There is no Stream(gpu, N) in current thread.
```

Bug exists at HEAD of `main` (verified at commit 4b7d4b6 of this repo). It's not caught by the existing test suite because there are no integration tests that boot the server and hit `/v1/chat/completions` end-to-end through the scheduler.

## Reproduction

1. Build/install a clean checkout of `cubist38/mlx-openai-server@4b7d4b6` (or current `main`).
2. Boot the server:
   ```bash
   python -m app.main launch --port 18099 --model mlx-community/Llama-3.2-1B-Instruct-4bit --host 127.0.0.1
   ```
3. Wait for `/healthz` to return 200.
4. Send a chat completion:
   ```bash
   curl -X POST http://127.0.0.1:18099/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"mlx-community/Llama-3.2-1B-Instruct-4bit","messages":[{"role":"user","content":"Say hello"}],"max_tokens":8}'
   ```
5. Returns HTTP 500 with `"There is no Stream(gpu, N) in current thread"` error.

Server-side traceback:

```
File ".../app/core/batch_scheduler.py", line 494, in _run
    prompt_responses, gen_responses = self._batch_generator.next()
File ".../mlx_lm/generate.py", line 1855, in next
    return self._next()
File ".../mlx_lm/generate.py", line 1841, in _next
    self._prompt_batch.prompt(prompts)
File ".../mlx_lm/generate.py", line 1161, in prompt
    mx.eval([c.state for c in self.prompt_cache])
RuntimeError: There is no Stream(gpu, 1) in current thread.
```

## Root cause

This is an mlx-lm issue: lazy MLX allocations triggered during the model's first forward pass capture a stream reference from whichever thread first uses the model. Since `BatchScheduler` runs on a dedicated worker thread that owns its own `mx.new_thread_local_stream(...)`, the cache state allocated under it can't be evaluated under a different stream context.

Filed upstream at [link to mlx-lm issue].

## Workaround

Warm up the model on the caller (loader) thread before spawning the scheduler thread. A one-token forward pass + `mx.eval` materializes every lazy allocation, leaving only static weights for the scheduler to read.

I have a working patch ready as a PR — happy to submit if you'd like the workaround merged ahead of the upstream mlx-lm fix.

## Embedding handler note

The same root-cause bug affects `app/handler/mlx_embeddings.py` (or wherever the embedding service spawns workers). My fork applied the same warm-up shape to both paths.

## Environment

- mlx-openai-server commit: 4b7d4b6
- mlx: <fill in>
- mlx-lm: <fill in>
- macOS Apple Silicon, Python 3.12
```

### B2. `cubist38/mlx-openai-server` PR description — DRAFT

**Title:** `Fix: warm up model on loader thread before scheduler spawns (unblocks chat under BatchScheduler)`

**Body:**

```markdown
## Summary

Fixes chat completions failing with `RuntimeError: There is no Stream(gpu, N) in current thread` when requests are routed through `BatchScheduler` (i.e., every chat request). See issue #<TODO: link the cubist38 issue>.

## Root cause

mlx-lm's model forward triggers lazy MLX allocations that bind to whichever thread first touches them. When that's the scheduler thread (rather than the loader thread), later cross-thread evaluations fail.

The root-cause fix belongs in mlx-lm; filed at ml-explore/mlx-lm#<TODO: link>. This PR is a server-side workaround that unblocks users today.

## What this changes

1. **`BatchScheduler._warm_up_model()`** (new): a one-token forward + `mx.eval` on the caller thread inside `start()`, before the scheduler thread is spawned. Materializes lazy MLX state under the loader thread so the scheduler thread only sees fully-resolved tensors.

2. **`MLXEmbeddingsHandler._warm_up()`** (new): same shape, applied to the embedding service. The same bug affects the embedding worker path; this fix unblocks it too.

3. **`BatchScheduler._run()` main loop wrap** (defensive): wraps the loop body in `with mx.stream(self._stream):`, matching the existing precedent at `_process_cancellations`. Belt-and-suspenders for any future MLX ops not already wrapped by `BatchGenerator`.

Warm-up exceptions are swallowed so they don't block startup; a real model error will surface cleanly on the first real request.

Total LOC: ~50 across two files. No API changes. No new dependencies. No behavior changes for already-working paths.

## Testing

Added an end-to-end integration test that boots the server and hits `/v1/chat/completions` with a real MLX model. Confirms the fix works against both streaming and non-streaming completions, and that 4 concurrent streaming requests preserve KV-cache isolation (no token cross-contamination).

Before this PR: integration tests fail with `RuntimeError: There is no Stream(gpu, N)`. After: all pass.

Unit test suite unchanged (14 pre-existing failures in `test_mixed_think_tool_handoff_*` and `test_batch_scheduler.py` — orthogonal mock-drift issues, not affected by this fix).

## Performance

Negligible startup cost: one forward pass on a single token. On M2 Pro with a 1B 4-bit model, warm-up takes <100ms.

## Diagnostic / minimal repro (no server)

```python
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx
import threading

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit", lazy=False)

def worker():
    s = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(s):
        c = make_prompt_cache(model)
        ids = mx.array([tok.encode("hi")])
        mx.eval(model(ids, cache=c), [x.state for x in c])

threading.Thread(target=worker).start()  # -> RuntimeError: There is no Stream(gpu, 1)
```

Adding a single `c = make_prompt_cache(model); mx.eval(...)` call on the main thread BEFORE the worker spawns eliminates the error. That's what this PR encodes inside `BatchScheduler.start()` and the embedding service init.
```

### C1. Sanitized commit message for cubist38 PR

Our internal commit `8547c67` references `cubist38/mlx-openai-server@4b7d4b6` (the pinned-fork SHA) and includes a `Co-Authored-By: Claude Opus 4.7` trailer. When upstreaming, sanitize:

**Sanitized:**

```
fix(scheduler): warm up model on caller thread before scheduler runs

Chat completions through the BatchScheduler 500 on every request with:

    RuntimeError: There is no Stream(gpu, N) in current thread.

Root cause is in mlx-lm: model forward triggers lazy MLX allocations
that bind to the loader thread's stream. The scheduler thread can't
later evaluate them. Filed upstream at ml-explore/mlx-lm#<TODO>.

Fix: run a one-token forward + mx.eval on the caller thread inside
start() before the scheduler thread is spawned. This materializes every
lazy allocation on the loader thread, leaving only static weights for
the scheduler thread to read. Skipped silently for unit-test mocks
without .layers.

Also wrap the scheduler's main loop body in mx.stream(self._stream) so
prompt-cache scaffolding allocated by _admit_pending / insert_segments /
_make_batch lands on the scheduler-thread-local stream. Mirrors the
existing precedent at _process_cancellations where remove(...) is
already wrapped.

Closes #<TODO: cubist38 issue number>
```

(Remove the `Co-Authored-By` trailer unless cubist38 commit conventions accept LLM co-authorship. Their commit log uses plain author lines; safest to drop.)

---

## Sanitization Checklist Before Submitting

When converting our internal commits to upstream PRs:

- [ ] Remove `cubist38/mlx-openai-server@4b7d4b6` SHA references (replace with `current main` or `HEAD`)
- [ ] Remove `Co-Authored-By: Claude` lines if upstream doesn't accept LLM co-authorship
- [ ] Remove any references to "Coldfire" or `getcoldfire/inference-server`
- [ ] Reframe "upstream bug" language — for the cubist38 PR, the bug IS upstream to cubist38 (it's in mlx-lm), so "upstream" means mlx-lm not us
- [ ] Replace internal repro paths (e.g., `/Users/mikewilliams/Source/...`) with generic placeholders
- [ ] Verify the test file paths in the PR match cubist38's layout if we're adding tests; if their CI structure differs, adapt
- [ ] Double-check the diagnostic script runs against a fresh `pip install mlx-lm mlx-community/Llama-3.2-1B-Instruct-4bit` install with no Coldfire artifacts

---

## Open Decisions (To Resolve Before Filing)

1. **Identity:** File as Mike Williams (personal) or under a `getcoldfire`/Coldfire org account? Personal is simpler and matches typical OSS contribution norms; org gives Coldfire visibility but invites Q&A about what Coldfire is doing with the fork.

2. **mlx-lm issue first vs. simultaneous?** Recommended sequence (A → B → C) implies waiting on mlx-lm acknowledgment before pinging cubist38. Alternative: file all three within a day so they cross-reference. Lower friction; small risk of looking like a contribution dump.

3. **Include benchmarks?** Our warm-up cost is <100ms on a 1B model. Worth including in the PR? Probably yes — preempts the "doesn't this slow startup?" concern.

4. **Submit the `mx.stream(self._stream):` wrap as part of the same PR or separately?** It's defense in depth. Same PR is simpler; separate PRs are cleaner (one fix per PR). Recommend same PR with a clear "two related changes" framing.

5. **Embeddings handler fix scope:** Our `EmbeddingService._warm_up()` is in our own permissive-licensed embeddings code. cubist38's embedding handler uses `mlx-embeddings` (GPLv3), which is part of why we rewrote it. The upstream PR would target cubist38's `mlx_embeddings.py` handler (the file we kept the name of, but rewrote internals). We need to apply the warm-up to cubist38's version, not ours. **Action item before submission:** test that the warm-up shape works against cubist38's mlx-embeddings-based code — likely identical but worth a verification step.

6. **Tests included in the PR?** Our integration tests live in `tests/integration/`. cubist38 doesn't have an `integration/` directory. We can either (a) add the directory structure as part of the PR (good — gives them coverage), or (b) only ship the fix and let them add tests separately. Recommend (a) — the test is what catches the regression in the future.

7. **License of test code:** Anything we ship to cubist38 must be MIT-compatible. Our tests are written by us under MIT terms; no concern.

---

## Useful Links

### Our fork's commits
- `8547c67` — BatchScheduler warm-up + main loop stream wrap: `git show 8547c67` in `getcoldfire/inference-server`
- `c90e01d` — EmbeddingService warm-up + nomic-bert remap (only the warm-up portion goes upstream; the nomic remap is Coldfire-specific because cubist38 doesn't need it after the GPLv3 dep removal)

### Reference reading
- mlx-lm source: `mlx_lm/generate.py:1161` (the `mx.eval([c.state for c in self.prompt_cache])` site that fails)
- mlx-lm `BatchGenerator.next()`: `mlx_lm/generate.py:1855` (where the forward pass IS correctly stream-wrapped)
- cubist38 `BatchScheduler._run()`: `app/core/batch_scheduler.py:494` (where the error surfaces)
- cubist38 `BatchScheduler._process_cancellations()`: existing `mx.stream(self._stream):` precedent for the main loop wrap

### Coldfire context (do NOT include in upstream PRs)
- Fork spec: `docs/superpowers/specs/2026-06-03-mlx-openai-server-fork-spec.md` in Coldfire monorepo
- Fork implementation plan: `docs/superpowers/plans/2026-06-03-mlx-openai-server-fork.md` in Coldfire monorepo
- Diagnostic transcript: Phase 6 of the implementation plan

---

## Status & Next Steps

**This document is a content cache. No PRs have been filed.**

When ready to file, the recommended sequence is:

1. Re-verify the minimal repro still works against a fresh `pip install mlx mlx-lm` (in case Apple has shipped a fix in the meantime). 10 minutes.
2. File the mlx-lm issue (Path A). Wait for acknowledgment / triage label.
3. File the cubist38 issue (Path B) referencing the mlx-lm issue.
4. Submit the cubist38 PR (Path C) referencing both issues. Include sanitized commit, both warm-ups, the main loop wrap, and integration tests.
5. Respond to review feedback from cubist38 maintainers.

Estimated effort: ~2 hours total once we start, assuming no major surprises. If cubist38 requests substantial test infrastructure additions, could grow to ~half a day.

---

## Revised Status & Next Steps (supersedes original Status & Next Steps section)

Added 2026-06-05 after the prior-art search. Replaces the original five-step sequence above; if these conflict, follow this section.

**Do not file new issues at `ml-explore/mlx` or `ml-explore/mlx-lm`. Do not file a fresh issue at `cubist38/mlx-openai-server` either — there is already a closed one (#290).** The single net-new contribution is the cubist38 PR.

Recommended sequence:

1. **Re-verify repro against current versions.** 10 minutes. `pip install -U mlx mlx-lm`, re-run the minimal repro from this doc, confirm the error still surfaces. Also check whether mlx-lm PR #1275 has merged in the meantime — if it has, re-run our cubist38 `BatchScheduler` repro with the updated mlx-lm to confirm warm-up is *still* needed (very likely yes, because the lazy model state path is separate from the `generation_stream` path #1275 fixes).

2. **Comment on `mlx-lm#1256` and PR `mlx-lm#1275`.** ~30 minutes.
   - On #1256: post our minimal repro showing the failure surfaces in cubist38's `BatchScheduler` even though it uses its own `new_thread_local_stream` per worker — i.e., the bug isn't purely about module-level `generation_stream`. Frame as supporting evidence that the symptom has multiple root paths.
   - On #1275: ask whether the per-thread `generation_stream` fix covers the case where the scheduler thread owns its own stream entirely. If reviewers want a second test case, offer ours.
   - Tone: contributing to existing work, not announcing competing work.

3. **Submit the cubist38 PR (the only net-new code).** ~1 hour.
   - **Reframe the description**: this PR completes the work begun in #295. Credit #295. Cite the May 14 #290 comment showing #295 was insufficient. Explain why per-thread streams alone don't cover the lazy-state path; explain why warm-up does.
   - Reference `mlx#3529` showing this is the maintainer-blessed pattern (`mx.eval(model.parameters())` before crossing threads — our warm-up is a stronger form of this).
   - Reference `mlx-lm#1256` and `#1275` as the related root-cause work upstream of cubist38.
   - Ship: `BatchScheduler._warm_up_model()`, `EmbeddingService._warm_up()` (test against cubist38's mlx-embeddings code first — open question #5 in the original Open Decisions section above), the defensive `_run` main-loop `mx.stream(self._stream):` wrap, and the integration tests.
   - The PR body / commit message in the original "Ready-To-Go Content" section above is still mostly reusable; rewrite the framing only.

4. **Do NOT reopen `cubist38#290`.** Let the PR speak for itself; maintainers can reopen #290 themselves if they choose. Avoid contradicting their close decision in writing — let the PR be the followup.

5. **Skip filing at mlx-lm and mlx entirely.** Sections "A1. mlx-lm issue body — DRAFT" and the corresponding submission step in the original plan are obsolete. The Path-A content remains in this document only as background for the cubist38 PR description and as reusable language if mlx-lm reviewers ask for elaboration.

Estimated effort post-revision: ~1.5 hours total. Down from the original ~2h because we're cutting two filings.

**Open decisions still pending from the original plan that this revision does not resolve:**
- Identity (personal vs. org account) — original Open Decisions #1
- Whether to ship tests in the same PR or separately — original Open Decisions #6
- Verifying warm-up works against cubist38's `mlx-embeddings`-based embedding handler — original Open Decisions #5

Resolve those before filing.
