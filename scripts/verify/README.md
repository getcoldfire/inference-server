# coldfire-mlx-server — verification suite

Run these to verify your install works correctly. They're pure bash + curl
+ python3 (for JSON parsing) — no `jq`, no venv, no Python deps. Compatible
with bash 3.2 (macOS default).

---

## Quickstart

Launch the server in one terminal:

```bash
coldfire-mlx-server launch \
  --host 127.0.0.1 \
  --port 8080 \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit
```

In another terminal, run the menu:

```bash
make verify
# or:
./scripts/verify/run.sh
```

You'll see a menu of 6 tests. Pick one with a number, run all with `A`.

---

## Non-interactive examples

```bash
# Run every applicable test, exit non-zero if any fail
./scripts/verify/run.sh --all

# Run just test 4 (the KV-isolation regression check)
./scripts/verify/run.sh --test 4

# Talk to a server on a different host/port
./scripts/verify/run.sh --host 192.168.1.10 --port 9000 --all
```

Each test script is also runnable on its own:

```bash
./scripts/verify/4-concurrent-kv-isolation.sh
```

---

## Environment variable overrides

| Variable | Default | Meaning |
|---|---|---|
| `COLDFIRE_HOST` | `127.0.0.1` | Server host |
| `COLDFIRE_PORT` | `8080` | Server port |
| `NO_COLOR` | (unset) | If set, disables colored output |

`--host` / `--port` CLI flags override env vars.

---

## What each test does

### 1. Health + models (`~5s`)

`GET /healthz` returns `{"status":"ok"}`. `GET /v1/models` returns at least
one model. Prints the loaded model id and detected capability (`chat` vs
`embeddings`).

### 2. Basic chat completion (`~10s`)

Single non-streaming chat request, `max_tokens=8`. Asserts HTTP 200 and
non-empty content. Doesn't grade output quality (small quantized models
drift on the exact wording).

### 3. Streaming chat completion (`~10s`)

Streaming request (`stream: true`), `max_tokens=20`. Asserts at least 3
`data: {...}` chunks parse as JSON and the stream terminates with
`data: [DONE]`.

### 4. Concurrent KV-isolation (`~30s`) — **the bug test**

Fires 4 chat completions in parallel, each with a unique marker
(`QPRTX42`, `MNBVC77`, `JKHGF13`, `ZXCVB99` — synthetic IDs designed so no
two share any common substring of length ≥ 2, and so they don't look like
names or credentials that would trip small-model RLHF safety filters).
Prints a 4×4 grid showing marker presence per response.

Three-state outcome:

| Outcome | Meaning | Exit |
|---|---|---|
| **PASS** (green) | All 4 own markers echoed, zero foreign markers anywhere. | 0 |
| **WARN** (yellow) | Zero foreign markers, but one or more responses failed to echo their own marker. Model refusal / sampling drift — **not** a KV bug. Re-run; if WARN persists, switch to a larger model. | 0 |
| **HARD FAIL** (red) | Any foreign marker present in any response. The upstream MLX stream-affinity / KV-cache cross-contamination bug — file an issue. | 1 |

This is the test that would have caught the upstream bug.

**Reliability note:** Test 4's PASS/WARN ratio is tied to model quirks. Small
quantized models (Llama-3.2-1B-Instruct-4bit in particular) will WARN
occasionally due to sampling variance or stray safety refusals. That is
**not** a regression. Only a HARD FAIL indicates the KV-isolation guarantee
is broken.

### 5. Clean shutdown (`<10s`)

Finds the server PID, sends SIGTERM, waits up to 10 seconds for exit.
PASS if exit is <5s; WARN (still passes) if 5-10s; FAIL if >10s.
After this test the server is gone — relaunch it manually.

### 6. Embeddings (`~5s`)

Skips on chat-only servers. Sends a 2-document `/v1/embeddings` request.
Asserts 2 vectors returned, dim > 0, and each L2 norm ≈ 1.0 ± 0.001 (most
embedding models including nomic-embed-text return normalized vectors).

---

## Troubleshooting

### "No server detected at http://127.0.0.1:8080"

Launch one in another terminal:

```bash
coldfire-mlx-server launch \
  --host 127.0.0.1 \
  --port 8080 \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit
```

First launch can take 30+ seconds while the model downloads from
HuggingFace. Subsequent launches are fast.

### Test 4 HARD FAIL — "KV-cache cross-contamination detected"

A foreign marker appeared in a response. This is a KV-cache
cross-contamination regression — the bug the warm-up fix is supposed to
prevent. Please file an issue at
<https://github.com/getcoldfire/mlx-openai-server/issues> with the response
bodies the test prints.

### Test 4 WARN — "N of 4 responses did not echo their own marker"

Softer signal — KV isolation is **correct** (zero foreign-marker leaks),
but the model didn't echo one or more of its own markers back. Typical
causes: RLHF safety refusal, sampling variance, queue saturation, or
quantization quirks. **Not a regression.** Re-run the test once or twice;
if it keeps WARNing, switch to a larger model
(`mlx-community/Llama-3.2-3B-Instruct-4bit`).

### Test 5 "shutdown test hangs" or times out at 10s

cli-v2 contract violation — `coldfire-mlx-server launch` should exit
within 5 seconds of SIGTERM. File an issue with `ps aux | grep
coldfire-mlx-server` output.

### Test 5 skips with "ambiguous — multiple PIDs"

You have multiple `coldfire-mlx-server` instances running. Stop the ones
you don't care about and re-run.

### Test 6 skips with capability mismatch

Test 6 requires an embedding model. Chat servers don't expose embeddings.
Launch a separate embedding server on a different port and re-run with
`COLDFIRE_PORT=8081 ./scripts/verify/run.sh --test 6`.

---

## Note on the KV-isolation test

Test #4 is what would have caught the upstream MLX stream-affinity bug
where concurrent requests shared a KV cache and one stream's tokens
leaked into another's output. The fix is server-side (per-request stream
affinity + warm-up), but only this kind of end-to-end check can confirm
the fix actually holds in the shipped binary.
