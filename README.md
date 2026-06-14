# coldfire-mlx-server

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

License-clean MLX-LM inference server with an OpenAI-compatible API. Fork of [`cubist38/mlx-openai-server`](https://github.com/cubist38/mlx-openai-server), MIT-licensed end-to-end.

> Apple Silicon only. macOS 13 Ventura or newer. Python 3.11+.

---

## Runtimes

`coldfire-mlx-server` hosts multiple inference runtimes in one Python process:

| Runtime | `model_type` | Use case |
|---|---|---|
| MLX-LM | `lm` | Chat completion on Apple Silicon |
| BERT encoder | `embeddings` | Vector embeddings (nomic-1.5, BGE, mxbai variants) |
| llama-cpp | `llama-cpp` | GGUF embeddings — wraps `llama-cpp-python` with Metal acceleration |

The package name (`coldfire-mlx-server`) is historical from when MLX was
the only runtime; a runtime-neutral rename is tracked as a future change.

## Install

The supported install path is the Coldfire Homebrew tap:

```bash
brew tap getcoldfire/coldfire
brew install coldfire-mlx-server
```

That installs the `coldfire-mlx-server` CLI on your `PATH` and bundles the third-party `NOTICES.txt` under `share/doc/coldfire-mlx-server/`.

Source installs from this repo are supported for development; see `AGENTS.md` for the dev environment setup.

## Quickstart

```bash
coldfire-mlx-server launch \
  --host 127.0.0.1 \
  --port 8080 \
  --model-path mlx-community/Llama-3.2-1B-Instruct-4bit
```

That starts an OpenAI-compatible HTTP server bound to loopback on port 8080. `--model-path` accepts a HuggingFace repo ID or a local directory.

Then from any OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-used")
resp = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello."}],
)
print(resp.choices[0].message.content)
```

## API endpoints

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible. Streaming (`stream: true`) and non-streaming. |
| `GET` | `/v1/models` | Lists the served model(s). |
| `POST` | `/v1/embeddings` | OpenAI-compatible. Requires `--model-type embeddings` at launch. |
| `GET` | `/healthz` | `200` once the model is loaded and ready; `503` during load. |

Audio (transcription/TTS), image generation, and Vision-Language Model endpoints from upstream have been removed — `coldfire-mlx-server` serves chat + embeddings only.

## Supported embedding models

The fork ships its own BERT-family encoder (configurable position-embedding, activation, and pooling) in place of the GPLv3 `mlx-embeddings` dependency. Coverage is grouped into three tiers:

**Validated** (correctness-tested against a Sentence-Transformers reference, cosine-sim ≥ 0.999 on a 100-doc corpus):
- `nomic-ai/nomic-embed-text-v1.5` (also `mlx-community/nomic-embed-text-v1.5`)
- `mxbai-embed-large-v1` (`mlx-community/mxbai-embed-large-v1` or `mixedbread-ai/mxbai-embed-large-v1`)

**Should work** (BERT-family configurations the encoder supports — not regression-tested per-release):
- `bge-*` (`BAAI/bge-large-en-v1.5`, `BAAI/bge-base-en-v1.5`, etc.)
- `gte-*` (`thenlper/gte-large`, `thenlper/gte-base`, etc.)
- `e5-*` (`intfloat/e5-large-v2`, etc.)
- `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`
- `arctic-embed-*` (`Snowflake/snowflake-arctic-embed-l`, etc.)

**Unsupported** (the loader will refuse to start with a loud error message — better than silently producing garbage embeddings):
- ModernBERT-based embeddings (`modernbert-embed-*`) — v0.2 follow-up
- Mamba / SSM-based architectures
- MoE-based embedding models

## License & NOTICES

This package is MIT-licensed; see `LICENSE`. The dependency tree is audited at every release — the CI license gate fails the build if any transitive runtime dependency picks up a GPL/AGPL/SSPL license.

To see all bundled third-party license attributions:

```bash
coldfire-mlx-server --licenses
```

That prints the full `NOTICES.txt`. With a Homebrew install, the file also lives at `$(brew --prefix)/share/doc/coldfire-mlx-server/NOTICES.txt`.

## Upstream relationship

This is a fork of [`cubist38/mlx-openai-server`](https://github.com/cubist38/mlx-openai-server), maintained under the Coldfire project. Compared to upstream, this fork:

- **Removes the GPLv3 dependency.** Upstream depends on [`Blaizzy/mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) for its `/v1/embeddings` endpoint, which is GPLv3-licensed. We replaced it with our own permissive BERT-family encoder (`app/handler/embeddings/`) — vanilla, RoPE, and SwiGLU variants with mean / CLS / last-token / matryoshka pooling.
- **Strips features Coldfire doesn't need.** Audio, image generation, and Vision-Language Model endpoints are gone. This keeps the dependency tree small and license-clean.
- **Patches an MLX stream-affinity bug.** The MLX runtime lazy-binds model state to whichever thread first triggers evaluation, which causes `RuntimeError: There is no Stream(gpu, 1) in current thread` when worker threads serve requests. We added a forward-pass warm-up on the caller thread in `MLXLMHandler.initialize()` and wrap the batch scheduler main loop in an explicit stream context. Upstream context: [`mlx-lm#1256`](https://github.com/ml-explore/mlx-lm/issues/1256), [`mlx-lm#1275`](https://github.com/ml-explore/mlx-lm/pull/1275), [`mlx#3529`](https://github.com/ml-explore/mlx/issues/3529). Fork-side detail: `docs/UPSTREAM_PR_PLAN.md`.
- **Audit-gated CI.** Every PR runs `tools/license_check.py` against the locked dependency set; the release workflow re-runs the audit before publishing artifacts.

The historical upstream relationship is documented in `UPSTREAM.md` and the top-level `NOTICE`.

## Differences from upstream at a glance

| Aspect | Upstream (`cubist38/mlx-openai-server`) | Coldfire fork |
|--------|------------------------------------------|---------------|
| Endpoints | chat, embeddings, audio (Whisper), image-gen, VLM | chat, embeddings, models, healthz |
| Embeddings library | `mlx-embeddings` (GPLv3) | In-tree BERT-family encoder (MIT) |
| License surface | Mixed (transitive GPL) | MIT end-to-end; CI-gated |
| Stream-affinity fix | Not applied | Applied (warm-up + stream wrap) |
| Console script | `mlx-openai-server` | `coldfire-mlx-server` |
| Distribution | PyPI | Homebrew tap (`getcoldfire/coldfire`) |
