"""License-clean embedding code path.

Replaces the GPLv3 `mlx-embeddings` dependency with a permissively-licensed
BERT-family encoder. Supports two architectural variants:

- Vanilla BERT: absolute position embeddings + GeLU MLP + token type ids
  (e.g. `bert-base-uncased`, `all-MiniLM-L6-v2`).
- Nomic-bert: rotary position embeddings + SwiGLU MLP, no token type ids
  (e.g. `nomic-ai/nomic-embed-text-v1.5`).
"""
