"""llama-cpp-python handler — embeddings only at v1.

Sibling to app/handler/lm (MLX-LM chat) and app/handler/embeddings
(BERT-family encoder). Dispatched when model_type == "llama-cpp".

Spec: docs/superpowers/specs/2026-06-14-mlx-server-gguf-extension-design.md
"""
