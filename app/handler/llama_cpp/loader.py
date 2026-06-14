"""Lazy loader for llama-cpp-python embedding models.

Construction is cheap (config only); the underlying Llama() instance is
built on first embed() call so admin-POST registration stays fast.
Mirrors the on-demand contract of the MLX-LM handler.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

from llama_cpp import Llama

log = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig:
    """Per-model knob set surfaced via admin POST / static YAML config.

    All fields except model_path are optional; llama-cpp-python applies
    its own defaults for omitted values.
    """
    model_path: str            # HF repo id ("org/repo") OR absolute local path ending in .gguf
    hf_file: Optional[str]     # required if model_path is an HF repo; ignored for local paths
    n_gpu_layers: int = -1     # -1 = all to Metal (default on Apple Silicon)
    n_ctx: Optional[int] = None
    n_batch: Optional[int] = None
    n_threads: Optional[int] = None


class LlamaCppEmbeddingsLoader:
    """Lazy loader. .llama is None until first ensure_loaded() call.

    Thread-safety: ensure_loaded() is idempotent and guarded by a lock so
    concurrent first-requests don't race two Llama() constructions.
    """

    def __init__(self, cfg: LlamaCppConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self.llama: Optional[Llama] = None

    @staticmethod
    def _is_local_path(model_path: str) -> bool:
        return model_path.startswith("/")

    def ensure_loaded(self) -> Llama:
        if self.llama is not None:
            return self.llama
        with self._lock:
            if self.llama is not None:  # double-check after lock
                return self.llama
            kwargs: dict = {
                "embedding": True,
                "n_gpu_layers": self.cfg.n_gpu_layers,
            }
            if self.cfg.n_ctx is not None:
                kwargs["n_ctx"] = self.cfg.n_ctx
            if self.cfg.n_batch is not None:
                kwargs["n_batch"] = self.cfg.n_batch
            if self.cfg.n_threads is not None:
                kwargs["n_threads"] = self.cfg.n_threads

            if self._is_local_path(self.cfg.model_path):
                log.info("llama_cpp: loading local file: %s", self.cfg.model_path)
                if not os.path.exists(self.cfg.model_path):
                    raise FileNotFoundError(
                        f"llama-cpp model_path is local but does not exist: {self.cfg.model_path}"
                    )
                self.llama = Llama(model_path=self.cfg.model_path, **kwargs)
            else:
                if not self.cfg.hf_file:
                    raise ValueError(
                        f"llama-cpp: hf_file is required for HF repo "
                        f"{self.cfg.model_path!r} (no local path detected)"
                    )
                log.info(
                    "llama_cpp: fetching from HF: %s/%s",
                    self.cfg.model_path, self.cfg.hf_file,
                )
                self.llama = Llama.from_pretrained(
                    repo_id=self.cfg.model_path,
                    filename=self.cfg.hf_file,
                    **kwargs,
                )
            log.info("llama_cpp: loaded %s", self.cfg.model_path)
            return self.llama
