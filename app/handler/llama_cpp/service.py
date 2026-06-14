"""Service layer for the llama-cpp embedding handler.

Takes raw texts, calls Llama.create_embedding(), passes the vector through
the shared matryoshka helper, returns the OpenAI-shaped response data.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.handler.embeddings_common import apply_dimensions
from app.handler.llama_cpp.loader import LlamaCppEmbeddingsLoader

log = logging.getLogger(__name__)


class LlamaCppEmbeddingsService:
    def __init__(self, loader: LlamaCppEmbeddingsLoader):
        self.loader = loader

    def embed(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Return one embedding vector per input text.

        If `dimensions` is set, the vector is truncated + L2-renormalized via
        the shared helper. None means "return full-dim raw vector" (still L2-
        normalized by llama-cpp internally when embedding=True).
        """
        llama = self.loader.ensure_loaded()
        out: List[List[float]] = []
        for text in texts:
            response = llama.create_embedding(input=text)
            # llama-cpp returns: {"object": "list", "data": [{"embedding": [...]}], ...}
            raw_vec = np.array(response["data"][0]["embedding"], dtype=np.float32)
            if dimensions is not None and dimensions != len(raw_vec):
                vec = apply_dimensions(raw_vec, dimensions)
            else:
                vec = raw_vec
            out.append(vec.tolist())
        return out
