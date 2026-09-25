"""Embedding providers behind the EmbeddingProvider protocol.

- SentenceTransformerProvider: production path (all-MiniLM-L6-v2, 384 dims,
  row-normalized so cosine == dot product). Model downloads once from HF.
- HashEmbeddingProvider: dependency-free deterministic vectors for unit tests
  and offline CI. NOT for quality evaluation.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


class SentenceTransformerProvider:
    name = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import time

            from sentence_transformers import SentenceTransformer

            last = None
            for attempt in range(3):  # transient HF rate limits fail open runs
                try:
                    self._model = SentenceTransformer(self.model_name)
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    time.sleep(2.0 * (attempt + 1))
            if self._model is None:
                raise RuntimeError(f"could not load {self.model_name}: {last}")
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                            show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbeddingProvider:
    """Deterministic token-hash vectors. Tests only — no semantic quality."""

    def __init__(self, dim: int = 64) -> None:
        self.name = "hash-test"
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out
