"""BM25 keyword index over chunk texts (rank-bm25), persisted as pickle.

The pickle stores the tokenized corpus + chunk ids + a corpus hash. The M3
pipeline rebuilds it whenever the chunk set changes; query-time metadata
filtering masks scores by allowed ids (computed from the chunks table).
"""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    name = "bm25"

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self.corpus_hash: str = ""

    @staticmethod
    def corpus_hash_for(ids: list[str]) -> str:
        return hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]

    def build(self, chunk_ids: list[str], texts: list[str]) -> "BM25Index":
        self.chunk_ids = list(chunk_ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.corpus_hash = self.corpus_hash_for(self.chunk_ids)
        return self

    def search(
        self, query: str, k: int = 8, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda t: t[1], reverse=True)
        out = [(cid, float(s)) for cid, s in ranked if s > 0]
        if allowed_ids is not None:
            out = [(cid, s) for cid, s in out if cid in allowed_ids]
        return out[:k]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "bm25": self._bm25,
                 "corpus_hash": self.corpus_hash}, f,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        idx = cls()
        with path.open("rb") as f:
            data = pickle.load(f)
        idx.chunk_ids = data["chunk_ids"]
        idx._bm25 = data["bm25"]
        idx.corpus_hash = data.get("corpus_hash", "")
        return idx
