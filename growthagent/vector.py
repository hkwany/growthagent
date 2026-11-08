"""Vector utilities: a dependency-light embedder and a cosine vector store.

``HashEmbedder`` is a deterministic, API-free embedding used so the demo
runs without any model serving.  It is intentionally crude — in a real
deployment you'd swap in ``sentence-transformers`` or an embedding API
behind the same ``embed(text) -> np.ndarray`` interface.

``VectorStore`` is a brute-force cosine store over numpy.  ``hnswlib`` is
used if installed, but the fallback is fine for demo-scale corpora.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # optional acceleration
    import hnswlib  # type: ignore

    _HAS_HNSW = True
except Exception:  # pragma: no cover
    _HAS_HNSW = False

_WORD_RE = re.compile(r"\w+")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class HashEmbedder:
    """Deterministic hashing embedder (char 3-grams + word tokens)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _WORD_RE.findall(text.lower())
        tokens += [text[i : i + 3].lower() for i in range(max(0, len(text) - 2))]
        for tok in tokens:
            if not tok:
                continue
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return _normalize(vec)


class VectorStore:
    """Cosine-similarity vector store with optional hnswlib acceleration."""

    def __init__(self, dim: int, max_elements: int = 10_000):
        self.dim = dim
        self._ids: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._hnsw = None
        if _HAS_HNSW:
            try:
                self._hnsw = hnswlib.Index(space="cosine", dim=dim)
                self._hnsw.init_index(max_elements=max_elements, ef_construction=200, M=16)
            except Exception:  # pragma: no cover
                self._hnsw = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, id_: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        v = _normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        if v.shape[0] != self.dim:
            raise ValueError(f"vector dim {v.shape[0]} != store dim {self.dim}")
        self._ids.append(id_)
        self._meta.append(metadata or {})
        if self._vectors.shape[0] == 0:
            self._vectors = v.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, v.reshape(1, -1)])
        if self._hnsw is not None:
            try:
                self._hnsw.add_items(v.reshape(1, -1), [len(self._ids) - 1])
            except Exception:  # pragma: no cover
                pass

    def search(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return top-k ``(id, score, metadata)`` by cosine similarity."""
        if not self._ids:
            return []
        k = min(k, len(self._ids))
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        if self._hnsw is not None:
            try:
                labels, dists = self._hnsw.knn_query(q.reshape(1, -1), k=k)
                out = []
                for label, dist in zip(labels[0], dists[0]):
                    i = int(label)
                    out.append((self._ids[i], float(1.0 - dist), self._meta[i]))
                return out
            except Exception:  # pragma: no cover
                pass
        scores = self._vectors @ q
        order = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i]), self._meta[i]) for i in order]
