"""Retrieval-augmented generation: index documents, retrieve by similarity.

Reuses the hash embedder + vector store so it works offline.  Drop in a
real embedder for production quality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..vector import HashEmbedder, VectorStore


class Retriever:
    def __init__(self, embedder: Optional[HashEmbedder] = None, dim: int = 256):
        self.embedder = embedder or HashEmbedder(dim)
        self.store = VectorStore(dim)

    def index(self, documents: List[Dict[str, Any]]) -> int:
        """Index documents of shape ``{"id": str, "text": str, ...}``.

        Returns the number of documents indexed.
        """
        n = 0
        for d in documents:
            doc_id = d.get("id") or d.get("text", "")[:32]
            self.store.add(doc_id, self.embedder.embed(d["text"]), d)
            n += 1
        return n

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return top-k ``(id, score, document)`` for a query."""
        return self.store.search(self.embedder.embed(query), k=k)

    def __len__(self) -> int:
        return len(self.store)
