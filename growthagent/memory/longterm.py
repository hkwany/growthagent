"""Long-term memory: cross-run recall backed by a vector store.

Lets the agent remember what it learned in earlier runs (e.g. "last
campaign for X used creative Y and got ROAS Z") and retrieve it by
semantic similarity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..vector import HashEmbedder, VectorStore


class LongTermMemory:
    def __init__(self, embedder: Optional[HashEmbedder] = None, dim: int = 256):
        self.embedder = embedder or HashEmbedder(dim)
        self.store = VectorStore(dim)

    def remember(self, text: str, id_: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store ``text`` and return the id used."""
        mid = id_ or f"mem_{len(self.store)}"
        vec = self.embedder.embed(text)
        meta = dict(metadata or {})
        meta.setdefault("text", text)
        self.store.add(mid, vec, meta)
        return mid

    def recall(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        vec = self.embedder.embed(query)
        return self.store.search(vec, k=k)

    def __len__(self) -> int:
        return len(self.store)
