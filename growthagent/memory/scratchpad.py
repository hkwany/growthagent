"""Short-term scratchpad: a tiny namespaced key/value store.

Models the "working memory" an agent uses within a single run.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class Scratchpad:
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self) -> List[str]:
        return sorted(self._data.keys())

    def clear(self) -> None:
        self._data.clear()

    def to_json(self) -> str:
        return json.dumps(self._data, ensure_ascii=False, default=str)

    def from_json(self, s: str) -> None:
        self._data = json.loads(s)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Scratchpad(keys={self.keys()})"
