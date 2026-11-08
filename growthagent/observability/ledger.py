"""Token / cost ledger: accumulate usage across runs and report totals.

The cost numbers use the per-1k-token prices from settings; if they are 0
(no pricing configured) the ledger still reports token counts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..schema import Usage, utc_iso


class TokenLedger:
    def __init__(self, path: Optional[str] = None, prices: Optional[Dict[str, float]] = None):
        self.path = path
        self.prices = prices or {}
        self._entries: List[Dict[str, Any]] = []
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def record(self, model: str, usage: Usage, latency_ms: float = 0.0) -> None:
        entry = {
            "ts": utc_iso(),
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_ms": round(latency_ms, 1),
        }
        self._entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _cost(self, entry: Dict[str, Any]) -> float:
        pp = self.prices.get("prompt_per_1k", 0.0)
        pc = self.prices.get("completion_per_1k", 0.0)
        return entry["prompt_tokens"] / 1000.0 * pp + entry["completion_tokens"] / 1000.0 * pc

    def report(self) -> Dict[str, Any]:
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "cost": 0.0,
            "latency_ms": 0.0,
        }
        by_model: Dict[str, Dict[str, Any]] = {}
        for e in self._entries:
            total["prompt_tokens"] += e["prompt_tokens"]
            total["completion_tokens"] += e["completion_tokens"]
            total["total_tokens"] += e["total_tokens"]
            total["calls"] += 1
            total["cost"] += self._cost(e)
            total["latency_ms"] += e["latency_ms"]
            m = by_model.setdefault(
                e["model"],
                {"calls": 0, "total_tokens": 0, "cost": 0.0},
            )
            m["calls"] += 1
            m["total_tokens"] += e["total_tokens"]
            m["cost"] += self._cost(e)
        total["cost"] = round(total["cost"], 6)
        total["latency_ms"] = round(total["latency_ms"], 1)
        for m in by_model.values():
            m["cost"] = round(m["cost"], 6)
        return {"totals": total, "by_model": by_model}


def ledger_from_settings(settings: Optional[Settings] = None) -> TokenLedger:
    settings = settings or get_settings()
    return TokenLedger(
        path=settings.ledger_path,
        prices={
            "prompt_per_1k": settings.price_prompt_per_1k,
            "completion_per_1k": settings.price_completion_per_1k,
        },
    )
