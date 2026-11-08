"""Mock web-search tool.

Deterministic and network-free: searches a small in-memory index of
marketing / growth topics so the demo stays reproducible without any API.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import tool

_INDEX: Dict[str, List[Dict[str, str]]] = {
    "productivity app ads": [
        {"title": "Top-performing productivity app ad creatives", "snippet": "UGC-style demos and benefit-led headlines outperform feature lists by ~40%."},
        {"title": "App install campaigns: creative best practices", "snippet": "Short 15s vertical video with a clear CTA in the first 3 seconds."},
    ],
    "budget optimization": [
        {"title": "How to allocate ad budget across campaigns", "snippet": "Shift budget toward campaigns with the highest recent ROAS; re-evaluate weekly."},
        {"title": "Multi-armed bandit for ad spend", "snippet": "Epsilon-greedy allocation beats equal split when conversion variance is high."},
    ],
    "ad creative testing": [
        {"title": "A/B testing ad creative at scale", "snippet": "Test one variable at a time; need ~50 conversions per variant for significance."},
    ],
}


@tool(
    name="search",
    description="Search a knowledge base of growth/marketing topics and return relevant results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query, e.g. 'budget optimization'."}
        },
        "required": ["query"],
    },
)
def search(query: str) -> Dict[str, Any]:
    q = query.lower().strip()
    results: List[Dict[str, str]] = []
    # exact-ish match first, then keyword overlap
    if q in _INDEX:
        results = _INDEX[q]
    else:
        for key, hits in _INDEX.items():
            if any(tok in key or key in tok for tok in q.split()):
                results.extend(hits)
    return {"query": query, "results": results or [{"title": "No results", "snippet": "Try a broader query."}]}
