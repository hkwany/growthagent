"""Mock URL fetcher.

Network-free: serves a tiny in-memory set of "pages" keyed by URL.  The
real fetch would use urllib/requests — this keeps tests and the demo
deterministic.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import tool

_PAGES: Dict[str, Dict[str, str]] = {
    "https://example.com/product/focusapp": {
        "title": "FocusApp — deep-work productivity timer",
        "content": "FocusApp helps knowledge workers block distractions with a minimal timer, session analytics and weekly focus reports. Target audience: developers and students. Key benefit: 2x focused hours.",
    },
    "https://example.com/landing/focusapp": {
        "title": "FocusApp landing page",
        "content": "Hero: 'Get 2x more deep work'. CTA: 'Start free trial'. Social proof: 'Loved by 40k developers'.",
    },
}


@tool(
    name="fetch_url",
    description="Fetch the title and text content of a web page (mock, offline).",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to fetch, e.g. 'https://example.com/product/focusapp'."}
        },
        "required": ["url"],
    },
)
def fetch_url(url: str) -> Dict[str, Any]:
    page = _PAGES.get(url)
    if page is None:
        return {"url": url, "error": "page not found in offline index"}
    return {"url": url, **page}
