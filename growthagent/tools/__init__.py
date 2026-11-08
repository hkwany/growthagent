"""Tool package: registry + the five built-in tools."""

from __future__ import annotations

from .base import Tool, ToolRegistry, tool
from .ad_platform import create_campaign, get_campaign_status, reset as _reset_ad
from .analytics import get_metrics, reset as _reset_analytics
from .calculator import calculator, safe_eval
from .fetch_url import fetch_url
from .search import search


def default_registry() -> ToolRegistry:
    """A registry with all built-in tools registered."""
    return ToolRegistry(
        tools=[
            search,
            fetch_url,
            calculator,
            create_campaign,
            get_campaign_status,
            get_metrics,
        ]
    )


def reset_state() -> None:
    """Reset all stateful mock tools (ad platform + analytics)."""
    _reset_ad()
    _reset_analytics()


__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "default_registry",
    "reset_state",
    "search",
    "fetch_url",
    "calculator",
    "safe_eval",
    "create_campaign",
    "get_campaign_status",
    "get_metrics",
]
