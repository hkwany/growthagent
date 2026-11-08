"""Golden evaluation tasks.

A golden task pins down *what the agent should do* (expected tool calls)
and *what a good final answer looks like* (substrings that must appear).
These are the source of truth for the eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GoldenTask:
    id: str
    task: str
    expected_tool_calls: List[Tuple[str, Dict]] = field(default_factory=list)
    #: substrings the final answer must contain for the task to count as a success
    expected_final_contains: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


GOLDEN_TASKS: List[GoldenTask] = [
    GoldenTask(
        id="calc",
        task="What is 3.2 times 4.7?",
        expected_tool_calls=[("calculator", {})],
        expected_final_contains=["15.04"],
        tags=["tool_use", "calculator"],
    ),
    GoldenTask(
        id="search",
        task="Find best practices for ad budget optimization.",
        expected_tool_calls=[("search", {})],
        expected_final_contains=["budget"],
        tags=["tool_use", "search"],
    ),
    GoldenTask(
        id="create_campaign",
        task="Create an app-install campaign with a $1000 daily budget.",
        expected_tool_calls=[("create_campaign", {"objective": "app_installs"})],
        expected_final_contains=["campaign"],
        tags=["ad_platform"],
    ),
    GoldenTask(
        id="campaign_and_metrics",
        task="Create a campaign and then fetch its performance metrics.",
        expected_tool_calls=[("create_campaign", {}), ("get_metrics", {})],
        expected_final_contains=["campaign", "roas"],
        tags=["multi_step", "ad_platform"],
    ),
    GoldenTask(
        id="calc_then_search",
        task="Multiply 2 and 3, then search for the product's ad creative.",
        expected_tool_calls=[("calculator", {}), ("search", {})],
        expected_final_contains=["6"],
        tags=["multi_step", "tool_use"],
    ),
]
