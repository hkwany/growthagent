"""Core data structures shared across the project.

Everything is a plain dataclass so it serializes cleanly to JSON for the
trace / ledger / eval layers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Usage:
    """Token usage for one LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ToolCall:
    """A request for the model to call a tool."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class Message:
    """One chat message.  Mirrors the OpenAI message shape."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # tool name for "tool" role messages

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for i, tc in enumerate(self.tool_calls)
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Result of one LLM completion."""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = ""


@dataclass
class TraceEvent:
    """One event in an agent run trace."""

    ts: float
    type: str  # llm_call | tool_call | tool_result | plan | reflect | step | finish | error
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "type": self.type, "data": self.data}


def now() -> float:
    return time.time()


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
