"""Reflection: post-run self-critique used to improve the next run."""

from __future__ import annotations

from typing import Any, Dict

from ..llm import LLMProvider
from ..schema import Message

_REFLECT_SYSTEM = (
    "You are a critical reviewer. Given a summary of an agent run, list in "
    "2-3 short bullets what went wrong and what to change next time. Be "
    "specific. Respond in plain text."
)


def _trace_summary(trace: List[Any]) -> Dict[str, Any]:
    tool_calls = [e.data.get("name") for e in trace if e.type == "tool_call"]
    errors = [e.data.get("result") for e in trace if e.type == "tool_result" and "error" in str(e.data.get("result", ""))]
    finish = next((e.data for e in reversed(trace) if e.type == "finish"), {})
    return {
        "tool_calls": tool_calls,
        "tool_errors": errors,
        "terminated": finish.get("terminated", "unknown"),
        "steps": finish.get("steps", 0),
    }


def reflect(llm: LLMProvider, trace: List[Any], task: str = "") -> str:
    summary = _trace_summary(trace)
    user_text = (
        f"Task: {task}\nRun summary: {summary}\n\n"
        "What should be improved next run?"
    )
    try:
        resp = llm.complete(
            [Message("system", _REFLECT_SYSTEM), Message("user", user_text)],
            tools=None,
        )
        return resp.content.strip()
    except Exception as e:  # noqa: BLE001
        return f"reflection unavailable: {e}"
