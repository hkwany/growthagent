"""Plan generation: ask the LLM to lay out steps before acting."""

from __future__ import annotations

import json
import re
from typing import Any, List

from ..llm import LLMProvider
from ..schema import Message

_PLAN_SYSTEM = (
    "You are a planning assistant. Break the task into a numbered list of "
    "concrete steps. Respond with ONLY a JSON array of strings, no markdown, "
    'e.g. ["step 1", "step 2"].'
)


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model reply (strips fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start == -1:
        start = text.find("{")
    if start == -1:
        raise ValueError("no JSON found")
    return json.loads(text[start:])


def generate_plan(llm: LLMProvider, task: str, max_steps: int = 5) -> List[str]:
    """Return a list of plan steps, or ``[task]`` as a safe fallback."""
    try:
        resp = llm.complete(
            [Message("system", _PLAN_SYSTEM), Message("user", task)],
            tools=None,
        )
        data = _extract_json(resp.content)
        if isinstance(data, list):
            steps = [str(s) for s in data]
        elif isinstance(data, dict) and isinstance(data.get("steps"), list):
            steps = [str(s) for s in data["steps"]]
        else:
            raise ValueError("unexpected plan shape")
        steps = [s.strip() for s in steps if str(s).strip()]
        return steps[:max_steps] or [task]
    except Exception:  # noqa: BLE001 - fall back to no plan
        return [task]
