"""Tool registry and decorator.

A tool is a plain Python function annotated with a name, description and a
JSON Schema for its arguments.  The registry turns that into the
``tools`` array passed to LLM providers and dispatches ``tool_calls`` back
to the functions.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], Tool]:
    """Decorate a function to become a :class:`Tool`.

    ``parameters`` must be a full JSON Schema object, e.g.::

        {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "..."}},
            "required": ["expression"],
        }
    """

    def deco(func: Callable) -> Tool:
        schema = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        return Tool(name, description, schema, func)

    return deco


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return its result as a JSON string.

        Errors are captured and returned as ``{"error": ...}`` so the agent
        loop can feed them back to the model instead of crashing.
        """
        t = self._tools.get(name)
        if t is None:
            return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
        try:
            result = t.func(**(arguments or {}))
        except Exception as e:  # noqa: BLE001 - surface errors to the model
            result = {"error": f"{type(e).__name__}: {e}"}
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
