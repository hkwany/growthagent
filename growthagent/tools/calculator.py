"""Safe arithmetic evaluator tool (no ``eval``).

Only allows numeric literals, the four basic operators, ``**``, unary
``+/-`` and a tiny whitelist of names (``pi``, ``e``).
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

from .base import tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    return _eval_node(ast.parse(expression, mode="eval").body)


@tool(
    name="calculator",
    description="Evaluate a safe arithmetic expression and return the numeric result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression using + - * / ** % and parentheses, e.g. '3.2 * 4.7'.",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> Dict[str, Any]:
    try:
        return {"result": safe_eval(expression)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
