"""Judges: deterministic keyword judge + optional LLM-as-judge.

The deterministic :class:`KeywordJudge` is the default and the one used in
CI.  :class:`LLMJudge` exists to demonstrate *why* judge-only eval is
unreliable — it can over-report success relative to tool-call matching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..agent.loop import AgentRun
from ..llm import LLMProvider
from ..schema import Message
from .tasks import GoldenTask

_JUDGE_SYSTEM = (
    "You are an evaluator. Score whether the agent's final answer satisfies "
    "the task, from 0 (completely wrong) to 100 (perfect). Respond with ONLY "
    "an integer score."
)


class KeywordJudge:
    """Deterministic: fraction of required substrings present -> 0..100."""

    def judge(self, run: AgentRun, task: GoldenTask) -> Dict[str, Any]:
        answer = (run.final_answer or "").lower()
        if not task.expected_final_contains:
            score = 100 if answer else 0
        else:
            hits = sum(1 for s in task.expected_final_contains if s.lower() in answer)
            score = round(100 * hits / len(task.expected_final_contains))
        return {"judge": "keyword", "score": score, "reason": f"{score}/100 keyword match"}


class LLMJudge:
    """LLM-as-judge (needs a real provider)."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def judge(self, run: AgentRun, task: GoldenTask) -> Dict[str, Any]:
        prompt = f"Task: {task.task}\n\nAgent answer:\n{run.final_answer or '(empty)'}"
        resp = self.llm.complete(
            [Message("system", _JUDGE_SYSTEM), Message("user", prompt)],
            tools=None,
        )
        score = 0
        try:
            score = int(resp.content.strip().split()[0])
        except Exception:  # noqa: BLE001
            pass
        return {"judge": "llm", "score": score, "reason": resp.content.strip()[:300]}


def make_judge(llm: Optional[LLMProvider] = None) -> Any:
    return LLMJudge(llm) if llm is not None else KeywordJudge()
