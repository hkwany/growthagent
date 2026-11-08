"""Eval metrics: did the run succeed, call the right tools, and at what cost?

The core signal is *tool-call matching* (deterministic), not LLM-as-judge —
this is deliberate: judge-only evals over-report quality (see README §
"Evaluation").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agent.loop import AgentRun, run_agent
from ..llm import LLMProvider
from ..tools.base import ToolRegistry
from .tasks import GoldenTask


def _args_subset_match(subset: Dict[str, Any], full: Dict[str, Any]) -> bool:
    return all(full.get(k) == v for k, v in subset.items())


def tool_call_precision(run: AgentRun, task: GoldenTask) -> Dict[str, Any]:
    expected = list(task.expected_tool_calls)
    executed = [(tc.name, tc.arguments) for tc in run.tool_calls]
    matched = 0
    used: List[bool] = [False] * len(executed)
    for name, args_subset in expected:
        for i, (ename, eargs) in enumerate(executed):
            if used[i]:
                continue
            if ename == name and _args_subset_match(args_subset, eargs):
                matched += 1
                used[i] = True
                break
    expected_n = len(expected) or 1
    precision = matched / expected_n
    recall = matched / len(executed) if executed else (1.0 if not expected else 0.0)
    return {
        "expected": [t[0] for t in expected],
        "executed": [t[0] for t in executed],
        "matched": matched,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


def evaluate(run: AgentRun, task: GoldenTask) -> Dict[str, Any]:
    """Score one run against one golden task."""
    answer = run.final_answer or ""
    content_ok = all(s.lower() in answer.lower() for s in task.expected_final_contains)
    tc = tool_call_precision(run, task)
    success = bool(answer) and content_ok and tc["precision"] >= 1.0
    return {
        "task_id": task.id,
        "task": task.task,
        "success": success,
        "content_ok": content_ok,
        "tool_calls": tc,
        "cost_tokens": run.usage.total_tokens,
        "latency_ms": round(run.latency_ms, 1),
        "steps": run.steps,
        "terminated": run.terminated,
        "final_answer": answer[:300],
    }


@dataclass
class EvalReport:
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r["success"]) / len(self.results)

    @property
    def avg_cost_tokens(self) -> float:
        if not self.results:
            return 0.0
        return sum(r["cost_tokens"] for r in self.results) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r["latency_ms"] for r in self.results) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_tasks": len(self.results),
            "success_rate": round(self.success_rate, 3),
            "avg_cost_tokens": round(self.avg_cost_tokens, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "results": self.results,
        }


def run_eval(
    tasks: List[GoldenTask],
    tools: ToolRegistry,
    llm: Optional[LLMProvider] = None,
    llm_factory: Optional[Any] = None,
    **run_kwargs: Any,
) -> EvalReport:
    """Run the agent on every task and score the results.

    ``llm_factory(task)`` (if given) builds a fresh provider per task —
    required for ``ScriptedLLM``, whose script is per-task.
    """
    report = EvalReport()
    for task in tasks:
        provider = llm_factory(task) if llm_factory is not None else llm
        if provider is None:
            raise ValueError("run_eval needs llm= or llm_factory=")
        run = run_agent(task.task, provider, tools, **run_kwargs)
        report.results.append(evaluate(run, task))
    return report
