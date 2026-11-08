"""The agent loop: plan -> act -> observe -> (reflect).

A single function-calling loop with hard guardrails:

* ``max_steps`` — caps the number of LLM calls (prevents infinite loops),
* ``token_budget`` — caps the context size (prevents context blow-up),
* tool errors are fed back to the model instead of crashing the run.

The loop is provider-agnostic: it only depends on ``llm.complete``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llm import LLMProvider
from ..schema import Message, ToolCall, TraceEvent, Usage, now
from ..tools.base import ToolRegistry
from .context import ContextManager

AgentEventCallback = Callable[[TraceEvent], None]


@dataclass
class AgentRun:
    task: str
    messages: List[Message] = field(default_factory=list)
    steps: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    trace: List[TraceEvent] = field(default_factory=list)
    terminated: str = "completed"  # completed | max_steps | token_budget | error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps": self.steps,
            "tool_calls": [tc.name for tc in self.tool_calls],
            "final_answer": self.final_answer,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "terminated": self.terminated,
        }


def run_agent(
    task: str,
    llm: LLMProvider,
    tools: ToolRegistry,
    *,
    system: Optional[str] = None,
    max_steps: int = 8,
    token_budget: int = 8000,
    use_planner: bool = False,
    use_reflector: bool = False,
    on_event: Optional[AgentEventCallback] = None,
) -> AgentRun:
    """Run one agent task end-to-end and return an :class:`AgentRun`."""
    start = now()
    trace: List[TraceEvent] = []

    def emit(event: TraceEvent) -> None:
        trace.append(event)
        if on_event is not None:
            on_event(event)

    messages: List[Message] = []
    if system:
        messages.append(Message("system", system))

    if use_planner:
        from .planner import generate_plan

        plan = generate_plan(llm, task)
        emit(TraceEvent(now(), "plan", {"steps": plan}))
        if plan and plan != [task]:
            plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
            messages.append(Message("system", f"Follow this plan:\n{plan_text}"))

    messages.append(Message("user", task))

    ctx = ContextManager(token_budget)
    usage = Usage()
    final_answer = ""
    terminated = "completed"
    executed_tool_calls: List[ToolCall] = []
    model = getattr(llm, "model", "unknown")

    steps = 0
    while steps < max_steps:
        steps += 1

        if ctx.over_budget(messages):
            trimmed = ctx.trim(messages)
            if ctx.over_budget(trimmed):
                terminated = "token_budget"
                emit(TraceEvent(now(), "error", {"message": "context still over budget after trim"}))
                break
            emit(TraceEvent(now(), "step", {"message": "context trimmed", "from": len(messages), "to": len(trimmed)}))
            messages = trimmed

        try:
            resp = llm.complete(messages, tools.schemas())
        except Exception as e:  # noqa: BLE001
            terminated = "error"
            emit(TraceEvent(now(), "error", {"message": f"{type(e).__name__}: {e}"}))
            break

        model = resp.model or model
        usage.prompt_tokens += resp.usage.prompt_tokens
        usage.completion_tokens += resp.usage.completion_tokens
        usage.total_tokens += resp.usage.total_tokens
        emit(
            TraceEvent(
                now(),
                "llm_call",
                {
                    "model": model,
                    "content": resp.content[:800],
                    "tool_calls": [tc.name for tc in resp.tool_calls],
                    "usage": resp.usage.to_dict(),
                    "latency_ms": round(resp.latency_ms, 1),
                },
            )
        )

        if resp.tool_calls:
            messages.append(Message("assistant", tool_calls=resp.tool_calls))
            for tc in resp.tool_calls:
                emit(TraceEvent(now(), "tool_call", {"name": tc.name, "arguments": tc.arguments}))
                result_str = tools.execute(tc.name, tc.arguments)
                emit(TraceEvent(now(), "tool_result", {"name": tc.name, "result": result_str[:800]}))
                executed_tool_calls.append(tc)
                messages.append(
                    Message("tool", content=result_str, tool_call_id=tc.id or "", name=tc.name)
                )
            continue

        final_answer = resp.content
        break
    else:
        terminated = "max_steps"

    if use_reflector and final_answer:
        from .reflector import reflect

        reflection = reflect(llm, trace, task)
        emit(TraceEvent(now(), "reflect", {"reflection": reflection[:800]}))

    emit(
        TraceEvent(
            now(),
            "finish",
            {"terminated": terminated, "steps": steps, "tool_calls": len(executed_tool_calls)},
        )
    )

    return AgentRun(
        task=task,
        messages=messages,
        steps=steps,
        tool_calls=executed_tool_calls,
        final_answer=final_answer,
        usage=usage,
        latency_ms=(now() - start) * 1000.0,
        model=model,
        trace=trace,
        terminated=terminated,
    )
