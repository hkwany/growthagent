"""Service logic (framework-agnostic).

This is the testable core that the FastAPI app wraps.  It wires a
provider + tool registry + agent loop together and optionally persists
traces and ledger entries.

Keeping this free of web-framework imports means the whole pipeline is
verified without FastAPI installed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..agent.loop import AgentRun, run_agent
from ..config import Settings, get_settings
from ..llm import LLMProvider, build_provider
from ..observability.ledger import TokenLedger, ledger_from_settings
from ..observability.trace import TraceWriter
from ..schema import TraceEvent
from ..tools import default_registry

SYSTEM_PROMPT = (
    "You are an autonomous growth agent. Plan, then use the provided tools to "
    "research, create campaigns, read metrics and make budget decisions. "
    "End with a concise final report containing concrete numbers and a "
    "recommendation."
)


class GrowthAgentService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def run(
        self,
        task: str,
        llm: Optional[LLMProvider] = None,
        on_event: Optional[Callable[[TraceEvent], None]] = None,
        use_planner: bool = False,
        use_reflector: bool = False,
        write_trace: bool = False,
        write_ledger: bool = False,
    ) -> AgentRun:
        """Run one task and return the :class:`AgentRun`."""
        provider = llm or build_provider(self.settings)
        tools = default_registry()
        run = run_agent(
            task,
            provider,
            tools,
            system=SYSTEM_PROMPT,
            max_steps=self.settings.max_steps,
            token_budget=self.settings.token_budget,
            use_planner=use_planner,
            use_reflector=use_reflector,
            on_event=on_event,
        )
        if write_trace:
            TraceWriter(self.settings.trace_dir + "/run.jsonl").write(run)
        if write_ledger:
            ledger_from_settings(self.settings).record(
                run.model, run.usage, run.latency_ms
            )
        return run

    def run_summary(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        run = self.run(task, **kwargs)
        return {
            "task": run.task,
            "final_answer": run.final_answer,
            "terminated": run.terminated,
            "steps": run.steps,
            "tool_calls": [tc.name for tc in run.tool_calls],
            "usage": run.usage.to_dict(),
            "latency_ms": round(run.latency_ms, 1),
            "model": run.model,
        }


__all__ = ["GrowthAgentService", "SYSTEM_PROMPT", "TokenLedger"]
