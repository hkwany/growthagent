#!/usr/bin/env python3
"""End-to-end demo of growthagent — no API key, no network required.

Runs four things:

1. a full "autonomous growth agent" run (scripted LLM) with trace,
2. the eval harness over all golden tasks,
3. the bandit optimizer (random vs epsilon-greedy),
4. RAG + long-term memory retrieval,

then prints a report.  Exit code 0 means everything worked.

Run:  python examples/run_growth_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from growthagent.agent.loop import run_agent
from growthagent.eval import GOLDEN_TASKS, make_judge, run_eval
from growthagent.llm import LLMResponse, ScriptedLLM, ToolCall, Usage
from growthagent.memory import LongTermMemory
from growthagent.optimizer import simulate
from growthagent.rag import Retriever
from growthagent.schema import Message
from growthagent.tools import default_registry

SEP = "=" * 72


def hr(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def growth_run():
    """One scripted growth flow: research -> create -> measure -> report."""
    tools = default_registry()

    script = [
        LLMResponse(
            tool_calls=[ToolCall("fetch_url", {"url": "https://example.com/landing/focusapp"})],
            usage=Usage(prompt_tokens=120, completion_tokens=20, total_tokens=140),
        ),
        LLMResponse(
            tool_calls=[ToolCall("search", {"query": "productivity app ads"})],
            usage=Usage(prompt_tokens=400, completion_tokens=25, total_tokens=425),
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "create_campaign",
                    {
                        "objective": "app_installs",
                        "budget": 1000.0,
                        "creative": "Get 2x more deep work — 15s vertical demo, CTA in first 3s.",
                    },
                )
            ],
            usage=Usage(prompt_tokens=600, completion_tokens=40, total_tokens=640),
        ),
        LLMResponse(
            tool_calls=[ToolCall("get_metrics", {"campaign_id": "camp_1", "spend": 500.0})],
            usage=Usage(prompt_tokens=700, completion_tokens=25, total_tokens=725),
        ),
        LLMResponse(
            content=(
                "Created campaign camp_1 (app_installs, $1000 budget) with a benefit-led "
                "creative. After $500 spend the campaign is live and generating metrics. "
                "Recommendation: keep the benefit-led creative, hold budget, and re-evaluate "
                "after 50 conversions for statistical significance."
            ),
            usage=Usage(prompt_tokens=900, completion_tokens=60, total_tokens=960),
        ),
    ]
    llm = ScriptedLLM(script)

    run = run_agent(
        "Launch a growth campaign for FocusApp (deep-work productivity timer).",
        llm,
        tools,
        system="You are an autonomous growth agent.",
        max_steps=8,
        token_budget=8000,
    )

    print(f"task: {run.task}")
    print(f"terminated: {run.terminated}  steps: {run.steps}  model: {run.model}")
    print(f"tokens: {run.usage.total_tokens}  latency: {run.latency_ms:.1f}ms\n")
    print("trace:")
    for e in run.trace:
        if e.type == "llm_call":
            tcs = e.data.get("tool_calls") or []
            print(f"  [{e.type}] tools={tcs} usage={e.data.get('usage')}")
        elif e.type == "tool_result":
            print(f"  [{e.type}] {e.data.get('name')} -> {e.data.get('result')[:120]}")
        elif e.type in ("tool_call", "finish", "plan", "reflect", "step"):
            print(f"  [{e.type}] {e.data}")
        elif e.type == "error":
            print(f"  [{e.type}] {e.data}")
    print(f"\nfinal answer:\n{run.final_answer}")
    return run


def eval_run():
    """Eval harness over all golden tasks, using a per-task scripted LLM."""
    tools = default_registry()

    def script_for(task_id: str):
        if task_id == "calc":
            return [LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "3.2*4.7"})]),
                    LLMResponse(content="3.2 * 4.7 = 15.04")]
        if task_id == "search":
            return [LLMResponse(tool_calls=[ToolCall("search", {"query": "budget optimization"})]),
                    LLMResponse(content="Best practice: shift budget toward high-ROAS campaigns.")]
        if task_id == "create_campaign":
            return [LLMResponse(tool_calls=[ToolCall("create_campaign", {"objective": "app_installs", "budget": 1000.0, "creative": "x"})]),
                    LLMResponse(content="Created the app-install campaign.")]
        if task_id == "campaign_and_metrics":
            return [LLMResponse(tool_calls=[ToolCall("create_campaign", {"objective": "app_installs", "budget": 100.0, "creative": "x"})]),
                    LLMResponse(tool_calls=[ToolCall("get_metrics", {"campaign_id": "camp_1"})]),
                    LLMResponse(content="Campaign created and metrics fetched: roas 0.9.")]
        if task_id == "calc_then_search":
            return [LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "2*3"})]),
                    LLMResponse(tool_calls=[ToolCall("search", {"query": "ad creative"})]),
                    LLMResponse(content="2*3 = 6. Search done.")]
        return [LLMResponse(content="done")]

    report = run_eval(GOLDEN_TASKS, tools, llm_factory=lambda t: ScriptedLLM(script_for(t.id)))
    judge = make_judge()
    print(f"tasks: {report.to_dict()['n_tasks']}  "
          f"success_rate: {report.success_rate:.3f}  "
          f"avg_cost_tokens: {report.avg_cost_tokens:.1f}  "
          f"avg_latency_ms: {report.avg_latency_ms:.1f}")
    print()
    for r in report.results:
        j = judge.judge(_run_from_result(r), _task_from_id(r["task_id"]))
        print(f"  {r['task_id']:<20} success={str(r['success']):<5} "
              f"tools={r['tool_calls']['executed']} judge_score={j['score']}")
    return report


def _task_from_id(task_id: str):
    for t in GOLDEN_TASKS:
        if t.id == task_id:
            return t
    raise KeyError(task_id)


def _run_from_result(r):
    from growthagent.agent.loop import AgentRun

    return AgentRun(
        task=r["task"],
        final_answer=r["final_answer"],
        tool_calls=[],
        steps=r["steps"],
        terminated=r["terminated"],
    )


def bandit_run():
    """Offline bandit simulation: random (equal split) vs epsilon-greedy."""
    res = simulate(horizon=500, n_arms=4, n_trials=200, epsilon=0.1,
                   arm_means=[0.1, 0.3, 0.5, 0.8], seed=0)
    print(f"horizon={res['horizon']} arms={res['n_arms']} trials={res['n_trials']}")
    print(f"  random (equal split)  final cumulative reward: {res['random_final']}")
    print(f"  epsilon-greedy        final cumulative reward: {res['bandit_final']}")
    print(f"  gap vs optimal: {res['regret_gap_vs_optimal']:.2%}")
    return res


def rag_run():
    """RAG retrieval + long-term memory recall."""
    docs = [
        {"id": "focusapp", "text": "FocusApp is a deep-work productivity timer for developers. Benefit: 2x focused hours."},
        {"id": "budget", "text": "Allocate ad budget toward campaigns with the highest recent ROAS; re-evaluate weekly."},
        {"id": "creative", "text": "UGC-style 15s vertical video with a clear CTA outperforms feature lists."},
    ]
    retriever = Retriever()
    retriever.index(docs)
    hits = retriever.search("how to make an ad creative for a productivity app", k=2)
    print("RAG query: 'how to make an ad creative for a productivity app'")
    for doc_id, score, doc in hits:
        print(f"  -> {doc_id} (score={score:.3f}): {doc['text'][:60]}")

    mem = LongTermMemory()
    mem.remember("Campaign for FocusApp used benefit-led creative and got ROAS 0.9.")
    mem.remember("Campaign for a game used UGC creative and got ROAS 0.3.")
    recall = mem.recall("what worked for the productivity app", k=1)
    print("memory recall: 'what worked for the productivity app'")
    for mid, score, meta in recall:
        print(f"  -> {mid} (score={score:.3f}): {meta.get('text', '')[:60]}")
    return hits, recall


def main() -> int:
    hr("1) Autonomous growth agent run (scripted LLM)")
    growth_run()

    hr("2) Eval harness over golden tasks")
    eval_run()

    hr("3) Bandit optimizer (random vs epsilon-greedy)")
    bandit_run()

    hr("4) RAG retrieval + long-term memory")
    rag_run()

    hr("DONE")
    print("All components ran successfully (no API key required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
