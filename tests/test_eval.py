import unittest

from growthagent.agent.loop import AgentRun
from growthagent.eval import GOLDEN_TASKS, evaluate, run_eval, tool_call_precision
from growthagent.eval.tasks import GoldenTask
from growthagent.llm import LLMResponse, ScriptedLLM, ToolCall, Usage
from growthagent.tools import default_registry


def _task(tid):
    return next(t for t in GOLDEN_TASKS if t.id == tid)


class EvaluateTest(unittest.TestCase):
    def test_success(self):
        run = AgentRun(
            task="x",
            tool_calls=[ToolCall("calculator", {"expression": "3.2*4.7"})],
            final_answer="3.2 * 4.7 = 15.04",
            usage=Usage(total_tokens=10),
        )
        result = evaluate(run, _task("calc"))
        self.assertTrue(result["success"])

    def test_fail_wrong_tools(self):
        run = AgentRun(
            task="x",
            tool_calls=[ToolCall("search", {})],
            final_answer="15.04",
            usage=Usage(total_tokens=10),
        )
        result = evaluate(run, _task("calc"))
        self.assertFalse(result["success"])
        self.assertEqual(result["tool_calls"]["precision"], 0.0)


class ToolCallPrecisionTest(unittest.TestCase):
    def test_precision(self):
        run = AgentRun(
            task="x",
            tool_calls=[ToolCall("calculator", {"expression": "2+2"})],
        )
        out = tool_call_precision(run, GoldenTask(id="t", task="x", expected_tool_calls=[("calculator", {})]))
        self.assertEqual(out["precision"], 1.0)
        self.assertEqual(out["matched"], 1)


class RunEvalTest(unittest.TestCase):
    def test_all_golden_tasks_pass(self):
        scripts = {
            "calc": [LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "3.2*4.7"})]),
                     LLMResponse(content="3.2 * 4.7 = 15.04")],
            "search": [LLMResponse(tool_calls=[ToolCall("search", {"query": "budget optimization"})]),
                       LLMResponse(content="Best practice: shift budget toward high-ROAS campaigns.")],
            "create_campaign": [LLMResponse(tool_calls=[ToolCall("create_campaign", {"objective": "app_installs", "budget": 1000.0, "creative": "x"})]),
                                LLMResponse(content="Created the app-install campaign.")],
            "campaign_and_metrics": [LLMResponse(tool_calls=[ToolCall("create_campaign", {"objective": "app_installs", "budget": 100.0, "creative": "x"})]),
                                     LLMResponse(tool_calls=[ToolCall("get_metrics", {"campaign_id": "camp_1"})]),
                                     LLMResponse(content="Campaign created and metrics fetched: roas 0.9.")],
            "calc_then_search": [LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "2*3"})]),
                                 LLMResponse(tool_calls=[ToolCall("search", {"query": "ad creative"})]),
                                 LLMResponse(content="2*3 = 6. Search done.")],
        }
        report = run_eval(GOLDEN_TASKS, default_registry(),
                          llm_factory=lambda t: ScriptedLLM(scripts[t.id]))
        self.assertEqual(report.success_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
