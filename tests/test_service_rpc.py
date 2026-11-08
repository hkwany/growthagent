import json
import unittest

from growthagent.app.service import GrowthAgentService
from growthagent.llm import LLMResponse, ScriptedLLM, ToolCall
from growthagent.rpc import LocalToolService
from growthagent.tools import default_registry


class ServiceTest(unittest.TestCase):
    def test_run_with_scripted_llm(self):
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "1+1"})]),
            LLMResponse(content="1+1 = 2"),
        ])
        svc = GrowthAgentService()
        run = svc.run("what is 1+1", llm=llm)
        self.assertEqual(run.final_answer, "1+1 = 2")
        self.assertEqual(run.terminated, "completed")

    def test_run_summary(self):
        llm = ScriptedLLM([LLMResponse(content="done")])
        svc = GrowthAgentService()
        summary = svc.run_summary("hello", llm=llm)
        self.assertEqual(summary["final_answer"], "done")
        self.assertIn("usage", summary)


class LocalToolServiceTest(unittest.TestCase):
    def test_call_json(self):
        svc = LocalToolService(default_registry())
        result = svc.call_json("calculator", json.dumps({"expression": "2+3"}))
        self.assertEqual(json.loads(result)["result"], 5)

    def test_unknown_tool(self):
        svc = LocalToolService(default_registry())
        result = svc.call_json("nope", "{}")
        self.assertIn("error", json.loads(result))


if __name__ == "__main__":
    unittest.main()
