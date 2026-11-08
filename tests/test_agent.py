import json
import unittest

from growthagent.agent.loop import run_agent
from growthagent.llm import LLMProvider, LLMResponse, ScriptedLLM, ToolCall, Usage
from growthagent.schema import Message
from growthagent.tools import default_registry


def _tool_calls(*names):
    return LLMResponse(tool_calls=[ToolCall(n, {"k": "v"}) for n in names])


class AgentLoopTest(unittest.TestCase):
    def test_completes_with_tool_then_answer(self):
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "3.2*4.7"})]),
            LLMResponse(content="3.2 * 4.7 = 15.04"),
        ])
        run = run_agent("What is 3.2 times 4.7?", llm, default_registry())
        self.assertEqual(run.terminated, "completed")
        self.assertEqual(run.final_answer, "3.2 * 4.7 = 15.04")
        self.assertEqual([tc.name for tc in run.tool_calls], ["calculator"])

    def test_tool_result_fed_back_to_model(self):
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "3.2*4.7"})]),
            LLMResponse(content="15.04"),
        ])
        run = run_agent("calc", llm, default_registry())
        tool_msgs = [m for m in run.messages if m.role == "tool"]
        self.assertTrue(tool_msgs)
        self.assertIn("15.04", tool_msgs[0].content)

    def test_trace_event_types(self):
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "1+1"})]),
            LLMResponse(content="2"),
        ])
        run = run_agent("calc", llm, default_registry())
        types = [e.type for e in run.trace]
        for t in ("llm_call", "tool_call", "tool_result", "finish"):
            self.assertIn(t, types)

    def test_max_steps_termination(self):
        llm = ScriptedLLM([_tool_calls("calculator")] * 20)
        run = run_agent("loop forever", llm, default_registry(), max_steps=2)
        self.assertEqual(run.terminated, "max_steps")
        self.assertEqual(run.steps, 2)

    def test_error_termination(self):
        class Boom(LLMProvider):
            model = "boom"

            def complete(self, messages, tools=None):
                raise RuntimeError("boom")

        run = run_agent("x", Boom(), default_registry())
        self.assertEqual(run.terminated, "error")

    def test_on_event_callback(self):
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("calculator", {"expression": "1+1"})]),
            LLMResponse(content="2"),
        ])
        events = []
        run_agent("calc", llm, default_registry(), on_event=events.append)
        self.assertTrue(events)
        self.assertEqual(events[-1].type, "finish")

    def test_usage_accumulates(self):
        llm = ScriptedLLM([
            LLMResponse(
                tool_calls=[ToolCall("calculator", {"expression": "1+1"})],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            LLMResponse(content="2", usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23)),
        ])
        run = run_agent("calc", llm, default_registry())
        self.assertEqual(run.usage.total_tokens, 38)


if __name__ == "__main__":
    unittest.main()
