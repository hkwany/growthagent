import unittest

from growthagent.config import Settings
from growthagent.llm import (
    LLMError,
    OpenAIProvider,
    ScriptedLLM,
    _messages_to_openai,
    _parse_openai_tool_calls,
    build_provider,
)
from growthagent.schema import LLMResponse, Message, ToolCall, Usage


class ScriptedLLMTest(unittest.TestCase):
    def test_pops_in_order(self):
        llm = ScriptedLLM([
            LLMResponse(content="first"),
            LLMResponse(content="second"),
        ])
        self.assertEqual(llm.complete([]).content, "first")
        self.assertEqual(llm.complete([]).content, "second")

    def test_exhausted_returns_empty(self):
        llm = ScriptedLLM([LLMResponse(content="only")])
        llm.complete([])
        resp = llm.complete([])
        self.assertEqual(resp.content, "")
        self.assertEqual(resp.tool_calls, [])


class MessageFormatTest(unittest.TestCase):
    def test_messages_to_openai(self):
        msgs = [
            Message("user", "hi"),
            Message("assistant", tool_calls=[ToolCall("calc", {"e": "1+1"}, id="c1")]),
        ]
        out = _messages_to_openai(msgs)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(out[1]["tool_calls"][0]["function"]["name"], "calc")

    def test_parse_tool_calls(self):
        msg = {
            "tool_calls": [
                {"id": "c0", "function": {"name": "calc", "arguments": '{"expression": "1+1"}'}}
            ]
        }
        tcs = _parse_openai_tool_calls(msg)
        self.assertEqual(tcs[0].name, "calc")
        self.assertEqual(tcs[0].arguments, {"expression": "1+1"})


class ProviderFactoryTest(unittest.TestCase):
    def test_openai_requires_key(self):
        with self.assertRaises(LLMError):
            OpenAIProvider(Settings(openai_api_key=""))

    def test_unknown_provider_raises(self):
        with self.assertRaises(LLMError):
            build_provider(Settings(llm_provider="nope"))


if __name__ == "__main__":
    unittest.main()
