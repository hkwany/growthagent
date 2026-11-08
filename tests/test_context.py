import unittest

from growthagent.agent.context import ContextManager, summarize
from growthagent.schema import Message


class SummarizeTest(unittest.TestCase):
    def test_short_unchanged(self):
        self.assertEqual(summarize("hello", max_chars=10), "hello")

    def test_long_truncated(self):
        out = summarize("x" * 1000, max_chars=20)
        self.assertIn("truncated", out)
        self.assertLessEqual(len(out), 20 + 30)  # marker adds some chars


class ContextManagerTest(unittest.TestCase):
    def _make_messages(self):
        return [
            Message("system", "sys"),
            Message("user", "the task"),
            Message("assistant", tool_calls=[]),
            Message("tool", "r1", name="t1"),
            Message("assistant", tool_calls=[]),
            Message("tool", "r2", name="t2"),
            Message("assistant", tool_calls=[]),
            Message("tool", "r3", name="t3"),
        ]

    def test_over_budget(self):
        cm = ContextManager(max_tokens=1)
        self.assertTrue(cm.over_budget(self._make_messages()))
        cm2 = ContextManager(max_tokens=10_000_000)
        self.assertFalse(cm2.over_budget(self._make_messages()))

    def test_trim_keeps_system_task_recent(self):
        cm = ContextManager(max_tokens=1)
        msgs = self._make_messages()
        trimmed = cm.trim(msgs, keep_recent=2)
        roles = [m.role for m in trimmed]
        # system first, task (first user) kept, then recent messages
        self.assertEqual(roles[0], "system")
        self.assertIn("user", roles)
        self.assertEqual(trimmed[0].content, "sys")
        # recent two non-system messages are the last two tool msgs
        self.assertTrue(any("r3" in m.content for m in trimmed))
        # middle dropped -> a marker was inserted
        self.assertTrue(any("history compressed" in m.content for m in trimmed))


if __name__ == "__main__":
    unittest.main()
