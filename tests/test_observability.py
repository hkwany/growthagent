import json
import os
import tempfile
import unittest

from growthagent.observability import TokenLedger, TraceWriter, run_to_jsonl
from growthagent.schema import Usage


class TokenLedgerTest(unittest.TestCase):
    def test_report_totals(self):
        ledger = TokenLedger(prices={"prompt_per_1k": 1.0, "completion_per_1k": 2.0})
        ledger.record("m1", Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500))
        ledger.record("m1", Usage(prompt_tokens=500, completion_tokens=0, total_tokens=500))
        rep = ledger.report()
        self.assertEqual(rep["totals"]["calls"], 2)
        self.assertEqual(rep["totals"]["total_tokens"], 2000)
        # cost = 1500/1000*1.0 + 500/1000*2.0 = 1.5 + 1.0 = 2.5
        self.assertAlmostEqual(rep["totals"]["cost"], 2.5)

    def test_ledger_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            ledger = TokenLedger(path=path)
            ledger.record("m", Usage(total_tokens=10))
            self.assertTrue(os.path.exists(path))


class TraceWriterTest(unittest.TestCase):
    def test_write_and_parse(self):
        from growthagent.agent.loop import AgentRun

        run = AgentRun(task="t", final_answer="ok", terminated="completed")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            TraceWriter(path).write(run)
            with open(path, encoding="utf-8") as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertTrue(lines)
            self.assertEqual(lines[0]["type"], "run_header")
            self.assertIn("task", lines[0])


class RunToJsonlTest(unittest.TestCase):
    def test_serializable(self):
        from growthagent.agent.loop import AgentRun

        run = AgentRun(task="t", final_answer="ok")
        lines = run_to_jsonl(run)
        for ln in lines:
            json.loads(ln)  # must parse


if __name__ == "__main__":
    unittest.main()
