import unittest

from growthagent.memory import LongTermMemory, Scratchpad
from growthagent.rag import Retriever


class ScratchpadTest(unittest.TestCase):
    def test_set_get_delete(self):
        s = Scratchpad()
        s.set("k", 1)
        self.assertEqual(s.get("k"), 1)
        self.assertEqual(s.get("missing", 0), 0)
        s.delete("k")
        self.assertNotIn("k", s.keys())

    def test_json_roundtrip(self):
        s = Scratchpad()
        s.set("a", {"x": [1, 2]})
        s2 = Scratchpad()
        s2.from_json(s.to_json())
        self.assertEqual(s2.get("a"), {"x": [1, 2]})


class LongTermMemoryTest(unittest.TestCase):
    def test_remember_and_recall(self):
        m = LongTermMemory()
        m.remember("FocusApp campaign used benefit-led creative, ROAS 0.9.")
        m.remember("Game campaign used UGC creative, ROAS 0.3.")
        hits = m.recall("what worked for the productivity app", k=1)
        self.assertEqual(hits[0][0], "mem_0")
        self.assertIn("FocusApp", hits[0][2]["text"])


class RetrieverTest(unittest.TestCase):
    def test_index_and_search(self):
        r = Retriever()
        r.index([
            {"id": "focus", "text": "FocusApp is a deep-work productivity timer."},
            {"id": "budget", "text": "Allocate budget toward high-ROAS campaigns."},
        ])
        self.assertEqual(len(r), 2)
        hits = r.search("productivity timer", k=1)
        self.assertEqual(hits[0][0], "focus")


if __name__ == "__main__":
    unittest.main()
