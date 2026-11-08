import unittest

from growthagent.schema import Message
from growthagent.tokenizer import estimate_messages_tokens, estimate_tokens


class TokenizerTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_latin(self):
        # heuristic only — assert it is positive and monotone
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertGreater(estimate_tokens("a" * 400), estimate_tokens("a" * 4))

    def test_cjk(self):
        self.assertGreater(estimate_tokens("你好世界"), 0)

    def test_messages(self):
        msgs = [
            Message("system", "you are helpful"),
            Message("user", "hello"),
        ]
        self.assertGreater(estimate_messages_tokens(msgs), 0)


if __name__ == "__main__":
    unittest.main()
