import unittest

import numpy as np

from growthagent.vector import HashEmbedder, VectorStore


class HashEmbedderTest(unittest.TestCase):
    def test_shape_and_normalization(self):
        e = HashEmbedder(dim=64)
        v = e.embed("hello world")
        self.assertEqual(v.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_similar_texts_closer(self):
        e = HashEmbedder(dim=256)
        a = e.embed("productivity app for deep work")
        b = e.embed("productivity app for deep work")
        c = e.embed("cooking recipes and baking")
        self.assertGreater(float(a @ b), float(a @ c))


class VectorStoreTest(unittest.TestCase):
    def test_add_and_search(self):
        s = VectorStore(dim=16)
        e = HashEmbedder(dim=16)
        s.add("a", e.embed("deep work productivity"), {"text": "a"})
        s.add("b", e.embed("cooking recipes"), {"text": "b"})
        # exact-text query -> cosine similarity ~1.0 for the matching doc
        hits = s.search(e.embed("deep work productivity"), k=1)
        self.assertEqual(hits[0][0], "a")
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)
        # partial query still ranks the closest doc first
        hits2 = s.search(e.embed("productivity"), k=2)
        self.assertEqual(hits2[0][0], "a")
        self.assertGreater(hits2[0][1], hits2[1][1])

    def test_empty_store(self):
        s = VectorStore(dim=8)
        self.assertEqual(s.search(np.zeros(8)), [])

    def test_dim_mismatch_raises(self):
        s = VectorStore(dim=8)
        with self.assertRaises(ValueError):
            s.add("x", np.zeros(16))


if __name__ == "__main__":
    unittest.main()
