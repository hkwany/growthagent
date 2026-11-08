import unittest

from growthagent.optimizer import BudgetOptimizer, EpsilonGreedyBandit, simulate


class BanditTest(unittest.TestCase):
    def test_select_and_update(self):
        b = EpsilonGreedyBandit(3, epsilon=0.0)  # pure exploit
        # all zero -> pick arm 0 (first max)
        self.assertEqual(b.select(), 0)
        b.update(2, 10.0)
        self.assertEqual(b.best_arm(), 2)
        self.assertEqual(b.select(), 2)

    def test_invalid_arms(self):
        with self.assertRaises(ValueError):
            EpsilonGreedyBandit(0)


class BudgetOptimizerTest(unittest.TestCase):
    def test_recommend_splits_total(self):
        opt = BudgetOptimizer()
        rec = opt.recommend_budget(["a", "b", "c", "d"], total_budget=100.0)
        self.assertAlmostEqual(sum(rec.values()), 100.0, places=2)
        # with no observations, uniform split
        self.assertAlmostEqual(rec["a"], 25.0, places=2)

    def test_observe_shifts_budget(self):
        opt = BudgetOptimizer()
        opt.observe("a", 5.0)
        opt.observe("b", 1.0)
        rec = opt.recommend_budget(["a", "b"], total_budget=100.0)
        self.assertGreater(rec["a"], rec["b"])


class SimulateTest(unittest.TestCase):
    def test_bandit_beats_random(self):
        res = simulate(horizon=500, n_arms=4, n_trials=100, epsilon=0.1,
                       arm_means=[0.1, 0.3, 0.5, 0.8], seed=0)
        self.assertGreaterEqual(res["bandit_final"], res["random_final"])
        self.assertGreater(res["regret_gap_vs_optimal"], 0)


if __name__ == "__main__":
    unittest.main()
