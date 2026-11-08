"""Multi-armed bandit for budget/bid optimization (the "Learn Engine").

Covers the ML keyword in the JD with a small, well-tested component:
epsilon-greedy allocation over campaign arms, plus an offline simulation
that demonstrates the bandit beating a random (equal-split) baseline.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence


class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.1, rng: Optional[random.Random] = None):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = rng or random.Random(0)
        self.counts: List[int] = [0] * n_arms
        self.values: List[float] = [0.0] * n_arms

    def select(self) -> int:
        """Pick an arm: explore with prob ``epsilon``, else exploit best.

        Ties are broken toward the least-played arm so unplayed arms get
        tried before re-exploiting an already-played one.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_arms)
        best = max(
            range(self.n_arms),
            key=lambda i: (self.values[i] if self.counts[i] > 0 else 0.0, -self.counts[i]),
        )
        return best

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def best_arm(self) -> int:
        return max(range(self.n_arms), key=lambda i: self.values[i])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counts": list(self.counts),
            "values": [round(v, 4) for v in self.values],
            "best_arm": self.best_arm(),
        }


class BudgetOptimizer:
    """Allocates a fixed budget across campaigns using a bandit."""

    def __init__(self, epsilon: float = 0.1, rng: Optional[random.Random] = None):
        self.epsilon = epsilon
        self.rng = rng or random.Random(0)
        self._arms: Dict[str, int] = {}
        self._bandit: Optional[EpsilonGreedyBandit] = None

    def _ensure_arms(self, campaign_ids: Sequence[str]) -> None:
        added = False
        for c in campaign_ids:
            if c not in self._arms:
                self._arms[c] = len(self._arms)
                added = True
        if self._bandit is None:
            self._bandit = EpsilonGreedyBandit(len(self._arms), self.epsilon, self.rng)
        elif added:
            # Grow the bandit without losing prior observations.
            old = self._bandit
            new = EpsilonGreedyBandit(len(self._arms), self.epsilon, self.rng)
            new.counts = old.counts + [0] * (len(self._arms) - len(old.counts))
            new.values = old.values + [0.0] * (len(self._arms) - len(old.values))
            self._bandit = new

    def recommend_budget(self, campaign_ids: Sequence[str], total_budget: float) -> Dict[str, float]:
        """Split ``total_budget`` across campaigns by estimated value."""
        self._ensure_arms(campaign_ids)
        assert self._bandit is not None
        values = self._bandit.values
        # greedy share for known arms, uniform for unknown
        raw = [values[self._arms[c]] if self._bandit.counts[self._arms[c]] > 0 else 0.0 for c in campaign_ids]
        if sum(raw) <= 0:
            share = [1.0 / len(campaign_ids)] * len(campaign_ids)
        else:
            total_v = sum(raw)
            share = [r / total_v for r in raw]
        return {c: round(total_budget * s, 2) for c, s in zip(campaign_ids, share)}

    def observe(self, campaign_id: str, reward: float) -> None:
        self._ensure_arms([campaign_id])
        assert self._bandit is not None
        self._bandit.update(self._arms[campaign_id], reward)

    def snapshot(self) -> Dict[str, Any]:
        if self._bandit is None:
            return {"arms": {}, "bandit": None}
        return {"arms": dict(self._arms), "bandit": self._bandit.snapshot()}


def simulate(
    horizon: int = 500,
    n_arms: int = 4,
    n_trials: int = 200,
    epsilon: float = 0.1,
    arm_means: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Offline comparison: epsilon-greedy vs random, averaged over trials.

    Returns cumulative-reward arrays and the per-trial final regret gap.
    """
    arm_means = list(arm_means) if arm_means is not None else [0.1, 0.3, 0.5, 0.8]
    assert n_arms == len(arm_means)
    best_mean = max(arm_means)

    def run_one(policy: str, rng: random.Random) -> List[float]:
        if policy == "bandit":
            b = EpsilonGreedyBandit(n_arms, epsilon, rng)
        elif policy == "random":
            b = EpsilonGreedyBandit(n_arms, epsilon=1.0, rng=rng)  # always explore
        else:
            raise ValueError(policy)
        cum: List[float] = []
        total = 0.0
        for _ in range(horizon):
            arm = b.select()
            reward = 1.0 if rng.random() < arm_means[arm] else 0.0
            b.update(arm, reward)
            total += reward
            cum.append(total)
        return cum

    bandit_cum = [0.0] * horizon
    random_cum = [0.0] * horizon
    for t in range(n_trials):
        r1, r2 = random.Random(seed + t), random.Random(seed + 10_000 + t)
        bc = run_one("bandit", r1)
        rc = run_one("random", r2)
        for i in range(horizon):
            bandit_cum[i] += bc[i]
            random_cum[i] += rc[i]
    bandit_cum = [x / n_trials for x in bandit_cum]
    random_cum = [x / n_trials for x in random_cum]
    regret_gap = (bandit_cum[-1] - random_cum[-1]) / (horizon * best_mean)
    return {
        "horizon": horizon,
        "n_arms": n_arms,
        "n_trials": n_trials,
        "bandit_cumulative": bandit_cum,
        "random_cumulative": random_cum,
        "bandit_final": round(bandit_cum[-1], 3),
        "random_final": round(random_cum[-1], 3),
        "regret_gap_vs_optimal": round(regret_gap, 4),
    }
