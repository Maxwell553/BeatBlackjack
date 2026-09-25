"""Monte Carlo control for play, and a true-count rule for insurance."""

from __future__ import annotations

import numpy as np

N_BASE = 22 * 2 * 11 * 10 * 2 * 2 * 2


def _base_index(ctx: dict) -> int:
    total = int(ctx["total"])
    total = 0 if total < 0 else 21 if total > 21 else total
    soft = 1 if ctx["soft"] else 0
    pair = int(ctx["pair_rank"]) if ctx["can_split"] else 0
    pair = 0 if pair < 0 else 10 if pair > 10 else pair
    up = int(ctx["up"]) - 1
    idx = total
    idx = idx * 2 + soft
    idx = idx * 11 + pair
    idx = idx * 10 + up
    idx = idx * 2 + (1 if ctx["can_double"] else 0)
    idx = idx * 2 + (1 if ctx["can_split"] else 0)
    idx = idx * 2 + (1 if ctx["can_surrender"] else 0)
    return idx


class QTable:
    """Every-visit averages of the return for each legal decision."""

    def __init__(self):
        self.q = np.zeros((N_BASE, 5), np.float64)
        self.q_n = np.zeros((N_BASE, 5), np.int32)
        self.q_sum = self.q

    def reset(self) -> None:
        self.q.fill(0.0)
        self.q_n.fill(0)

    def update(self, ctx: dict, action: int, target: float, alpha: float = 0.12) -> None:
        b = _base_index(ctx)
        n = int(self.q_n[b, action])
        if n == 0:
            self.q[b, action] = float(target)
        else:
            self.q[b, action] += alpha * (float(target) - self.q[b, action])
        self.q_n[b, action] = n + 1

    def max_q(self, ctx: dict, legal: np.ndarray) -> float:
        b = _base_index(ctx)
        scores = []
        for action in np.flatnonzero(legal > 0.0):
            if self.q_n[b, action] > 0:
                scores.append(self.q[b, action])
        if not scores:
            return 0.0
        return float(max(scores))

    def act(self, ctx: dict, legal: np.ndarray, rng: np.random.Generator | None, epsilon: float) -> int:
        options = np.flatnonzero(legal > 0.0)
        if len(options) == 0:
            return 1
        if rng is not None and rng.random() < epsilon:
            return int(rng.choice(options))
        b = _base_index(ctx)
        scores = np.full(5, -1e9)
        tried = (legal > 0.0) & (self.q_n[b] > 0)
        if not np.any(tried):
            return int(options[0])
        scores[tried] = self.q[b, tried]
        return int(np.argmax(scores))


class InsuranceRule:
    """Take insurance at a true count only when the average return there is positive."""

    def __init__(self):
        self.sum = np.zeros(21, np.float64)
        self.n = np.zeros(21, np.int32)

    def reset(self) -> None:
        self.sum.fill(0.0)
        self.n.fill(0)

    def _bin(self, true_count: float) -> int:
        return int(np.clip(np.rint(true_count), -10, 10)) + 10

    def update(self, true_count: float, take_return: float) -> None:
        i = self._bin(true_count)
        self.sum[i] += float(take_return)
        self.n[i] += 1

    def take(self, true_count: float, min_samples: int = 30) -> bool:
        i = self._bin(true_count)
        if int(self.n[i]) < min_samples:
            return False
        return self.sum[i] / self.n[i] > 0.0

    def threshold(self) -> float | None:
        """Lowest true count at which the learned rule takes insurance and stays on it."""
        taking = False
        start = None
        for tc in range(-10, 11):
            on = self.take(float(tc), min_samples=30)
            if on and not taking:
                start = float(tc)
                taking = True
            elif not on:
                taking = False
                start = None
        return start
