"""Bet sizing fit from observed hand results. Half-Kelly, with an optional sit-out."""

from __future__ import annotations

import numpy as np

from blackjack_solver.stats import ols_cluster


class BetModel:
    def __init__(self, intercept: float, slope: float, variance: float, bankroll: float = 800.0, max_bet: float = 12.0):
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.variance = float(max(variance, 1.0))
        self.bankroll = float(bankroll)
        self.max_bet = float(max_bet)

    def mu(self, true_count: float) -> float:
        # A linear edge in the true count, clipped so a bad fit cannot demand a wild stake.
        raw = self.intercept + self.slope * true_count
        return float(np.clip(raw, -0.05, 0.08))

    def bet(self, true_count: float, wong: bool) -> float:
        edge = self.mu(true_count)
        if edge <= 0.0:
            return 0.0 if wong else 1.0
        stake = self.bankroll * edge / self.variance
        return float(np.clip(stake, 1.0, self.max_bet))


def fit_tc_model(true_counts: np.ndarray, profits: np.ndarray, clusters: np.ndarray) -> tuple[BetModel, dict]:
    fit = ols_cluster(profits, true_counts, clusters)
    resid = profits - (fit["intercept"] + fit["slope"] * true_counts)
    variance = float(np.var(resid, ddof=2))
    model = BetModel(fit["intercept"], fit["slope"], variance)
    fit["variance"] = variance
    return model, fit
