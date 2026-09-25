"""Optimal play for a fixed card mix, used to teach the network count-dependent decisions."""

from __future__ import annotations

import numpy as np

from blackjack_solver.constants import DOUBLE, HIT, SPLIT, STAND, SURRENDER
from blackjack_solver.exact import add_card, hand_value, stand_ev


class MixStrategy:
    def __init__(self, probs: np.ndarray):
        self.p = np.asarray(probs, dtype=np.float64)
        self.p = self.p / self.p[1:].sum()
        self.dealer: dict[int, np.ndarray] = {}
        self.cache: dict[tuple, tuple[float, int]] = {}
        self._build_dealer()

    def _dealer_from(self, total: int, soft: int, memo: dict) -> np.ndarray:
        key = (total, soft)
        if key in memo:
            return memo[key]
        if total > 21:
            dist = np.zeros(6)
            dist[0] = 1.0
            memo[key] = dist
            return dist
        if total >= 17:
            dist = np.zeros(6)
            dist[total - 16] = 1.0
            memo[key] = dist
            return dist
        acc = np.zeros(6)
        for rank in range(1, 11):
            nxt, nxt_soft = add_card(total, soft, rank)
            acc += self.p[rank] * self._dealer_from(nxt, nxt_soft, memo)
        memo[key] = acc
        return acc

    def _build_dealer(self) -> None:
        memo: dict = {}
        for up in range(1, 11):
            acc = np.zeros(6)
            mass = 0.0
            for hole in range(1, 11):
                if (up == 1 and hole == 10) or (up == 10 and hole == 1):
                    continue
                total, soft = hand_value([up, hole])
                acc += self.p[hole] * self._dealer_from(total, soft, memo)
                mass += self.p[hole]
            self.dealer[up] = acc / mass

    def best(
        self,
        total: int,
        soft: int,
        pair_rank: int,
        up: int,
        can_double: int,
        can_split: int,
        can_surrender: int,
        splits_left: int,
    ) -> tuple[float, int]:
        key = (total, soft, pair_rank, up, can_double, can_split, can_surrender, splits_left)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        dist = self.dealer[up]
        evs = [-1e9] * 5
        evs[STAND] = stand_ev(total, dist)
        ev_hit = 0.0
        for rank in range(1, 11):
            nxt, nxt_soft = add_card(total, soft, rank)
            if nxt > 21:
                ev_hit += self.p[rank] * -1.0
            else:
                ev_hit += self.p[rank] * self.best(nxt, nxt_soft, 0, up, 0, 0, 0, 0)[0]
        evs[HIT] = ev_hit
        if can_double:
            ev_double = 0.0
            for rank in range(1, 11):
                nxt, _soft = add_card(total, soft, rank)
                if nxt > 21:
                    ev_double += self.p[rank] * -2.0
                else:
                    ev_double += self.p[rank] * (2.0 * stand_ev(nxt, dist))
            evs[DOUBLE] = ev_double
        if can_split and pair_rank and splits_left > 0:
            evs[SPLIT] = 2.0 * self._split_one(pair_rank, up, splits_left - 1)
        if can_surrender:
            evs[SURRENDER] = -0.5
        action = STAND
        value = -1e100
        tie = (0.0, 3e-15, 2e-15, 1e-15, 0.0)
        for act in (STAND, HIT, DOUBLE, SPLIT, SURRENDER):
            if evs[act] < -1e8:
                continue
            score = evs[act] + tie[act]
            if score > value:
                value = score
                action = act
        found = (evs[action], action)
        self.cache[key] = found
        return found

    def _split_one(self, rank: int, up: int, splits_left: int) -> float:
        ev = 0.0
        dist = self.dealer[up]
        for card in range(1, 11):
            total, soft = hand_value([rank, card])
            if rank == 1:
                ev += self.p[card] * stand_ev(total, dist)
            else:
                pair = rank if card == rank else 0
                can_split = 1 if pair and splits_left > 0 else 0
                ev += self.p[card] * self.best(total, soft, pair, up, 1, can_split, 0, splits_left)[0]
        return ev

    def action(self, ctx: dict) -> int:
        total = int(ctx["total"])
        if total >= 21:
            return STAND
        pair = int(ctx["pair_rank"]) if ctx["can_split"] else 0
        splits_left = int(ctx["splits_left"]) if ctx["can_split"] else 0
        _ev, action = self.best(
            total,
            int(ctx["soft"]),
            pair,
            int(ctx["up"]),
            int(ctx["can_double"]),
            int(ctx["can_split"]),
            int(ctx["can_surrender"]),
            splits_left,
        )
        return action
