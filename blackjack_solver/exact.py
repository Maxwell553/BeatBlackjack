"""Infinite-deck optimal basic strategy by backward induction.

Card probabilities are the infinite-shoe limits (ten-value cards 4/13). The
recursion is derived from the rules in this project; it is not a copied chart.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from blackjack_solver.constants import DOUBLE, HIT, SPLIT, STAND, SURRENDER

P = np.zeros(11)
P[1] = 1.0 / 13.0
for _rank in range(2, 10):
    P[_rank] = 1.0 / 13.0
P[10] = 4.0 / 13.0

_DEALER: list[np.ndarray] = []
_P_BJ: list[float] = []
_READY = False


def hand_value(cards: list[int]) -> tuple[int, int]:
    total = 0
    aces = 0
    for card in cards:
        if card == 1:
            aces += 1
            total += 1
        else:
            total += card
    soft = 0
    if aces and total + 10 <= 21:
        total += 10
        soft = 1
    return total, soft


def add_card(total: int, soft: int, rank: int) -> tuple[int, int]:
    if rank == 1:
        if soft:
            if total + 1 <= 21:
                return total + 1, 1
            return total + 1 - 10, 0
        if total + 11 <= 21:
            return total + 11, 1
        return total + 1, 0
    if soft:
        if total + rank <= 21:
            return total + rank, 1
        return total + rank - 10, 0
    return total + rank, 0


@lru_cache(maxsize=None)
def _dealer_from(total: int, soft: int) -> tuple[float, ...]:
    if total > 21:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if total >= 17:
        dist = [0.0] * 6
        dist[total - 16] = 1.0
        return tuple(dist)
    acc = [0.0] * 6
    for rank in range(1, 11):
        nxt, nxt_soft = add_card(total, soft, rank)
        sub = _dealer_from(nxt, nxt_soft)
        prob = float(P[rank])
        for k in range(6):
            acc[k] += prob * sub[k]
    return tuple(acc)


def stand_ev(player_total: int, dist: np.ndarray) -> float:
    if player_total > 21:
        return -1.0
    ev = float(dist[0])
    for offset, total in enumerate(range(17, 22)):
        prob = float(dist[offset + 1])
        if player_total > total:
            ev += prob
        elif player_total < total:
            ev -= prob
    return ev


def _dealer_given_upcard(up: int) -> tuple[float, np.ndarray]:
    p_bj = float(P[10] if up == 1 else (P[1] if up == 10 else 0.0))
    acc = np.zeros(6)
    p_play = 0.0
    for hole in range(1, 11):
        if (up == 1 and hole == 10) or (up == 10 and hole == 1):
            continue
        total, soft = hand_value([up, hole])
        acc += P[hole] * np.asarray(_dealer_from(total, soft))
        p_play += float(P[hole])
    acc /= p_play
    return p_bj, acc


@lru_cache(maxsize=None)
def _split_one(rank: int, up: int, splits_left: int) -> float:
    ev = 0.0
    dist = _DEALER[up]
    for card in range(1, 11):
        total, soft = hand_value([rank, card])
        if rank == 1:
            ev += float(P[card]) * stand_ev(total, dist)
        else:
            pair = rank if card == rank else 0
            can_split = 1 if pair and splits_left > 0 else 0
            ev += float(P[card]) * best(total, soft, pair, up, 1, can_split, 0, splits_left)[0]
    return ev


@lru_cache(maxsize=None)
def best(
    total: int,
    soft: int,
    pair_rank: int,
    up: int,
    can_double: int,
    can_split: int,
    can_surrender: int,
    splits_left: int,
) -> tuple[float, int]:
    """Return (best expected profit per unit bet, action)."""
    dist = _DEALER[up]
    evs = [0.0] * 5
    legal = [False] * 5
    evs[STAND] = stand_ev(total, dist)
    legal[STAND] = True

    ev_hit = 0.0
    for rank in range(1, 11):
        nxt, nxt_soft = add_card(total, soft, rank)
        if nxt > 21:
            ev_hit += float(P[rank]) * -1.0
        else:
            ev_hit += float(P[rank]) * best(nxt, nxt_soft, 0, up, 0, 0, 0, 0)[0]
    evs[HIT] = ev_hit
    legal[HIT] = True

    if can_double:
        ev_double = 0.0
        for rank in range(1, 11):
            nxt, _soft = add_card(total, soft, rank)
            if nxt > 21:
                ev_double += float(P[rank]) * -2.0
            else:
                ev_double += float(P[rank]) * (2.0 * stand_ev(nxt, dist))
        evs[DOUBLE] = ev_double
        legal[DOUBLE] = True

    if can_split and pair_rank and splits_left > 0:
        evs[SPLIT] = 2.0 * _split_one(pair_rank, up, splits_left - 1)
        legal[SPLIT] = True

    if can_surrender:
        evs[SURRENDER] = -0.5
        legal[SURRENDER] = True

    # Tiny preference for standing when two actions tie to the last bit.
    tie = (0.0, 3e-15, 2e-15, 1e-15, 0.0)
    best_i = STAND
    best_v = -1e100
    for action in (STAND, HIT, DOUBLE, SPLIT, SURRENDER):
        if not legal[action]:
            continue
        score = evs[action] + tie[action]
        if score > best_v:
            best_v = score
            best_i = action
    return evs[best_i], best_i


def action_evs(
    total: int,
    soft: int,
    pair_rank: int,
    up: int,
    can_double: int,
    can_split: int,
    can_surrender: int,
    splits_left: int,
) -> tuple[float, int, np.ndarray]:
    """Best EV, best action, and every action's EV (-1e9 if illegal)."""
    dist = _DEALER[up]
    evs = np.full(5, -1e9)
    evs[STAND] = stand_ev(total, dist)
    ev_hit = 0.0
    for rank in range(1, 11):
        nxt, nxt_soft = add_card(total, soft, rank)
        if nxt > 21:
            ev_hit += float(P[rank]) * -1.0
        else:
            ev_hit += float(P[rank]) * best(nxt, nxt_soft, 0, up, 0, 0, 0, 0)[0]
    evs[HIT] = ev_hit
    if can_double:
        ev_double = 0.0
        for rank in range(1, 11):
            nxt, _soft = add_card(total, soft, rank)
            if nxt > 21:
                ev_double += float(P[rank]) * -2.0
            else:
                ev_double += float(P[rank]) * (2.0 * stand_ev(nxt, dist))
        evs[DOUBLE] = ev_double
    if can_split and pair_rank and splits_left > 0:
        evs[SPLIT] = 2.0 * _split_one(pair_rank, up, splits_left - 1)
    if can_surrender:
        evs[SURRENDER] = -0.5
    tie = np.array([0.0, 3e-15, 2e-15, 1e-15, 0.0])
    action = int(np.argmax(np.where(evs > -1e8, evs + tie, evs)))
    return float(evs[action]), action, evs


def recommend(ctx: dict) -> tuple[int, float, np.ndarray]:
    total = int(ctx["total"])
    if total >= 21:
        evs = np.full(5, -1e9)
        evs[STAND] = stand_ev(total, _DEALER[int(ctx["up"])])
        return STAND, float(evs[STAND]), evs
    pair = int(ctx["pair_rank"]) if ctx["can_split"] else 0
    splits_left = int(ctx["splits_left"]) if ctx["can_split"] else 0
    return action_evs(
        total,
        int(ctx["soft"]),
        pair,
        int(ctx["up"]),
        int(ctx["can_double"]),
        int(ctx["can_split"]),
        int(ctx["can_surrender"]),
        splits_left,
    )


def game_ev() -> float:
    """Expected profit of optimal play, infinite deck, no insurance, bet = 1."""
    ev = 0.0
    for c1 in range(1, 11):
        for c2 in range(1, 11):
            for up in range(1, 11):
                prob = float(P[c1] * P[c2] * P[up])
                player_bj = (c1 == 1 and c2 == 10) or (c2 == 1 and c1 == 10)
                p_bj = _P_BJ[up]
                if player_bj:
                    ev += prob * ((1.0 - p_bj) * 1.5)
                else:
                    total, soft = hand_value([c1, c2])
                    pair = c1 if c1 == c2 else 0
                    play, _action = best(
                        total,
                        soft,
                        pair,
                        up,
                        1,
                        1 if pair else 0,
                        1,
                        3 if pair else 0,
                    )
                    ev += prob * (p_bj * -1.0 + (1.0 - p_bj) * play)
    return ev


def dealer_bust_probs() -> dict[int, float]:
    return {up: float(_DEALER[up][0]) for up in range(1, 11)}


def build() -> None:
    global _READY
    if _READY:
        return
    _DEALER.clear()
    _P_BJ.clear()
    _DEALER.append(np.zeros(6))
    _P_BJ.append(0.0)
    for up in range(1, 11):
        p_bj, dist = _dealer_given_upcard(up)
        if abs(float(dist.sum()) - 1.0) > 1e-8:
            raise RuntimeError(f"dealer distribution for {up} sums to {dist.sum()}")
        _DEALER.append(dist)
        _P_BJ.append(p_bj)
    _READY = True


def strategy_lines() -> list[str]:
    """Text table of the computed infinite-deck strategy."""
    from blackjack_solver.constants import ACTION_NAMES

    lines = ["Infinite-deck strategy computed by backward induction (S17, DAS, late surrender, split to 4, no resplit aces)."]
    ups = list(range(2, 11)) + [1]
    header = "Hand".ljust(8) + "".join(f"{'A' if u == 1 else str(u):>6}" for u in ups)
    lines.append(header)
    lines.append("Hard")
    for total in range(5, 21):
        row = f"{total:<8}"
        for up in ups:
            _ev, action = best(total, 0, 0, up, 1, 0, 1, 0)
            row += f"{ACTION_NAMES[action][:5]:>6}"
        lines.append(row)
    lines.append("Soft")
    for total in range(13, 21):
        row = f"A,{total - 11:<5}"
        for up in ups:
            _ev, action = best(total, 1, 0, up, 1, 0, 1, 0)
            row += f"{ACTION_NAMES[action][:5]:>6}"
        lines.append(row)
    lines.append("Pairs")
    for rank in list(range(2, 11)) + [1]:
        label = "A,A" if rank == 1 else ("T,T" if rank == 10 else f"{rank},{rank}")
        total, soft = hand_value([rank, rank])
        row = f"{label:<8}"
        for up in ups:
            _ev, action = best(total, soft, rank, up, 1, 1, 1, 3)
            row += f"{ACTION_NAMES[action][:5]:>6}"
        lines.append(row)
    lines.append(f"Infinite-deck game EV (no insurance): {game_ev():+.6f} per unit bet")
    bust = dealer_bust_probs()
    lines.append(
        "Dealer bust probabilities: "
        + ", ".join(f"{'A' if u == 1 else u}={bust[u]:.3f}" for u in ups)
    )
    return lines
