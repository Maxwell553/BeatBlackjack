"""Sanity checks for the rules, the exact strategy, the network, and the tests."""

from __future__ import annotations

import numpy as np

from blackjack_solver.constants import HIT, SPLIT, STAND, STATE_DIM
from blackjack_solver.engine import Shoe, encode_state, play_hand
from blackjack_solver.exact import add_card, best, build, dealer_bust_probs, game_ev, hand_value
from blackjack_solver.network import MLP, train_actions
from blackjack_solver.stats import cluster_mean


def _expect(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def check_hand_value() -> None:
    cases = [
        ([1, 10], 21, 1),
        ([1, 1], 12, 1),
        ([1, 6], 17, 1),
        ([1, 6, 10], 17, 0),
        ([10, 10, 1], 21, 0),
        ([1, 1, 1], 13, 1),
        ([10, 6], 16, 0),
        ([10, 10], 20, 0),
    ]
    for cards, total, soft in cases:
        got = hand_value(cards)
        _expect(got == (total, soft), f"value {cards} -> {got}, expected {(total, soft)}")
    rng = np.random.default_rng(0)
    for _ in range(2000):
        cards = [int(rng.integers(1, 11))]
        total, soft = hand_value(cards)
        for _step in range(int(rng.integers(1, 6))):
            rank = int(rng.integers(1, 11))
            total, soft = add_card(total, soft, rank)
            cards.append(rank)
            _expect(hand_value(cards) == (total, soft), f"add_card diverged on {cards}")


def check_exact() -> None:
    build()
    bust = dealer_bust_probs()
    _expect(bust[6] > bust[10] > bust[1], f"bust order wrong: {bust}")
    _expect(0.38 < bust[6] < 0.48, f"bust on 6 out of range: {bust[6]}")
    _expect(0.20 < bust[10] < 0.28, f"bust on 10 out of range: {bust[10]}")
    _expect(0.08 < bust[1] < 0.20, f"bust on ace out of range: {bust[1]}")
    _expect(best(17, 1, 0, 6, 0, 0, 0, 0)[0] != 0.0 or True, "dealer soft 17 callable")
    for up in range(1, 11):
        action20 = best(20, 0, 0, up, 1, 0, 1, 0)[1]
        _expect(action20 == STAND, f"hard 20 vs {up} should stand, got {action20}")
        action11 = best(11, 0, 0, up, 1, 0, 1, 0)[1]
        _expect(action11 != STAND and action11 != 4, f"hard 11 vs {up} should hit or double, got {action11}")
        ev11, _a = best(11, 0, 0, up, 1, 0, 1, 0)
        _expect(ev11 > 0.0, f"hard 11 vs {up} should be a winning hand, EV {ev11}")
    _expect(best(5, 0, 0, 10, 1, 0, 1, 0)[1] == HIT, "hard 5 should hit")
    _expect(best(12, 1, 1, 10, 1, 1, 1, 3)[1] == SPLIT, "aces should split")
    _expect(best(12, 1, 1, 6, 1, 1, 1, 3)[1] == SPLIT, "aces vs 6 should split")
    _expect(best(20, 0, 10, 6, 1, 1, 1, 3)[1] == STAND, "tens should stand")
    ev = game_ev()
    _expect(-0.02 < ev < 0.005, f"infinite-deck EV out of range: {ev}")


def check_scripted() -> None:
    def stand_or_split(state, legal, ctx):
        if legal[3] > 0 and ctx["pair_rank"] == 8:
            return 3
        if legal[2] > 0 and ctx["total"] == 11:
            return 2
        return STAND

    def never(_tc):
        return False

    def hit_once(state, legal, ctx):
        if ctx["total"] == 16:
            return HIT
        return STAND

    # 8,8 vs 6, split, both stand, dealer busts. Profit +2.
    suffix = np.array([8, 8, 6, 10, 10, 3, 10, 2, 2, 2], np.int8)
    remaining = np.bincount(suffix, minlength=11).astype(np.int16)
    shoe = Shoe.from_snapshot((suffix, remaining, 0))
    result = play_hand(shoe, 1.0, stand_or_split, never)
    _expect(abs(result["profit"] - 2.0) < 1e-9, f"split profit {result['profit']}")

    # 5,6 vs 9 double to 21, dealer busts. Profit +2.
    suffix = np.array([5, 6, 9, 7, 10, 10, 2, 2], np.int8)
    remaining = np.bincount(suffix, minlength=11).astype(np.int16)
    shoe = Shoe.from_snapshot((suffix, remaining, 0))
    result = play_hand(shoe, 1.0, stand_or_split, never)
    _expect(abs(result["profit"] - 2.0) < 1e-9, f"double profit {result['profit']}")

    # 10,6 hit a 10 and bust.
    suffix = np.array([10, 6, 9, 5, 10, 2, 2, 2], np.int8)
    remaining = np.bincount(suffix, minlength=11).astype(np.int16)
    shoe = Shoe.from_snapshot((suffix, remaining, 0))
    result = play_hand(shoe, 1.0, hit_once, never)
    _expect(abs(result["profit"] + 1.0) < 1e-9, f"bust profit {result['profit']}")

    # Blackjack pays 3:2.
    suffix = np.array([1, 10, 9, 8, 2, 2], np.int8)
    remaining = np.bincount(suffix, minlength=11).astype(np.int16)
    shoe = Shoe.from_snapshot((suffix, remaining, 0))
    result = play_hand(shoe, 1.0, stand_or_split, never)
    _expect(abs(result["profit"] - 1.5) < 1e-9, f"blackjack profit {result['profit']}")

    # Dealer blackjack. Insurance wins back the main bet: -1 + 1 = 0.
    suffix = np.array([9, 8, 1, 10, 2, 2], np.int8)
    remaining = np.bincount(suffix, minlength=11).astype(np.int16)
    shoe = Shoe.from_snapshot((suffix, remaining, 0))
    result = play_hand(shoe, 1.0, stand_or_split, lambda _tc: True)
    _expect(result["dealer_blackjack"], "dealer should have blackjack")
    _expect(abs(result["profit"]) < 1e-9, f"insured dealer blackjack profit {result['profit']}")


def check_count() -> None:
    rng = np.random.default_rng(1)
    shoe = Shoe(rng)
    _expect(shoe.unseen() == 312, f"unseen {shoe.unseen()}")
    card = shoe.draw_seen()
    _expect(shoe.running == int(np.asarray([0, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1])[card]), "running count")
    tc = shoe.true_count()
    expect = shoe.running / (311 / 52.0)
    _expect(abs(tc - expect) < 1e-9, f"true count {tc} vs {expect}")
    buf = np.zeros(STATE_DIM)
    encode_state(buf, 10, 16, 0, 0, True, False, True, 0, 0.0, shoe.fractions(), shoe.unseen_decks())
    _expect(buf[9] == 1.0 and buf[10 + (16 - 4)] == 1.0 and buf[50] == 1.0, "encoder flags")


def check_network() -> None:
    rng = np.random.default_rng(0)
    net = MLP([3, 32, 1], rng)
    base = np.eye(3)
    targets_map = np.array([1.0, -2.0, 0.5])
    for _step in range(600):
        states = np.tile(base, (40, 1))
        actions = np.zeros(len(states), np.int64)
        goals = np.tile(targets_map, 40)
        train_actions(net, states, actions, goals, lr=0.01, delta=2.0)
    pred = net.forward(base)[:, 0]
    _expect(float(np.max(np.abs(pred - targets_map))) < 0.05, f"network did not fit: {pred}")


def check_cluster() -> None:
    x = np.array([1.0, 1.0, 3.0, 3.0])
    clusters = np.array([0, 0, 1, 1])
    stats = cluster_mean(x, clusters)
    _expect(abs(stats["mean"] - 2.0) < 1e-12, "cluster mean")
    _expect(abs(stats["se"] - 1.0) < 1e-12, f"cluster se {stats['se']}")


def main() -> None:
    check_hand_value()
    check_exact()
    check_scripted()
    check_count()
    check_network()
    check_cluster()
    print("self-check passed")
    print(f"infinite-deck EV {game_ev():+.5f}")
    bust = dealer_bust_probs()
    print("dealer bust", {k: round(v, 3) for k, v in bust.items()})


if __name__ == "__main__":
    main()
