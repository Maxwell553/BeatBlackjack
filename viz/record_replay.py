"""Record the same shoes played by the untrained bot and the trained bot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackjack_solver.constants import STAND
from blackjack_solver.engine import Shoe, hand_value, play_hand
from blackjack_solver.exact import build
from blackjack_solver.experiment import generate_snapshots
from blackjack_solver.internal import Solver, gather, play_decide, train_insurance, train_play, train_value
from blackjack_solver.network import MLP, greedy_action

OUT = Path(__file__).resolve().parent / "replay.json"
NAMES = {0: "hit", 1: "stand", 2: "double", 3: "split", 4: "surrender"}


class Tap:
    def __init__(self, shoe: Shoe):
        self.events: list[tuple] = []
        self._seen = shoe.draw_seen
        self._hidden = shoe.draw_hidden
        self._reveal = shoe.reveal_hidden
        shoe.draw_seen = self.seen
        shoe.draw_hidden = self.hidden
        shoe.reveal_hidden = self.reveal

    def seen(self):
        card = int(self._seen())
        self.events.append(("card", card))
        return card

    def hidden(self):
        card = int(self._hidden())
        self.events.append(("hole", card))
        return card

    def reveal(self):
        card = int(self._reveal())
        self.events.append(("reveal", card))
        return card


def _take(events, kind):
    item = events.pop(0)
    if item[0] != kind:
        raise RuntimeError(f"expected {kind}, got {item}")
    return item


def trace_round(snap, bet: float, decide, insure, edge: float) -> dict:
    shoe = Shoe.from_snapshot(snap)
    tap = Tap(shoe)

    def logged(state, legal, ctx):
        action = int(decide(state, legal, ctx))
        if legal[action] <= 0.0:
            action = STAND
        tap.events.append(("action", NAMES[action]))
        return action

    def logged_ins(tc, live):
        taken = bool(insure(tc, live))
        tap.events.append(("insurance", taken))
        return taken

    result = play_hand(shoe, bet, logged, logged_ins)
    events = list(tap.events)
    player = [_take(events, "card")[1], _take(events, "card")[1]]
    up = _take(events, "card")[1]
    insurance = False
    if up == 1 and events and events[0][0] == "insurance":
        insurance = bool(events.pop(0)[1])
    hole = _take(events, "hole")[1]
    steps = []
    hands = [{"cards": player[:], "bet": float(bet), "splits_left": 3, "aces": False, "done": False}]
    early = result["player_blackjack"] or result["dealer_blackjack"]
    if not early:
        i = 0
        while i < len(hands):
            hand = hands[i]
            while not hand["done"]:
                total, _soft = hand_value(hand["cards"])
                if total >= 21:
                    hand["done"] = True
                    break
                action = _take(events, "action")[1]
                step = {"op": action, "hand": i}
                if action in ("hit", "double"):
                    card = _take(events, "card")[1]
                    hand["cards"].append(card)
                    step["card"] = card
                    if action == "double":
                        hand["bet"] *= 2.0
                        hand["done"] = True
                    elif hand_value(hand["cards"])[0] > 21:
                        hand["done"] = True
                elif action == "split":
                    left = _take(events, "card")[1]
                    right = _take(events, "card")[1]
                    rank = hand["cards"][0]
                    step["cards"] = [left, right]
                    child = 0 if rank == 1 else hand["splits_left"] - 1
                    nxt_l = {"cards": [rank, left], "bet": hand["bet"], "splits_left": child, "aces": rank == 1, "done": rank == 1}
                    nxt_r = {"cards": [rank, right], "bet": hand["bet"], "splits_left": child, "aces": rank == 1, "done": rank == 1}
                    hands[i] = nxt_l
                    hands.insert(i + 1, nxt_r)
                    hand = hands[i]
                else:
                    hand["done"] = True
                steps.append(step)
            i += 1
        _take(events, "reveal")
        dealer = [up, hole]
        while events and events[0][0] == "card":
            card = _take(events, "card")[1]
            dealer.append(card)
            steps.append({"op": "dealer", "card": card})
    else:
        if events and events[0][0] == "reveal":
            _take(events, "reveal")
        dealer = [up, hole]
    label = _label(result, hands, dealer)
    return {
        "bet": float(bet),
        "edge": float(edge),
        "insurance": insurance,
        "player": player,
        "dealerUp": up,
        "hole": hole,
        "steps": steps,
        "dealer": dealer,
        "finalHands": [{"cards": h["cards"], "bet": h["bet"]} for h in hands],
        "profit": float(result["profit"]),
        "label": label,
    }


def _label(result, hands, dealer) -> str:
    if result["player_blackjack"] and result["dealer_blackjack"]:
        return "Push"
    if result["player_blackjack"]:
        return "Blackjack"
    if result["dealer_blackjack"]:
        return "Dealer blackjack"
    if len(hands) == 1 and len(hands[0]["cards"]) == 2 and abs(result["profit"] + 0.5 * hands[0]["bet"]) < 1e-6:
        return "Surrender"
    total, _ = hand_value(dealer)
    if all(hand_value(h["cards"])[0] > 21 for h in hands):
        return "Bust"
    if total > 21:
        return "Dealer bust"
    if result["profit"] > 1e-9:
        return "Win"
    if result["profit"] < -1e-9:
        return "Lose"
    return "Push"


def main() -> None:
    build()
    print("training the after bot", flush=True)
    play = train_play(8)
    features, returns, ins_x, ins_y = gather(play, 120000, 11)
    value, variance = train_value(features, returns, 5)
    insurance = train_insurance(ins_x, ins_y, 6)
    solver = Solver(play, value, insurance, variance, fraction=1.0, max_bet=22.2)
    after_decide = play_decide(play)
    rng = np.random.default_rng(3)
    before_net = MLP([64, 128, 128, 5], rng)

    def before_decide(state, legal, ctx):
        state = state.copy()
        state[51] = 0.0
        return greedy_action(before_net, state, legal)

    def never(_tc, _shoe=None):
        return False

    snaps, _ids = generate_snapshots(360, 99)
    pairs = []
    for snap in snaps:
        shoe = Shoe.from_snapshot(snap)
        edge = solver.edge(shoe)
        bet = solver.stake(shoe)
        after = trace_round(snap, bet, after_decide, solver.take_insurance, edge)
        before = trace_round(snap, 1.0, before_decide, never, 0.0)
        pairs.append({"before": before, "after": after})
    best_i, best_score = 0, -1e9
    width = 8
    for i in range(0, len(pairs) - width):
        window = pairs[i : i + width]
        raised = sum(1 for p in window if p["after"]["bet"] >= 3.0)
        plays = sum(len(p["after"]["steps"]) for p in window)
        score = raised * 5 + min(plays, 24) * 0.1
        if score > best_score:
            best_score = score
            best_i = i
    chosen = pairs[best_i : best_i + width]
    OUT.write_text(json.dumps({"startBank": 800, "pairs": chosen}))
    bets = [round(p["after"]["bet"], 1) for p in chosen]
    print(f"wrote {OUT} hands {best_i}-{best_i + width - 1} bets {bets}", flush=True)


if __name__ == "__main__":
    main()
