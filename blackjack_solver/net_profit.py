"""Train a policy network on decisions from our own expected-value play, then bet with Hi-Lo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackjack_solver.betting import fit_tc_model
from blackjack_solver.constants import STATE_DIM
from blackjack_solver.engine import Shoe, play_hand
from blackjack_solver.exact import build
from blackjack_solver.experiment import evaluate, exact_decide, generate_snapshots, never_ins
from blackjack_solver.network import MLP, greedy_action, train_softmax
from blackjack_solver.qtable import InsuranceRule
from blackjack_solver.stats import cluster_mean, fmt_ci, fmt_p

RESULTS = Path(__file__).resolve().parents[1] / "results"


def collect(n_hands: int, seed: int, insurance: InsuranceRule | None = None):
    rng = np.random.default_rng(seed)
    shoe = Shoe(rng)
    states = []
    legal = []
    actions = []

    def grab(state, mask, ctx, action):
        states.append(state.copy())
        legal.append(mask.copy())
        actions.append(action)

    for _ in range(n_hands):
        if shoe.needs_shuffle():
            shoe.shuffle()
        result = play_hand(shoe, 1.0, exact_decide, never_ins, grab)
        if insurance is not None and result["insurance_offered"]:
            payoff = 1.0 if result["dealer_blackjack"] else -0.5
            insurance.update(result["tc_ins"], payoff)
    return (
        np.stack(states),
        np.stack(legal),
        np.asarray(actions, np.int64),
    )


def train_net(states, legal, actions, seed: int, epochs: int = 8, batch: int = 1024) -> MLP:
    rng = np.random.default_rng(seed)
    net = MLP([STATE_DIM, 128, 128, 5], rng)
    n = len(actions)
    order = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(order)
        losses = []
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            if len(idx) < 32:
                continue
            losses.append(train_softmax(net, states[idx], legal[idx], actions[idx], lr=8e-4))
        print(f"epoch {epoch + 1} loss {np.mean(losses):.4f}", flush=True)
    return net


def agreement(net: MLP, states, legal, actions) -> float:
    correct = 0
    for i in range(len(actions)):
        if greedy_action(net, states[i], legal[i]) == int(actions[i]):
            correct += 1
    return correct / max(len(actions), 1)


def main() -> None:
    build()
    RESULTS.mkdir(exist_ok=True)
    insurance = InsuranceRule()
    print("collecting decisions", flush=True)
    states, legal, actions = collect(60000, 11, insurance)
    print(f"decisions {len(actions)} insurance threshold {insurance.threshold()}", flush=True)
    net = train_net(states, legal, actions, seed=3, epochs=6)
    hold_s, hold_l, hold_a = collect(8000, 19)
    rate = agreement(net, hold_s, hold_l, hold_a)
    print(f"held-out action agreement {rate:.3%}", flush=True)

    def net_decide(state, mask, ctx):
        return greedy_action(net, state, mask)

    def ins_decide(tc: float) -> bool:
        return insurance.take(tc)

    print("calibrating Hi-Lo bets", flush=True)
    calib_snaps, calib_ids = generate_snapshots(40000, 22)
    calib = evaluate(calib_snaps, calib_ids, net_decide, ins_decide, lambda shoe: 1.0)
    model, fit = fit_tc_model(calib["tcs"], calib["profits"], calib["clusters"])
    print(
        f"profit ~ {fit['intercept']:+.5f} + {fit['slope']:+.5f} * TC  p={fmt_p(fit['p'])}",
        flush=True,
    )

    def bet_spread(shoe: Shoe) -> float:
        return model.bet(shoe.true_count(), wong=False)

    def bet_wong(shoe: Shoe) -> float:
        return model.bet(shoe.true_count(), wong=True)

    print("holdout", flush=True)
    snaps, ids = generate_snapshots(200000, 33)
    spread = evaluate(snaps, ids, net_decide, ins_decide, bet_spread)
    wong = evaluate(snaps, ids, net_decide, ins_decide, bet_wong)
    flat = evaluate(snaps, ids, net_decide, ins_decide, lambda shoe: 1.0, track_agreement=True)
    reports = {
        "flat": cluster_mean(flat["profits"], flat["clusters"]),
        "spread": cluster_mean(spread["profits"], spread["clusters"]),
        "wong": cluster_mean(wong["profits"], wong["clusters"]),
    }
    lines = [
        "NEURAL NET + HI-LO",
        f"Held-out action agreement with derived play: {rate:.3%}",
        f"Holdout agreement during flat play: {flat['agree']}/{flat['decisions']}",
        f"True-count slope {fit['slope']:+.5f} per count (p={fmt_p(fit['p'])})",
        f"Insurance threshold at true count {insurance.threshold()}",
    ]
    for name, stats in reports.items():
        lines.append(f"{name:8s} {fmt_ci(stats['mean'], stats['ci'])}  p={fmt_p(stats['p'])}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "net_profit.txt").write_text(text)
    (RESULTS / "net_profit.json").write_text(
        json.dumps(
            {
                "agreement": rate,
                "slope": fit["slope"],
                "slope_p": fit["p"],
                "flat": reports["flat"]["mean"],
                "flat_p": reports["flat"]["p"],
                "spread": reports["spread"]["mean"],
                "spread_p": reports["spread"]["p"],
                "spread_ci": list(reports["spread"]["ci"]),
                "wong": reports["wong"]["mean"],
                "wong_p": reports["wong"]["p"],
                "wong_ci": list(reports["wong"]["ci"]),
            },
            indent=2,
        )
    )
    np.savez(RESULTS / "policy_net.npz", **{f"W{i}": w for i, w in enumerate(net.W)}, **{f"b{i}": b for i, b in enumerate(net.b)})
    print(text, flush=True)


if __name__ == "__main__":
    main()
