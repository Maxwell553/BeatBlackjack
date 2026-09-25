"""Train the network on shoe-mix play and compare it to neutral-shoe play on the same hands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackjack_solver.composition import MixStrategy
from blackjack_solver.constants import STATE_DIM
from blackjack_solver.engine import Shoe, play_hand
from blackjack_solver.exact import build, recommend
from blackjack_solver.experiment import evaluate, exact_decide, generate_snapshots, never_ins
from blackjack_solver.network import MLP, greedy_action, train_softmax
from blackjack_solver.stats import cluster_diff, fmt_ci, fmt_p

RESULTS = Path(__file__).resolve().parents[1] / "results"


def average_mixes(n_hands: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    shoe = Shoe(rng)
    sums = {tc: np.zeros(11) for tc in range(-8, 9)}
    counts = {tc: 0 for tc in range(-8, 9)}

    def grab(state, legal, ctx, action):
        tc = int(np.clip(np.rint(ctx["tc"]), -8, 8))
        sums[tc][1:] += shoe.fractions()
        counts[tc] += 1

    for _ in range(n_hands):
        if shoe.needs_shuffle():
            shoe.shuffle()
        play_hand(shoe, 1.0, exact_decide, never_ins, grab)
    mixes = {}
    base = np.zeros(11)
    base[1:10] = 1.0 / 13.0
    base[10] = 4.0 / 13.0
    for tc, total in counts.items():
        if total < 50:
            mixes[tc] = base
        else:
            vec = np.zeros(11)
            vec[1:] = sums[tc][1:] / total
            mixes[tc] = vec
    return mixes


def build_book(mixes: dict[int, np.ndarray]) -> dict[int, MixStrategy]:
    book = {}
    for tc, probs in mixes.items():
        print(f"building mix TC {tc:+d}", flush=True)
        book[tc] = MixStrategy(probs)
    return book


class MixCache:
    def __init__(self):
        self.book: dict[tuple, MixStrategy] = {}

    def action(self, shoe: Shoe, ctx: dict, mask: np.ndarray) -> int:
        return self.action_from_fractions(shoe.fractions(), ctx, mask)

    def action_from_fractions(self, fr: np.ndarray, ctx: dict, mask: np.ndarray) -> int:
        key = tuple(np.round(fr * 20.0).astype(int))
        strat = self.book.get(key)
        if strat is None:
            probs = np.zeros(11)
            probs[1:] = fr
            strat = MixStrategy(probs)
            self.book[key] = strat
        action = strat.action(ctx)
        if mask[action] <= 0.0:
            action = int(np.argmax(mask))
        return action


def collect(cache: MixCache, n_hands: int, seed: int):
    rng = np.random.default_rng(seed)
    shoe = Shoe(rng)
    states, legal, actions = [], [], []
    differ = 0

    def decide(state, mask, ctx):
        nonlocal differ
        action = cache.action(shoe, ctx, mask)
        if action != recommend(ctx)[1]:
            differ += 1
        states.append(state.copy())
        legal.append(mask.copy())
        actions.append(action)
        return action

    for _ in range(n_hands):
        if shoe.needs_shuffle():
            shoe.shuffle()
        play_hand(shoe, 1.0, decide, never_ins)
    print(
        f"composition labels differ from neutral on {differ}/{len(actions)} "
        f"across {len(cache.book)} mixes",
        flush=True,
    )
    return np.stack(states), np.stack(legal), np.asarray(actions, np.int64)


def train_net(states, legal, actions, seed: int) -> MLP:
    rng = np.random.default_rng(seed)
    net = MLP([STATE_DIM, 128, 128, 5], rng)
    n = len(actions)
    order = np.arange(n)
    for epoch in range(16):
        rng.shuffle(order)
        losses = []
        for start in range(0, n, 1024):
            idx = order[start : start + 1024]
            if len(idx) < 32:
                continue
            losses.append(train_softmax(net, states[idx], legal[idx], actions[idx], lr=8e-4))
        print(f"epoch {epoch + 1} loss {float(np.mean(losses)):.4f}", flush=True)
    return net


def main() -> None:
    build()
    RESULTS.mkdir(exist_ok=True)
    print("collecting composition decisions", flush=True)
    cache = MixCache()
    states, legal, actions = collect(cache, 120000, 8)
    # How often the mix strategy differs from the neutral calculation.
    net = train_net(states, legal, actions, seed=4)

    def net_decide(state, mask, ctx):
        return greedy_action(net, state, mask)

    print("same holdout, flat bet, every hand", flush=True)
    snaps, ids = generate_snapshots(2000000, 99)

    def book_decide(state, mask, ctx):
        return cache.action_from_fractions(state[52:62], ctx, mask)

    book_out = evaluate(snaps, ids, book_decide, never_ins, lambda shoe: 1.0)
    base_book = evaluate(snaps, ids, exact_decide, never_ins, lambda shoe: 1.0)
    book_diff = cluster_diff(book_out["profits"], base_book["profits"], ids)
    print(
        f"teacher minus probability {fmt_ci(book_diff['mean'], book_diff['ci'])} p={fmt_p(book_diff['p'])}",
        flush=True,
    )
    net_out = evaluate(snaps, ids, net_decide, never_ins, lambda shoe: 1.0, track_agreement=True)
    base_out = evaluate(snaps, ids, exact_decide, never_ins, lambda shoe: 1.0)
    diff = cluster_diff(net_out["profits"], base_out["profits"], ids)
    lines = [
        "FLAT BET, SAME HOLDOUT, EVERY HAND",
        f"hands {len(snaps)}",
        f"network agreement with neutral play during holdout {net_out['agree']}/{net_out['decisions']}",
        f"network {fmt_ci(net_out['profits'].mean(), (float('nan'), float('nan')))}",
        f"paired network minus probability {fmt_ci(diff['mean'], diff['ci'])} z={diff['z']:.2f} p={fmt_p(diff['p'])}",
    ]
    # Replace the crude mean line with cluster means.
    from blackjack_solver.stats import cluster_mean

    net_stats = cluster_mean(net_out["profits"], ids)
    base_stats = cluster_mean(base_out["profits"], ids)
    lines = [
        "FLAT BET, SAME HOLDOUT, EVERY HAND",
        f"hands {len(snaps)}",
        f"teacher minus probability {fmt_ci(book_diff['mean'], book_diff['ci'])} p={fmt_p(book_diff['p'])}",
        f"network agreement with neutral play {net_out['agree']}/{net_out['decisions']}",
        f"network profit {fmt_ci(net_stats['mean'], net_stats['ci'])} p={fmt_p(net_stats['p'])}",
        f"probability profit {fmt_ci(base_stats['mean'], base_stats['ci'])} p={fmt_p(base_stats['p'])}",
        f"network minus probability {fmt_ci(diff['mean'], diff['ci'])} z={diff['z']:.2f} p={fmt_p(diff['p'])}",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "same_holdout.txt").write_text(text)
    (RESULTS / "same_holdout.json").write_text(
        json.dumps(
            {
                "diff_mean": diff["mean"],
                "diff_p": diff["p"],
                "diff_ci": list(diff["ci"]),
                "net_mean": net_stats["mean"],
                "base_mean": base_stats["mean"],
                "agree": net_out["agree"],
                "decisions": net_out["decisions"],
            },
            indent=2,
        )
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
