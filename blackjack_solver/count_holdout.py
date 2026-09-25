"""Same paired holdout as the flat comparison, with one shared Hi-Lo bet schedule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackjack_solver.betting import fit_tc_model
from blackjack_solver.exact import build
from blackjack_solver.experiment import evaluate, exact_decide, generate_snapshots, never_ins
from blackjack_solver.network import greedy_action
from blackjack_solver.same_holdout import MixCache, collect, train_net
from blackjack_solver.stats import cluster_diff, cluster_mean, fmt_ci, fmt_p

RESULTS = Path(__file__).resolve().parents[1] / "results"


def count_ins(tc: float) -> bool:
    return float(tc) >= 3.0


def main() -> None:
    build()
    RESULTS.mkdir(exist_ok=True)
    print("collecting composition decisions", flush=True)
    cache = MixCache()
    states, legal, actions = collect(cache, 120000, 8)
    net = train_net(states, legal, actions, seed=4)

    def net_decide(state, mask, ctx):
        return greedy_action(net, state, mask)

    print("fitting one shared count bet on calibration shoes", flush=True)
    calib_snaps, calib_ids = generate_snapshots(100000, 123)
    calib = evaluate(calib_snaps, calib_ids, exact_decide, never_ins, lambda shoe: 1.0)
    model, fit = fit_tc_model(calib["tcs"], calib["profits"], calib["clusters"])
    print(
        f"shared edge {fit['intercept']:+.5f} {fit['slope']:+.5f} per true count, variance {fit['variance']:.3f}",
        flush=True,
    )

    def shared_bet(shoe):
        return model.bet(shoe.true_count(), wong=False)

    print("same holdout, shared count bets, every hand", flush=True)
    snaps, ids = generate_snapshots(2000000, 99)
    net_out = evaluate(snaps, ids, net_decide, count_ins, shared_bet, track_agreement=True)
    base_out = evaluate(snaps, ids, exact_decide, count_ins, shared_bet)
    diff = cluster_diff(net_out["profits"], base_out["profits"], ids)
    net_stats = cluster_mean(net_out["profits"], ids)
    base_stats = cluster_mean(base_out["profits"], ids)
    lines = [
        "SHARED HI-LO BETS, SAME HOLDOUT, EVERY HAND",
        f"hands {len(snaps)}",
        f"bet edge {fit['intercept']:+.5f} {fit['slope']:+.5f} per true count, variance {fit['variance']:.3f}, bankroll {model.bankroll:.0f}, max bet {model.max_bet:.0f}",
        f"mean bet {float(net_out['wagers'].mean()):.3f}",
        "insurance taken by both at true count >= 3",
        f"network agreement with neutral play {net_out['agree']}/{net_out['decisions']}",
        f"network profit {fmt_ci(net_stats['mean'], net_stats['ci'])} p={fmt_p(net_stats['p'])}",
        f"probability profit {fmt_ci(base_stats['mean'], base_stats['ci'])} p={fmt_p(base_stats['p'])}",
        f"network minus probability {fmt_ci(diff['mean'], diff['ci'])} z={diff['z']:.2f} p={fmt_p(diff['p'])}",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "count_holdout.txt").write_text(text)
    (RESULTS / "count_holdout.json").write_text(
        json.dumps(
            {
                "diff_mean": diff["mean"],
                "diff_p": diff["p"],
                "diff_ci": list(diff["ci"]),
                "net_mean": net_stats["mean"],
                "base_mean": base_stats["mean"],
                "mean_bet": float(net_out["wagers"].mean()),
                "agree": net_out["agree"],
                "decisions": net_out["decisions"],
                "intercept": fit["intercept"],
                "slope": fit["slope"],
                "variance": fit["variance"],
            },
            indent=2,
        )
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
