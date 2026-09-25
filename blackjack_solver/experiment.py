"""Train the solver and test its edge, its learning, and its match to exact play."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from blackjack_solver.betting import fit_tc_model
from blackjack_solver.constants import ACTION_NAMES, N_DECKS, STATE_DIM
from blackjack_solver.engine import Shoe, play_hand
from blackjack_solver.exact import build, recommend, strategy_lines
from blackjack_solver.qtable import InsuranceRule, QTable
from blackjack_solver.self_check import main as self_check
from blackjack_solver.stats import cluster_diff, cluster_mean, cluster_ratio, fmt_ci, fmt_p, ols_cluster, wilson

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def exact_decide(state, legal, ctx):
    _ev, action, _evs = recommend(ctx)
    if legal[action] <= 0.0:
        return int(np.argmax(legal))
    return int(action)


def never_ins(_tc: float) -> bool:
    return False


def generate_snapshots(n_hands: int, seed: int) -> tuple[list, np.ndarray]:
    rng = np.random.default_rng(seed)
    shoe = Shoe(rng, n_decks=N_DECKS)
    snaps: list = []
    shoe_ids = np.empty(n_hands, np.int64)
    sid = 0
    while len(snaps) < n_hands:
        if shoe.needs_shuffle():
            shoe.shuffle()
            sid += 1
        snaps.append(shoe.snapshot())
        shoe_ids[len(snaps) - 1] = sid
        play_hand(shoe, 1.0, exact_decide, never_ins)
    return snaps, shoe_ids


def insurance_greedy(rule: InsuranceRule, tc: float) -> bool:
    return rule.take(tc)


def make_decide(net: MLP, rng: np.random.Generator | None, epsilon: float):
    def decide(state, legal, ctx):
        if rng is not None and rng.random() < epsilon:
            options = np.flatnonzero(legal > 0.0)
            return int(rng.choice(options))
        return greedy_action(net, state, legal)

    return decide


def evaluate(snaps, shoe_ids, decide, decide_insurance, bet_fn, track_agreement: bool = False) -> dict:
    n = len(snaps)
    profits = np.zeros(n)
    wagers = np.zeros(n)
    tcs = np.zeros(n)
    agree = costly_agree = costly_n = neutral_n = decisions_n = 0
    gap_neutral = 0.0
    ins_n = ins_match = 0

    def on_decision(state, legal, ctx, action):
        nonlocal agree, costly_agree, costly_n, neutral_n, gap_neutral, decisions_n
        _best_ev, dp_action, evs = recommend(ctx)
        legal_ev = evs[evs > -1e8]
        best_ev = float(legal_ev.max())
        chosen = float(evs[action]) if evs[action] > -1e8 else best_ev - 1.0
        gap = best_ev - chosen
        matched = int(dp_action == action)
        decisions_n += 1
        agree += matched
        if len(legal_ev) > 1 and best_ev - float(np.partition(legal_ev, -2)[-2]) > 0.02:
            costly_n += 1
            costly_agree += matched
        if abs(ctx["tc"]) < 1.0:
            neutral_n += 1
            gap_neutral += gap

    for i, snap in enumerate(snaps):
        shoe = Shoe.from_snapshot(snap)
        tc = shoe.true_count()
        tcs[i] = tc
        bet = float(bet_fn(shoe))
        wagers[i] = bet
        if bet <= 0.0:
            continue
        result = play_hand(
            shoe,
            bet,
            decide,
            decide_insurance,
            on_decision if track_agreement else None,
        )
        profits[i] = result["profit"]
        if track_agreement and result["insurance_offered"] and result["ten_density"] is not None:
            ins_n += 1
            oracle = result["ten_density"] > (1.0 / 3.0)
            if bool(result["insurance_taken"]) == bool(oracle):
                ins_match += 1
    return {
        "profits": profits,
        "wagers": wagers,
        "tcs": tcs,
        "clusters": shoe_ids,
        "agree": agree,
        "decisions": decisions_n if track_agreement else 0,
        "costly_agree": costly_agree,
        "costly_n": costly_n,
        "neutral_n": neutral_n,
        "gap_neutral": gap_neutral,
        "ins_n": ins_n,
        "ins_match": ins_match,
    }


def flat_bet(_shoe: Shoe) -> float:
    return 1.0


def probe_line(table: QTable) -> str:
    from blackjack_solver.engine import encode_state

    fractions = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 16], dtype=np.float64) / 52.0
    specs = [
        ("H16 vs 10", 16, 0, 0, 10, True, False, True, 0),
        ("H20 vs 6", 20, 0, 0, 6, False, False, False, 0),
        ("H11 vs 6", 11, 0, 0, 6, True, False, True, 0),
        ("AA vs 10", 12, 1, 1, 10, True, True, True, 3),
        ("TT vs 6", 20, 0, 10, 6, True, True, True, 3),
    ]
    bits = []
    buf = np.empty(STATE_DIM)
    for name, total, soft, pair, up, dbl, split, surrender, splits in specs:
        encode_state(buf, up, total, soft, pair if split else 0, dbl, split, surrender, splits, 0.0, fractions, 6.0)
        legal = np.zeros(5)
        legal[0] = legal[1] = 1.0
        if dbl:
            legal[2] = 1.0
        if split:
            legal[3] = 1.0
        if surrender:
            legal[4] = 1.0
        ctx = {
            "total": total,
            "soft": soft,
            "pair_rank": pair,
            "up": up,
            "can_double": dbl,
            "can_split": split,
            "can_surrender": surrender,
            "splits_left": splits,
            "tc": 0.0,
        }
        action = table.act(ctx, legal, None, 0.0)
        _ev, dp_action, _evs = recommend(ctx)
        bits.append(f"{name}:{ACTION_NAMES[action]}/{ACTION_NAMES[dp_action]}")
    return " ".join(bits)


def train(args, curve_snaps, curve_ids):
    rng = np.random.default_rng(args.seed)
    table = QTable()
    table0 = QTable()
    ins = InsuranceRule()
    ins0 = InsuranceRule()
    shoe = Shoe(rng)
    history = []
    marks = set(int(round(args.train_hands * k / (args.checkpoints - 1))) for k in range(args.checkpoints))
    print("checkpoint 0 (untrained)", flush=True)
    history.append(eval_checkpoint(table, ins, curve_snaps, curve_ids, 0))
    print(probe_line(table), flush=True)
    for hand_i in range(1, args.train_hands + 1):
        if shoe.needs_shuffle():
            shoe.shuffle()
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * (hand_i / args.train_hands))
        result = play_hand(
            shoe,
            1.0,
            lambda state, legal, ctx, _eps=eps: table.act(ctx, legal, rng, _eps),
            lambda tc, _rule=ins, _rng=rng, _eps=eps: (
                bool(_rng.random() < 0.5) if _rng.random() < _eps else _rule.take(tc)
            ),
        )
        for tr in result["transitions"]:
            if tr["action"] == 0 and tr.get("bust"):
                target = -1.0
            elif tr["action"] == 0 and "next_ctx" in tr:
                target = table.max_q(tr["next_ctx"], tr["next_legal"])
            else:
                target = tr["G"]
            table.update(tr["ctx"], tr["action"], target)
        if result["insurance_offered"]:
            take_return = 1.0 if result["dealer_blackjack"] else -0.5
            ins.update(result["tc_ins"], take_return)
        if hand_i in marks or hand_i == args.train_hands:
            print(f"checkpoint {hand_i}  {probe_line(table)}", flush=True)
            history.append(eval_checkpoint(table, ins, curve_snaps, curve_ids, hand_i))
    return table, ins, table0, ins0, history


def eval_checkpoint(table, ins, snaps, shoe_ids, hand_i: int) -> dict:
    t0 = time.time()
    out = evaluate(
        snaps,
        shoe_ids,
        lambda state, legal, ctx: table.act(ctx, legal, None, 0.0),
        lambda tc: insurance_greedy(ins, tc),
        flat_bet,
    )
    stats = cluster_mean(out["profits"], out["clusters"])
    stats["hands_trained"] = hand_i
    stats["seconds"] = time.time() - t0
    print(
        f"  flat EV {fmt_ci(stats['mean'], stats['ci'])}  z={stats['z']:.2f} p={fmt_p(stats['p'])}  ({stats['seconds']:.1f}s)",
        flush=True,
    )
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "checkpoints.jsonl").open("a") as fh:
        fh.write(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in stats.items()}) + "\n")
    return {"hands_trained": hand_i, "profits": out["profits"], "clusters": out["clusters"], "stats": stats}


def write_svg(path: Path, history: list[dict], finals: list[tuple[str, dict]]) -> None:
    width, height = 880, 360
    pad_l, pad_r, pad_t, pad_b = 56, 20, 28, 40

    def panel(x0, y0, w, h, title, xs, ys, yerr, labels=None):
        ymin = min(y - e for y, e in zip(ys, yerr))
        ymax = max(y + e for y, e in zip(ys, yerr))
        if abs(ymax - ymin) < 1e-9:
            ymax += 1.0
            ymin -= 1.0
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad

        def px(x, i):
            return x0 + (i / max(len(xs) - 1, 1)) * w

        def py(y):
            return y0 + (ymax - y) / (ymax - ymin) * h

        parts = [f'<text x="{x0}" y="{y0 - 8}" font-size="13" font-family="sans-serif">{title}</text>']
        zero = py(0.0)
        if y0 <= zero <= y0 + h:
            parts.append(f'<line x1="{x0}" y1="{zero:.1f}" x2="{x0 + w}" y2="{zero:.1f}" stroke="#bbb"/>')
        pts = []
        for i, (y, e) in enumerate(zip(ys, yerr)):
            x = px(0, i)
            parts.append(
                f'<line x1="{x:.1f}" y1="{py(y - e):.1f}" x2="{x:.1f}" y2="{py(y + e):.1f}" stroke="#1d4e89"/>'
            )
            pts.append(f"{x:.1f},{py(y):.1f}")
        parts.append(f'<polyline fill="none" stroke="#1d4e89" stroke-width="2" points="{" ".join(pts)}"/>')
        for i, lab in enumerate(labels or []):
            parts.append(
                f'<text x="{px(0, i):.1f}" y="{y0 + h + 16}" font-size="10" text-anchor="middle" font-family="sans-serif">{lab}</text>'
            )
        return "".join(parts)

    left = panel(
        pad_l,
        pad_t,
        460,
        260,
        "Flat-bet profit per hand during training",
        list(range(len(history))),
        [h["stats"]["mean"] for h in history],
        [h["stats"]["se"] for h in history],
        [str(h["hands_trained"] // 1000) + "k" for h in history],
    )
    right = panel(
        540,
        pad_t,
        300,
        260,
        "Holdout profit per hand",
        list(range(len(finals))),
        [row[1]["mean"] for row in finals],
        [row[1]["se"] for row in finals],
        [row[0] for row in finals],
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>{left}{right}</svg>'
    )
    path.write_text(svg)


def run(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "checkpoints.jsonl").write_text("")
    print("self-check", flush=True)
    self_check()
    build()
    (RESULTS / "basic_strategy_table.txt").write_text("\n".join(strategy_lines()) + "\n")
    print("generating paired deals", flush=True)
    t0 = time.time()
    curve_snaps, curve_ids = generate_snapshots(args.curve_hands, args.seed + 11)
    calib_snaps, calib_ids = generate_snapshots(args.calib_hands, args.seed + 22)
    final_snaps, final_ids = generate_snapshots(args.final_hands, args.seed + 33)
    print(f"deals ready in {time.time() - t0:.1f}s", flush=True)

    table, ins, table0, ins0, history = train(args, curve_snaps, curve_ids)
    play_decide = lambda state, legal, ctx: table.act(ctx, legal, None, 0.0)
    untrained_decide = lambda state, legal, ctx: table0.act(ctx, legal, None, 0.0)

    print("calibrating bet ramp on held-in shoes", flush=True)
    calib = evaluate(calib_snaps, calib_ids, play_decide, lambda tc: insurance_greedy(ins, tc), flat_bet)
    model, tc_fit = fit_tc_model(calib["tcs"], calib["profits"], calib["clusters"])
    exact_calib = evaluate(calib_snaps, calib_ids, exact_decide, lambda tc: insurance_greedy(ins, tc), flat_bet)
    exact_model, exact_tc_fit = fit_tc_model(exact_calib["tcs"], exact_calib["profits"], exact_calib["clusters"])
    print(
        f"  exact play: {exact_tc_fit['intercept']:+.5f} + {exact_tc_fit['slope']:+.5f} * true count"
        f"  (SE {exact_tc_fit['se']:.5f}, p={fmt_p(exact_tc_fit['p'])})",
        flush=True,
    )
    print(
        f"  profit ~ {tc_fit['intercept']:+.5f} + {tc_fit['slope']:+.5f} * true count"
        f"  (SE {tc_fit['se']:.5f}, p={fmt_p(tc_fit['p'])})",
        flush=True,
    )

    def bet_always(shoe: Shoe) -> float:
        return model.bet(shoe.true_count(), wong=False)

    def bet_wong(shoe: Shoe) -> float:
        return model.bet(shoe.true_count(), wong=True)

    def exact_bet_always(shoe: Shoe) -> float:
        return exact_model.bet(shoe.true_count(), wong=False)

    def exact_bet_wong(shoe: Shoe) -> float:
        return exact_model.bet(shoe.true_count(), wong=True)

    print("holdout evaluation", flush=True)
    systems = {
        "untrained": evaluate(final_snaps, final_ids, untrained_decide, lambda tc: insurance_greedy(ins0, tc), flat_bet),
        "trained_flat": evaluate(
            final_snaps,
            final_ids,
            play_decide,
            lambda tc: insurance_greedy(ins, tc),
            flat_bet,
            track_agreement=True,
        ),
        "exact_flat": evaluate(final_snaps, final_ids, exact_decide, never_ins, flat_bet),
        "trained_spread": evaluate(
            final_snaps,
            final_ids,
            play_decide,
            lambda tc: insurance_greedy(ins, tc),
            bet_always,
        ),
        "trained_wong": evaluate(
            final_snaps,
            final_ids,
            play_decide,
            lambda tc: insurance_greedy(ins, tc),
            bet_wong,
        ),
        "exact_spread": evaluate(final_snaps, final_ids, exact_decide, lambda tc: insurance_greedy(ins, tc), exact_bet_always),
        "exact_wong": evaluate(final_snaps, final_ids, exact_decide, lambda tc: insurance_greedy(ins, tc), exact_bet_wong),
    }

    # Insurance threshold learned by the network, scanned across the true count.
    threshold = ins.threshold()

    comparisons = {
        "trained_minus_untrained": cluster_diff(systems["trained_flat"]["profits"], systems["untrained"]["profits"], final_ids),
        "trained_minus_exact": cluster_diff(systems["trained_flat"]["profits"], systems["exact_flat"]["profits"], final_ids),
        "spread_minus_flat": cluster_diff(systems["trained_spread"]["profits"], systems["trained_flat"]["profits"], final_ids),
        "wong_minus_flat": cluster_diff(systems["trained_wong"]["profits"], systems["trained_flat"]["profits"], final_ids),
        "trained_spread_minus_exact_spread": cluster_diff(
            systems["trained_spread"]["profits"], systems["exact_spread"]["profits"], final_ids
        ),
        "exact_wong_minus_exact_flat": cluster_diff(
            systems["exact_wong"]["profits"], systems["exact_flat"]["profits"], final_ids
        ),
        "exact_spread_minus_exact_flat": cluster_diff(
            systems["exact_spread"]["profits"], systems["exact_flat"]["profits"], final_ids
        ),
    }
    curve_x = []
    curve_y = []
    curve_c = []
    for row in history:
        curve_y.append(row["profits"])
        curve_x.append(np.full(len(row["profits"]), row["hands_trained"]))
        curve_c.append(row["clusters"].astype(np.int64) + np.int64(row["hands_trained"]) * np.int64(100000))
    trend = ols_cluster(np.concatenate(curve_y), np.concatenate(curve_x), np.concatenate(curve_c))
    before_after = cluster_diff(history[-1]["profits"], history[0]["profits"], history[0]["clusters"])

    edges = {}
    for name, out in systems.items():
        edges[name] = {
            "per_hand": cluster_mean(out["profits"], out["clusters"]),
            "per_wager": cluster_ratio(out["profits"], out["wagers"], out["clusters"]),
            "mean_wager": float(out["wagers"].mean()),
            "hands_bet": int(np.sum(out["wagers"] > 0)),
        }

    agree_n = systems["trained_flat"]["decisions"]
    agree_k = systems["trained_flat"]["agree"]
    phat, w_lo, w_hi = wilson(agree_k, agree_n)
    costly_n = systems["trained_flat"]["costly_n"]
    costly_k = systems["trained_flat"]["costly_agree"]
    c_hat, c_lo, c_hi = wilson(costly_k, costly_n)
    ins_n = systems["trained_flat"]["ins_n"]
    ins_k = systems["trained_flat"]["ins_match"]
    i_hat, i_lo, i_hi = wilson(ins_k, ins_n)
    neutral_n = max(systems["trained_flat"]["neutral_n"], 1)
    ev_loss = systems["trained_flat"]["gap_neutral"] / neutral_n

    def pack(stats: dict) -> dict:
        return {
            "mean": stats.get("mean", stats.get("ratio", stats.get("slope"))),
            "se": stats["se"],
            "z": stats["z"],
            "p": stats["p"],
            "ci": list(stats["ci"]),
            "n": stats["n"],
            "clusters": stats["clusters"],
        }

    metrics = {
        "rules": "6 decks, 75% penetration, S17, 3:2, DAS, late surrender, split to 4, no hit/resplit aces",
        "tc_regression": tc_fit,
        "trend_per_training_hand": trend,
        "curve_final_minus_start": before_after,
        "edges": {k: {"per_hand": pack(v["per_hand"]), "per_wager": pack(v["per_wager"]), "mean_wager": v["mean_wager"], "hands_bet": v["hands_bet"]} for k, v in edges.items()},
        "comparisons": {k: pack(v) for k, v in comparisons.items()},
        "agreement": {"rate": phat, "ci": [w_lo, w_hi], "k": agree_k, "n": agree_n},
        "decisive_agreement": {"rate": c_hat, "ci": [c_lo, c_hi], "k": costly_k, "n": costly_n},
        "insurance_agreement": {"rate": i_hat, "ci": [i_lo, i_hi], "k": ins_k, "n": ins_n, "learned_tc_threshold": threshold},
        "neutral_ev_loss": ev_loss,
        "bet_model": {"intercept": model.intercept, "slope": model.slope, "variance": model.variance},
    }
    # JSON cannot hold raw numpy types from tc_fit.
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        return obj

    (RESULTS / "metrics.json").write_text(json.dumps(convert(metrics), indent=2))
    np.savez(
        RESULTS / "solver.npz",
        q_sum=table.q_sum,
        q_n=table.q_n,
        insurance_sum=ins.sum,
        insurance_n=ins.n,
    )

    lines = []
    lines.append("BLACKJACK SOLVER REPORT")
    lines.append(metrics["rules"])
    lines.append("Play is one-step Q-learning on the hand and the dealer upcard. A hit bootstraps the best next action; stand, double, split, and surrender use the realized return.")
    lines.append("Insurance is taken at a true count only when the average insurance return in that count bin is positive.")
    lines.append("Bet size is full Kelly on a bankroll of 800 units, capped at 12, from a linear fit of profit on the pre-deal true count. The fit is frozen before the holdout.")
    lines.append("The exact column is infinite-deck backward induction under the same rules. It is an evaluator, not a training label.")
    lines.append("Standard errors are cluster-robust by shoe. Tests are two-sided normal approximations, which the hand counts support.")
    lines.append(
        f"Samples: {args.train_hands} training hands, {args.calib_hands} calibration hands, {args.final_hands} holdout hands."
    )
    lines.append("")
    lines.append("Learning curve (flat bet, same deals at every checkpoint)")
    for row in history:
        st = row["stats"]
        lines.append(f"  after {row['hands_trained']:6d} hands   {fmt_ci(st['mean'], st['ci'])}   p vs 0 = {fmt_p(st['p'])}")
    lines.append(
        f"Slope of profit on hands trained: {trend['slope']:+.3e} per hand  "
        f"(95% CI {trend['ci'][0]:+.3e} to {trend['ci'][1]:+.3e}, p={fmt_p(trend['p'])})"
    )
    lines.append(
        f"Last checkpoint minus first, paired: {fmt_ci(before_after['mean'], before_after['ci'])}  p={fmt_p(before_after['p'])}"
    )
    lines.append("")
    lines.append("Holdout edges")
    for name, edge in edges.items():
        ph = edge["per_hand"]
        pw = edge["per_wager"]
        lines.append(
            f"  {name:22s}  per hand {fmt_ci(ph['mean'], ph['ci'])} p={fmt_p(ph['p'])}"
            f"   per unit posted {fmt_ci(pw['ratio'], pw['ci'], 5)} p={fmt_p(pw['p'])}"
            f"   mean bet {edge['mean_wager']:.2f}"
        )
    lines.append("")
    lines.append("Paired comparisons on the holdout deals")
    labels = {
        "trained_minus_untrained": "Trained flat minus untrained flat",
        "trained_minus_exact": "Trained flat minus exact flat",
        "spread_minus_flat": "Count spread minus trained flat",
        "wong_minus_flat": "Sit-out spread minus trained flat",
        "trained_spread_minus_exact_spread": "Trained spread minus exact spread",
        "exact_wong_minus_exact_flat": "Exact play with sit-out bets minus exact flat",
        "exact_spread_minus_exact_flat": "Exact play with count spread minus exact flat",
    }
    for key, label in labels.items():
        st = comparisons[key]
        lines.append(f"  {label}: {fmt_ci(st['mean'], st['ci'])}  z={st['z']:.2f}  p={fmt_p(st['p'])}")
    lines.append("")
    lines.append(
        f"True-count slope on calibration (exact play): {exact_tc_fit['slope']:+.5f} per true count"
        f"  (95% CI {exact_tc_fit['ci'][0]:+.5f} to {exact_tc_fit['ci'][1]:+.5f}, p={fmt_p(exact_tc_fit['p'])})"
    )
    lines.append(
        f"True-count slope on calibration (flat trained play): {tc_fit['slope']:+.5f} per true count"
        f"  (95% CI {tc_fit['ci'][0]:+.5f} to {tc_fit['ci'][1]:+.5f}, p={fmt_p(tc_fit['p'])})"
    )
    lines.append(
        f"Action agreement with exact strategy: {agree_k}/{agree_n} = {phat:.3%}  (Wilson 95% CI {w_lo:.3%} to {w_hi:.3%})"
    )
    lines.append(
        f"Agreement where the best action leads by more than 0.02 EV: {costly_k}/{costly_n} = {c_hat:.3%}"
        f"  (Wilson 95% CI {c_lo:.3%} to {c_hi:.3%})"
    )
    lines.append(f"Mean exact-EV loss per decision at |true count| < 1: {ev_loss:.5f}")
    lines.append(
        f"Insurance agreement with ten-density > 1/3: {ins_k}/{ins_n} = {i_hat:.3%}"
        f"  (Wilson 95% CI {i_lo:.3%} to {i_hi:.3%}); learned take-threshold at true count {threshold}"
    )
    lines.append("")
    lines.append("Reading the result: a flat basic-strategy bettor is supposed to lose a fraction of a percent.")
    lines.append("A counter's edge shows up as a positive slope of profit on the true count, and as a higher profit when the bet spread is allowed to use that slope.")
    lines.append("Matching the exact strategy means the paired gap versus exact play is near zero and decisive-action agreement is high.")
    text = "\n".join(lines) + "\n"
    (RESULTS / "report.txt").write_text(text)
    write_svg(
        RESULTS / "learning_curve.svg",
        history,
        [
            ("untr", edges["untrained"]["per_hand"]),
            ("flat", edges["trained_flat"]["per_hand"]),
            ("exact", edges["exact_flat"]["per_hand"]),
            ("spread", edges["trained_spread"]["per_hand"]),
            ("wong", edges["trained_wong"]["per_hand"]),
        ],
    )
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hands", type=int, default=80000)
    parser.add_argument("--curve-hands", type=int, default=8000)
    parser.add_argument("--calib-hands", type=int, default=20000)
    parser.add_argument("--final-hands", type=int, default=40000)
    parser.add_argument("--checkpoints", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--replay", type=int, default=100000)
    parser.add_argument("--ins-replay", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--ins-lr", type=float, default=2e-3)
    parser.add_argument("--eps-end", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.train_hands = 2000
        args.curve_hands = 400
        args.calib_hands = 400
        args.final_hands = 400
        args.checkpoints = 3
        args.updates = 1
    run(args)


if __name__ == "__main__":
    main()
