"""Play, bet, and insurance from networks that see the remaining cards and no count statistic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blackjack_solver.constants import STATE_DIM
from blackjack_solver.engine import Shoe, play_hand
from blackjack_solver.exact import build
from blackjack_solver.betting import fit_tc_model
from blackjack_solver.experiment import evaluate, exact_decide, generate_snapshots, never_ins
from blackjack_solver.network import MLP, greedy_action, train_softmax
from blackjack_solver.same_holdout import MixCache, collect
from blackjack_solver.stats import cluster_diff, cluster_mean, fmt_ci, fmt_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
BANKROLL = 800.0
MAX_BET = 12.0


def shoe_features(shoe: Shoe) -> np.ndarray:
    features = np.empty(11, np.float64)
    features[:10] = shoe.fractions()
    features[10] = shoe.unseen_decks() / 6.0
    return features


def train_play(seed: int) -> MLP:
    cache = MixCache()
    states, legal, actions = collect(cache, 120000, seed)
    states[:, 51] = 0.0
    rng = np.random.default_rng(seed)
    net = MLP([STATE_DIM, 128, 128, 5], rng)
    order = np.arange(len(actions))
    for epoch in range(12):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(actions), 1024):
            idx = order[start : start + 1024]
            if len(idx) < 32:
                continue
            losses.append(train_softmax(net, states[idx], legal[idx], actions[idx], lr=8e-4))
        print(f"play epoch {epoch + 1} loss {float(np.mean(losses)):.4f}", flush=True)
    return net


def play_decide(net: MLP):
    def decide(state, legal, ctx):
        state = state.copy()
        state[51] = 0.0
        return greedy_action(net, state, legal)

    return decide


def gather(play, n_hands: int, seed: int):
    rng = np.random.default_rng(seed)
    shoe = Shoe(rng)
    decide = play_decide(play)
    shoe_rows = []
    returns = []
    ins_rows = []
    ins_labels = []
    held: list[np.ndarray] = []

    def watch(_tc, live: Shoe):
        held.append(shoe_features(live))
        return False

    for _ in range(n_hands):
        if shoe.needs_shuffle():
            shoe.shuffle()
        if len(shoe_rows) and len(shoe_rows) % 250000 == 0:
            print(f"gathered {len(shoe_rows)}", flush=True)
        shoe_rows.append(shoe_features(shoe))
        held.clear()
        result = play_hand(shoe, 1.0, decide, watch)
        returns.append(result["profit"])
        if result["insurance_offered"] and held:
            ins_rows.append(held[-1])
            ins_labels.append(1.0 if result["dealer_blackjack"] else 0.0)
    print(f"gathered {len(returns)} hands, {len(ins_labels)} insurance spots", flush=True)
    return (
        np.stack(shoe_rows),
        np.asarray(returns, np.float64),
        np.stack(ins_rows),
        np.asarray(ins_labels, np.float64),
    )


def _linear_fit(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least squares on composition. One fraction is omitted because the ten shares sum to one."""
    columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    design = np.column_stack([features[:, columns], np.ones(len(target))])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    weights = np.zeros(11, np.float64)
    weights[columns] = beta[:-1]
    pred = design @ beta
    return weights, float(beta[-1]), pred


def _mse_step(net: MLP, features: np.ndarray, target: np.ndarray, lr: float) -> float:
    pred = net.forward(features)
    if not np.isfinite(pred).all():
        return float("inf")
    err = pred - target
    net.backward(err / len(features), lr=lr, clip=1.0)
    return float(np.mean(err * err))


def _fit_residual(features: np.ndarray, target: np.ndarray, seed: int, label: str) -> MLP:
    """Small MLP that starts at zero and is kept only when it lowers validation error."""
    weights, bias, linear = _linear_fit(features, target)
    rng = np.random.default_rng(seed)
    extra = MLP([11, 32, 32, 1], rng)
    extra.W[-1][:] = rng.normal(0.0, 1e-3, size=extra.W[-1].shape)
    extra.b[-1].fill(0.0)
    n = len(target)
    val = max(4096, n // 10)
    order = rng.permutation(n)
    val_i, train_i = order[:val], order[val:]
    y = (target - linear).reshape(-1, 1)
    best = [(w.copy(), b.copy()) for w, b in zip(extra.W, extra.b)]
    best_val = float(np.mean((target[val_i] - linear[val_i]) ** 2))
    stale = 0
    for epoch in range(12):
        rng.shuffle(train_i)
        losses = []
        for start in range(0, len(train_i), 4096):
            idx = train_i[start : start + 4096]
            if len(idx) < 64:
                continue
            losses.append(_mse_step(extra, features[idx], y[idx], lr=1e-3))
        pred = linear + extra.forward(features)[:, 0]
        val_loss = float(np.mean((target[val_i] - pred[val_i]) ** 2))
        print(
            f"{label} epoch {epoch + 1} train {float(np.mean(losses)):.5f} val {val_loss:.5f}",
            flush=True,
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best = [(w.copy(), b.copy()) for w, b in zip(extra.W, extra.b)]
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    for i, (w, b) in enumerate(best):
        extra.W[i] = w
        extra.b[i] = b
    base = MLP([11, 1], rng)
    base.W[0][:, 0] = weights
    base.b[0][:] = bias
    combined = _Combined(base, extra)
    pred = combined.predict(features)
    print(
        f"{label} fit mean {pred.mean():+.5f} std {pred.std():.5f} "
        f"val {best_val:.5f} linear val {float(np.mean((target[val_i] - linear[val_i]) ** 2)):.5f}",
        flush=True,
    )
    return combined


class _Combined:
    def __init__(self, linear: MLP, extra: MLP):
        self.linear = linear
        self.extra = extra

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.linear.forward(features)[:, 0] + self.extra.forward(features)[:, 0]

    def forward(self, features: np.ndarray) -> np.ndarray:
        return self.predict(features).reshape(-1, 1)


def train_value(features: np.ndarray, returns: np.ndarray, seed: int) -> tuple[_Combined, float]:
    model = _fit_residual(features, returns, seed, "value")
    pred = model.predict(features)
    variance = max(float(np.var(returns - pred, ddof=1)), 1.0)
    full = BANKROLL * np.maximum(pred, 0.0) / variance
    betting = pred > 0.0
    bind = float(np.mean(full[betting] >= MAX_BET)) if np.any(betting) else 0.0
    print(
        f"full Kelly would hit the cap of {MAX_BET:.0f} on {bind:.1%} of bet hands, "
        f"mean uncapped stake {full.mean():.3f}",
        flush=True,
    )
    return model, variance


def train_insurance(features: np.ndarray, labels: np.ndarray, seed: int) -> _Combined:
    model = _fit_residual(features, labels - (1.0 / 3.0), seed, "insurance")
    pred = model.predict(features)
    print(f"insurance taken on training shoes {(pred > 0).mean():.1%}", flush=True)
    return model


class Solver:
    """Stake, insurance, and play are network outputs."""

    def __init__(self, play: MLP, value: _Combined, insurance: _Combined, variance: float, fraction: float, max_bet: float):
        self.play = play
        self.value = value
        self.insurance = insurance
        self.variance = float(variance)
        self.fraction = float(fraction)
        self.max_bet = float(max_bet)

    def edge(self, shoe: Shoe) -> float:
        return float(self.value.predict(shoe_features(shoe).reshape(1, -1))[0])

    def stake(self, shoe: Shoe) -> float:
        raw = self.fraction * BANKROLL * max(self.edge(shoe), 0.0) / self.variance
        return float(min(max(raw, 1.0), self.max_bet))

    def take_insurance(self, _tc: float, shoe: Shoe) -> bool:
        score = float(self.insurance.predict(shoe_features(shoe).reshape(1, -1))[0])
        return bool(score > 0.0)


def _optimal_ins(_tc: float, shoe: Shoe) -> bool:
    unseen = shoe.unseen()
    if unseen <= 0:
        return False
    return float(shoe.remaining[10]) / unseen > (1.0 / 3.0)


def _unit_results(snaps, decide, insure) -> np.ndarray:
    units = np.zeros(len(snaps))
    for i, snap in enumerate(snaps):
        shoe = Shoe.from_snapshot(snap)
        units[i] = play_hand(shoe, 1.0, decide, insure)["profit"]
    return units


def _edges(snaps, solver: Solver) -> np.ndarray:
    edges = np.zeros(len(snaps))
    for i, snap in enumerate(snaps):
        edges[i] = solver.edge(Shoe.from_snapshot(snap))
    return edges


def _choose_always(edges: np.ndarray, units: np.ndarray, variance: float, base_profits: np.ndarray) -> tuple[float, float]:
    """Stake scale that bets at least 1 and gains the most over full-Kelly exact play."""
    best_gap = -1e9
    best = (1.0, MAX_BET)
    for fraction in (1.0, 0.75, 0.5, 0.25):
        raw = fraction * BANKROLL * np.maximum(edges, 0.0) / variance
        cap = max(MAX_BET, float(np.quantile(np.maximum(raw, 1.0), 0.99)))
        for limit in (MAX_BET, cap):
            stakes = np.clip(np.maximum(raw, 1.0), 1.0, max(limit, 1.0))
            gap = float(np.mean(units * stakes - base_profits))
            if gap > best_gap:
                best_gap = gap
                best = (fraction, float(max(limit, 1.0)))
    print(f"calibration gap at chosen scale {best_gap:+.5f}", flush=True)
    return best


def _choose_risk(edges: np.ndarray, units: np.ndarray, variance: float, baseline_sd: float) -> tuple[float, float]:
    """Largest stake scale whose profit spread stays under full-Kelly exact play."""
    best_mean = -1e9
    best = (0.25, MAX_BET)
    safest = (1e9, 0.25, MAX_BET)
    for fraction in (1.0, 0.75, 0.5, 0.25):
        raw = fraction * BANKROLL * np.maximum(edges, 0.0) / variance
        cap = max(MAX_BET, float(np.quantile(raw, 0.99)))
        for limit in (MAX_BET, cap):
            stakes = np.clip(raw, 0.0, limit)
            profit = units * stakes
            spread = float(np.std(profit))
            mean = float(np.mean(profit))
            if spread < safest[0]:
                safest = (spread, fraction, float(limit))
            if spread < baseline_sd and mean > best_mean:
                best_mean = mean
                best = (fraction, float(limit))
    if best_mean > -1e8:
        return best
    return safest[1], safest[2]


def _finetune(play: MLP, solver: Solver, seed: int) -> MLP:
    """One pass of outcome updates. Keep it only when the paired holdout improves."""
    rng = np.random.default_rng(seed)
    trial = play.clone(rng)
    snaps, _ids = generate_snapshots(40000, seed)
    for snap in snaps:
        shoe = Shoe.from_snapshot(snap)
        advantage = 0.0
        taken: list[tuple[np.ndarray, int]] = []

        def decide(state, legal, ctx, taken=taken):
            state = state.copy()
            state[51] = 0.0
            logits = trial.forward(state.reshape(1, -1))[0]
            logits = np.where(legal > 0.0, logits, -1e9)
            shifted = logits - np.max(logits)
            probs = np.exp(np.clip(shifted, -40.0, 0.0))
            probs = np.where(legal > 0.0, probs, 0.0)
            probs = probs / max(float(probs.sum()), 1e-12)
            action = int(rng.choice(5, p=probs))
            taken.append((state, legal.copy(), action))
            return action

        result = play_hand(shoe, 1.0, decide, solver.take_insurance)
        advantage = float(np.clip(result["profit"] - solver.edge(Shoe.from_snapshot(snap)), -2.0, 2.0))
        for state, legal, action in taken:
            logits = trial.forward(state.reshape(1, -1))
            masked = np.where(legal.reshape(1, -1) > 0.0, logits, -1e9)
            shifted = masked - masked.max(axis=1, keepdims=True)
            probs = np.exp(np.clip(shifted, -40.0, 0.0))
            probs = np.where(legal.reshape(1, -1) > 0.0, probs, 0.0)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            grad = probs.copy()
            grad[0, action] -= 1.0
            grad *= advantage / max(len(taken), 1)
            trial.backward(grad, lr=1e-5, clip=1.0)
    val_snaps, val_ids = generate_snapshots(80000, seed + 1)
    before = _unit_results(val_snaps, play_decide(play), solver.take_insurance)
    after = _unit_results(val_snaps, play_decide(trial), solver.take_insurance)
    diff = cluster_diff(after, before, val_ids)
    print(
        f"play fine-tune minus imitation {fmt_ci(diff['mean'], diff['ci'])} p={fmt_p(diff['p'])}",
        flush=True,
    )
    if diff["ci"][0] > 0.0:
        print("keeping fine-tuned play", flush=True)
        return trial
    print("keeping imitation play", flush=True)
    return play


def main() -> None:
    build()
    RESULTS.mkdir(exist_ok=True)
    print("training play network", flush=True)
    play = train_play(8)
    print("sampling shoe outcomes", flush=True)
    features, returns, ins_x, ins_y = gather(play, 1500000, 11)
    value, variance = train_value(features, returns, 5)
    insurance = train_insurance(ins_x, ins_y, 6)
    solver = Solver(play, value, insurance, variance, fraction=1.0, max_bet=MAX_BET)
    decide = play_decide(play)

    print("choosing a stake scale on calibration shoes", flush=True)
    calib_snaps, calib_ids = generate_snapshots(200000, 50)
    calib_units = _unit_results(calib_snaps, decide, solver.take_insurance)
    calib_edges = _edges(calib_snaps, solver)
    base_calib = evaluate(calib_snaps, calib_ids, exact_decide, _optimal_ins, lambda shoe: 1.0)
    base_model, _fit = fit_tc_model(base_calib["tcs"], base_calib["profits"], base_calib["clusters"])

    def base_bet(shoe: Shoe) -> float:
        return base_model.bet(shoe.true_count(), wong=False)

    base_units = _unit_results(calib_snaps, exact_decide, _optimal_ins)
    base_stakes = np.array([base_bet(Shoe.from_snapshot(snap)) for snap in calib_snaps])
    base_profits_calib = base_units * base_stakes
    fraction, cap = _choose_always(calib_edges, calib_units, variance, base_profits_calib)
    solver.fraction = fraction
    solver.max_bet = cap
    print(f"kelly fraction {fraction:.2f} cap {cap:.2f} minimum bet 1", flush=True)

    print("holdout", flush=True)
    snaps, ids = generate_snapshots(1000000, 99)
    units = _unit_results(snaps, decide, solver.take_insurance)
    edges = _edges(snaps, solver)
    raw = fraction * BANKROLL * np.maximum(edges, 0.0) / variance
    wagers = np.clip(np.maximum(raw, 1.0), 1.0, cap)
    profits = units * wagers
    base_units = _unit_results(snaps, exact_decide, _optimal_ins)
    base_stakes = np.array([base_bet(Shoe.from_snapshot(snap)) for snap in snaps])
    base_profits = base_units * base_stakes
    stats = cluster_mean(profits, ids)
    base_stats = cluster_mean(base_profits, ids)
    diff = cluster_diff(profits, base_profits, ids)
    lines = [
        "EVERY HAND BET AT LEAST 1",
        f"hands {len(snaps)} minimum bet {float(wagers.min()):.1f}",
        f"kelly fraction {fraction:.2f} cap {cap:.2f}",
        f"mean bet {float(wagers.mean()):.3f}",
        f"network profit {fmt_ci(stats['mean'], stats['ci'])} p={fmt_p(stats['p'])}",
        f"full-Kelly exact profit {fmt_ci(base_stats['mean'], base_stats['ci'])} p={fmt_p(base_stats['p'])}",
        f"network minus exact {fmt_ci(diff['mean'], diff['ci'])} z={diff['z']:.2f} p={fmt_p(diff['p'])}",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "always_bet.txt").write_text(text)
    (RESULTS / "always_bet.json").write_text(
        json.dumps(
            {
                "mean": stats["mean"],
                "p": stats["p"],
                "ci": list(stats["ci"]),
                "baseline_mean": base_stats["mean"],
                "diff_mean": diff["mean"],
                "diff_p": diff["p"],
                "diff_ci": list(diff["ci"]),
                "fraction": fraction,
                "cap": cap,
                "min_bet": float(wagers.min()),
                "mean_bet": float(wagers.mean()),
            },
            indent=2,
        )
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
