"""Cluster-robust tests and intervals. Large-sample normal approximations."""

from __future__ import annotations

import math

import numpy as np


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def two_sided_p(z: float) -> float:
    if math.isnan(z) or math.isinf(z):
        return float("nan")
    return min(1.0, 2.0 * norm_sf(z))


def cluster_mean(x: np.ndarray, clusters: np.ndarray) -> dict:
    """Mean of x with a cluster-robust standard error (shoe-level dependence)."""
    x = np.asarray(x, dtype=np.float64)
    clusters = np.asarray(clusters)
    n = len(x)
    _uniq, inv = np.unique(clusters, return_inverse=True)
    c = int(inv.max()) + 1 if n else 0
    mu = float(x.mean()) if n else float("nan")
    if c < 2 or n < 2:
        return {"mean": mu, "se": float("nan"), "z": float("nan"), "p": float("nan"), "n": n, "clusters": c, "ci": (float("nan"), float("nan"))}
    centered = x - mu
    sums = np.bincount(inv, weights=centered)
    se = math.sqrt((c / (c - 1)) * float(np.sum(sums**2)) / (n**2))
    z = mu / se if se > 0 else float("inf")
    return {
        "mean": mu,
        "se": se,
        "z": z,
        "p": two_sided_p(z),
        "n": n,
        "clusters": c,
        "ci": (mu - 1.96 * se, mu + 1.96 * se),
    }


def cluster_diff(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> dict:
    return cluster_mean(np.asarray(x) - np.asarray(y), clusters)


def cluster_ratio(profits: np.ndarray, wagers: np.ndarray, clusters: np.ndarray) -> dict:
    """Total profit / total posted wager, cluster-robust delta-method interval."""
    profits = np.asarray(profits, dtype=np.float64)
    wagers = np.asarray(wagers, dtype=np.float64)
    wager_sum = float(wagers.sum())
    n = len(profits)
    _uniq, inv = np.unique(clusters, return_inverse=True)
    c = int(inv.max()) + 1
    if wager_sum <= 0 or c < 2:
        return {"ratio": float("nan"), "se": float("nan"), "z": float("nan"), "p": float("nan"), "ci": (float("nan"), float("nan")), "n": n, "clusters": c}
    ratio = float(profits.sum()) / wager_sum
    resid = profits - ratio * wagers
    sums = np.bincount(inv, weights=resid)
    se = math.sqrt((c / (c - 1)) * float(np.sum(sums**2))) / wager_sum
    z = ratio / se if se > 0 else float("inf")
    return {
        "ratio": ratio,
        "se": se,
        "z": z,
        "p": two_sided_p(z),
        "ci": (ratio - 1.96 * se, ratio + 1.96 * se),
        "n": n,
        "clusters": c,
    }


def ols_cluster(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict:
    """Simple regression y = a + b x with a cluster-robust slope test."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    _uniq, inv = np.unique(clusters, return_inverse=True)
    c = int(inv.max()) + 1
    meat = np.zeros((2, 2))
    for k in range(c):
        mask = inv == k
        xk = X[mask]
        ek = resid[mask]
        score = xk.T @ ek
        meat += np.outer(score, score)
    xtx_inv = np.linalg.inv(X.T @ X)
    # Liang-Zeger with a finite-sample factor.
    scale = (c / max(c - 1, 1)) * ((n - 1) / max(n - 2, 1))
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    slope = float(beta[1])
    z = slope / se if se > 0 else float("inf")
    return {
        "intercept": float(beta[0]),
        "slope": slope,
        "se": se,
        "z": z,
        "p": two_sided_p(z),
        "ci": (slope - 1.96 * se, slope + 1.96 * se),
        "n": n,
        "clusters": c,
        "sigma": float(np.std(resid, ddof=2)),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    den = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / den
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / den
    return phat, max(0.0, center - margin), min(1.0, center + margin)


def fmt_p(p: float) -> str:
    if p != p:
        return "nan"
    if p < 1e-6:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_ci(mean: float, ci: tuple[float, float], digits: int = 5) -> str:
    return f"{mean:+.{digits}f}  (95% CI {ci[0]:+.{digits}f} to {ci[1]:+.{digits}f})"
