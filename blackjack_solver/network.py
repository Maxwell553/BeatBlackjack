"""Small multilayer perceptron and replay buffer, implemented with NumPy only."""

from __future__ import annotations

import numpy as np


class MLP:
    """ReLU network with an Adam optimizer. The last layer is linear."""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.sizes = list(sizes)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.mW: list[np.ndarray] = []
        self.vW: list[np.ndarray] = []
        self.mb: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            weight = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            bias = np.zeros(fan_out)
            self.W.append(weight)
            self.b.append(bias)
            self.mW.append(np.zeros_like(weight))
            self.vW.append(np.zeros_like(weight))
            self.mb.append(np.zeros_like(bias))
            self.vb.append(np.zeros_like(bias))
        self.t = 0
        self._acts: list[np.ndarray] | None = None
        self._zs: list[np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        acts = [x]
        zs: list[np.ndarray] = []
        h = x
        last = len(self.W) - 1
        for i, (weight, bias) in enumerate(zip(self.W, self.b)):
            z = h @ weight + bias
            zs.append(z)
            h = z if i == last else np.maximum(z, 0.0)
            acts.append(h)
        self._acts = acts
        self._zs = zs
        return h

    def backward(self, d_out: np.ndarray, lr: float = 1e-3, clip: float = 10.0) -> None:
        if self._acts is None or self._zs is None:
            raise RuntimeError("backward called before forward")
        if not np.isfinite(d_out).all():
            return
        delta = d_out
        grads_w: list[np.ndarray | None] = [None] * len(self.W)
        grads_b: list[np.ndarray | None] = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            grads_w[i] = self._acts[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T
                delta *= self._zs[i - 1] > 0.0
        norm_sq = 0.0
        for grad in grads_w:
            norm_sq += float(np.sum(grad * grad))
        for grad in grads_b:
            norm_sq += float(np.sum(grad * grad))
        if not np.isfinite(norm_sq) or norm_sq <= 0.0:
            return
        scale = min(1.0, clip / (np.sqrt(norm_sq) + 1e-12))
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            g_w = grads_w[i] * scale
            g_b = grads_b[i] * scale
            self.mW[i] = b1 * self.mW[i] + (1.0 - b1) * g_w
            self.vW[i] = b2 * self.vW[i] + (1.0 - b2) * (g_w * g_w)
            self.mb[i] = b1 * self.mb[i] + (1.0 - b1) * g_b
            self.vb[i] = b2 * self.vb[i] + (1.0 - b2) * (g_b * g_b)
            m_w = self.mW[i] / (1.0 - b1**self.t)
            v_w = self.vW[i] / (1.0 - b2**self.t)
            m_b = self.mb[i] / (1.0 - b1**self.t)
            v_b = self.vb[i] / (1.0 - b2**self.t)
            self.W[i] -= lr * m_w / (np.sqrt(v_w) + eps)
            self.b[i] -= lr * m_b / (np.sqrt(v_b) + eps)

    def copy_params_from(self, other: "MLP") -> None:
        for i in range(len(self.W)):
            self.W[i] = other.W[i].copy()
            self.b[i] = other.b[i].copy()
        self.t = 0
        for i in range(len(self.W)):
            self.mW[i].fill(0.0)
            self.vW[i].fill(0.0)
            self.mb[i].fill(0.0)
            self.vb[i].fill(0.0)

    def clone(self, rng: np.random.Generator) -> "MLP":
        other = MLP(self.sizes, rng)
        other.copy_params_from(self)
        return other


def huber_grad(pred: np.ndarray, target: np.ndarray, delta: float) -> np.ndarray:
    err = pred - target
    return np.where(np.abs(err) <= delta, err, delta * np.sign(err))


def train_actions(
    net: MLP,
    states: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    lr: float,
    delta: float = 2.0,
) -> float:
    targets = np.clip(targets, -8.0, 8.0)
    q = net.forward(states)
    if not np.isfinite(q).all():
        return 0.0
    rows = np.arange(len(actions))
    pred = q[rows, actions]
    grad = huber_grad(pred, targets, delta)
    loss = float(np.mean(np.where(np.abs(pred - targets) <= delta, 0.5 * (pred - targets) ** 2, delta * (np.abs(pred - targets) - 0.5 * delta))))
    d_q = np.zeros_like(q)
    d_q[rows, actions] = grad / len(actions)
    net.backward(d_q, lr=lr)
    return loss


def train_all_outputs(
    net: MLP,
    states: np.ndarray,
    targets: np.ndarray,
    lr: float,
    delta: float = 1.0,
) -> float:
    pred = net.forward(states)
    err = pred - targets
    grad = huber_grad(pred, targets, delta)
    loss = float(np.mean(np.where(np.abs(err) <= delta, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))))
    net.backward(grad / len(states), lr=lr)
    return loss


class Replay:
    def __init__(self, capacity: int, dim: int):
        self.S = np.zeros((capacity, dim))
        self.A = np.zeros(capacity, np.int64)
        self.G = np.zeros(capacity)
        self.capacity = capacity
        self.i = 0
        self.n = 0

    def add(self, state: np.ndarray, action: int, ret: float) -> None:
        self.S[self.i] = state
        self.A[self.i] = action
        self.G[self.i] = ret
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, k: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, size=k)
        return self.S[idx], self.A[idx], self.G[idx]


def train_softmax(
    net: MLP,
    states: np.ndarray,
    legal: np.ndarray,
    actions: np.ndarray,
    lr: float = 1e-3,
) -> float:
    """Cross-entropy on the legal actions. Returns mean negative log likelihood."""
    logits = net.forward(states)
    if not np.isfinite(logits).all():
        return float("inf")
    masked = np.where(legal > 0.0, logits, -1e9)
    shifted = masked - masked.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 0.0))
    exp = np.where(legal > 0.0, exp, 0.0)
    denom = exp.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    probs = exp / denom
    rows = np.arange(len(actions))
    picked = np.clip(probs[rows, actions], 1e-12, 1.0)
    loss = float(-np.mean(np.log(picked)))
    grad = probs.copy()
    grad[rows, actions] -= 1.0
    grad = np.where(legal > 0.0, grad, 0.0) / len(actions)
    net.backward(grad, lr=lr, clip=5.0)
    return loss


def greedy_action(net: MLP, state: np.ndarray, legal: np.ndarray) -> int:
    q = net.forward(state.reshape(1, -1))[0]
    masked = np.where(legal > 0.0, q, -1e9)
    return int(np.argmax(masked))
