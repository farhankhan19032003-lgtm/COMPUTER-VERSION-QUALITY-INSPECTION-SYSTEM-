"""Small metric helpers shared by the training / evaluation scripts (numpy only)."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def auc_binary(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    """ROC AUC via the rank statistic (ties get average ranks). None if one class is missing."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sens_spec(y: np.ndarray, pred: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return (tp / (tp + fn) if tp + fn else None, tn / (tn + fp) if tn + fp else None)


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
    n: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap confidence interval for any metric fn(y, s)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], s[idx])
        if v is not None:
            vals.append(v)
    if len(vals) < n * 0.5:
        return None, None
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def fmt(v: Optional[float], d: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"
