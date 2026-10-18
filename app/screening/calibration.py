"""Calibration, threshold selection and out-of-distribution (OOD) guard for screening models.

Why these matter for a medical-style classifier:
  * Raw softmax scores are over-confident. Temperature scaling (one number fitted on validation
    data) makes "0.8" closer to "about 80% of such cases really are positive".
  * Triage thresholds should be chosen for a target SENSITIVITY (do not miss real findings), not
    left at an arbitrary 0.5.
  * A model must know when an image is unlike anything it was trained on (a cat photo, the wrong
    body part, another camera). A Mahalanobis distance on penultimate features gives that signal.

Everything here is plain numpy so it can be unit-tested without a GPU. All fitting must use a
VALIDATION split that was kept separate from training, by patient.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ temperature scaling
def _nll(logits: np.ndarray, y: np.ndarray, T: float, multilabel: bool) -> float:
    z = logits / T
    if multilabel:
        # numerically stable binary cross-entropy with logits
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y.astype(int)].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, multilabel: bool = False) -> float:
    """Single temperature T minimising validation NLL (grid + refinement; robust, no scipy needed).

    y: class indices (n,) for single-label, or a 0/1 matrix (n, k) for multilabel.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    if len(y) < 30:
        return 1.0                                     # too little data to fit anything honestly
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 120))
    best = min(grid, key=lambda t: _nll(logits, y, t, multilabel))
    fine = np.linspace(best * 0.93, best * 1.07, 40)
    best = min(fine, key=lambda t: _nll(logits, y, t, multilabel))
    return float(round(best, 4))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE for a binary event: p = predicted probability, y = 0/1 outcome."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


# ---------------------------------------------------------------- threshold selection
def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> Optional[float]:
    """Highest threshold whose sensitivity is still >= target (so the fewest false alarms for that goal).

    Returns None if there are too few positives to choose a threshold responsibly.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pos = np.sort(p[y == 1])
    if len(pos) < 20:
        return None
    # allow at most (1-target) of positives to fall below the threshold
    k = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[k])


def threshold_for_specificity(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> Optional[float]:
    """Lowest threshold whose specificity is >= target (used for the HIGH tier)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    neg = np.sort(p[y == 0])
    if len(neg) < 20:
        return None
    k = int(np.ceil(target * len(neg))) - 1
    return float(min(1.0, neg[min(k, len(neg) - 1)] + 1e-6))


def choose_triage_thresholds(y: np.ndarray, p: np.ndarray, target_sens: float = 0.90,
                             target_spec: float = 0.95) -> Dict[str, object]:
    """MODERATE catches `target_sens` of real findings; HIGH is where only (1-target_spec) of normals reach.

    Falls back to conservative defaults (0.30 / 0.60) if there are too few examples, and says so.
    """
    mod = threshold_for_sensitivity(y, p, target_sens)
    high = threshold_for_specificity(y, p, target_spec)
    if mod is None or high is None:
        return {"moderate_threshold": 0.30, "high_threshold": 0.60, "source": "default (too few validation examples)"}
    mod = float(np.clip(mod, 0.02, 0.90))
    high = float(np.clip(max(high, mod + 0.05), mod + 0.05, 0.99))
    return {"moderate_threshold": round(mod, 4), "high_threshold": round(high, 4),
            "source": f"validation: MODERATE for {int(target_sens * 100)}% sensitivity, "
                      f"HIGH for {int(target_spec * 100)}% specificity (optimistic: re-test on a fresh test set)"}


# ----------------------------------------------------------- Mahalanobis OOD guard
class MahalanobisOOD:
    """Class-conditional Gaussians with a shared covariance on penultimate-layer features.

    score(f) = min over classes of (f - mu_c)^T P (f - mu_c). Large = unlike the training images.
    """

    def __init__(self, means: np.ndarray, precision: np.ndarray, threshold: float) -> None:
        self.means = means.astype(np.float32)
        self.precision = precision.astype(np.float32)
        self.threshold = float(threshold)

    @classmethod
    def fit(cls, feats: np.ndarray, labels: np.ndarray, val_feats: np.ndarray, quantile: float = 0.99,
            margin: float = 1.25, shrink: float = 0.10) -> "MahalanobisOOD":
        """Means/covariance from training features; threshold from held-out validation features."""
        feats = np.asarray(feats, np.float64)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        means = np.stack([feats[labels == c].mean(0) for c in classes])
        centred = np.concatenate([feats[labels == c] - means[i] for i, c in enumerate(classes)])
        cov = centred.T @ centred / max(1, len(centred))
        d = cov.shape[0]
        cov = (1 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)     # shrinkage keeps it invertible
        prec = np.linalg.pinv(cov)
        obj = cls(means, prec, 0.0)
        scores = obj.score(val_feats)
        obj.threshold = float(np.quantile(scores, quantile) * margin)
        return obj

    def score(self, feats: np.ndarray) -> np.ndarray:
        f = np.atleast_2d(np.asarray(feats, np.float32))
        out = np.full(len(f), np.inf, dtype=np.float64)
        for mu in self.means:
            d = f - mu
            out = np.minimum(out, np.einsum("ij,jk,ik->i", d, self.precision, d))
        return out

    def is_ood(self, feats: np.ndarray) -> np.ndarray:
        return self.score(feats) > self.threshold

    def save(self, path) -> None:
        np.savez_compressed(path, means=self.means, precision=self.precision, threshold=self.threshold)

    @classmethod
    def load(cls, path) -> "MahalanobisOOD":
        z = np.load(path)
        return cls(z["means"], z["precision"], float(z["threshold"]))


# -------------------------------------------------------------- torch helpers (lazy)
def final_linear(net):
    """The last nn.Linear in a torchvision classifier (its input = the penultimate feature vector)."""
    import torch.nn as nn

    lin = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if not lin:
        raise ValueError("model has no Linear head")
    return lin[-1]


def collect_outputs(net, loader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a loader; return (logits, penultimate features, targets). targets may be (n,) or (n,k)."""
    import torch

    head = final_linear(net)
    store: List = []
    h = head.register_forward_pre_hook(lambda m, inp: store.append(inp[0].detach().float().cpu()))
    logits, ys = [], []
    net.eval()
    try:
        with torch.inference_mode():
            for x, y in loader:
                logits.append(net(x.to(device)).float().cpu())
                ys.append(y)
    finally:
        h.remove()
    return (torch.cat(logits).numpy(), torch.cat(store).numpy(), torch.cat(ys).numpy())
