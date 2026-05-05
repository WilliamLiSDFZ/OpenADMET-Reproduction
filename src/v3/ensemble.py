"""Per-endpoint weighted-average ensembling.

For each endpoint we have predictions from K base learners on a holdout
set with known labels. The "best ensemble" is the convex combination
of those predictions that minimizes MAE on the holdout. We solve it as a
small Non-Negative Least Squares problem with equality constraint
∑w_k = 1, then drop weights below a floor (``ENSEMBLE_MIN_WEIGHT``) and
re-normalize.

This is what the JCIM 2025 paper called "weighted average ensemble"; it
handily out-performs simple equal-weight averaging when one learner is
clearly better (and it gracefully reduces to single-best when only one
learner is good).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import minimize

from .config import ENSEMBLE_MIN_WEIGHT


def _nnls_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find w ∈ simplex (w_k ≥ 0, ∑w_k = 1) minimizing |P @ w - y|.

    P is (n_samples, n_models) of predictions, y is (n_samples,) ground
    truth. Solved as a small constrained LS via scipy.minimize.
    """
    K = P.shape[1]
    if K == 1:
        return np.ones(1)
    w0 = np.full(K, 1.0 / K)
    bounds = [(0.0, 1.0) for _ in range(K)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(
        fun=lambda w: float(np.mean((P @ w - y) ** 2)),
        x0=w0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def fit_ensemble_weights(holdout_preds: Dict[str, np.ndarray],
                         y_holdout: np.ndarray) -> Dict[str, float]:
    """Given {model_name: pred_on_holdout} -> {model_name: weight}."""
    names = list(holdout_preds.keys())
    P = np.column_stack([holdout_preds[n] for n in names])
    w = _nnls_simplex(P, y_holdout)

    # Drop tiny weights and re-normalize
    mask = w >= ENSEMBLE_MIN_WEIGHT
    if not mask.any():
        # everything got dropped; fall back to equal weights
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = np.where(mask, w, 0.0)
        w = w / w.sum()
    return {n: float(wi) for n, wi in zip(names, w)}


def apply_weights(test_preds: Dict[str, np.ndarray],
                  weights: Dict[str, float]) -> np.ndarray:
    """Combine per-model test predictions using the chosen weights."""
    out = None
    for name, w in weights.items():
        if w <= 0 or name not in test_preds:
            continue
        chunk = w * np.asarray(test_preds[name], dtype=np.float64)
        out = chunk if out is None else out + chunk
    return out
