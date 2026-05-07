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

Robustness layer (added 2026-05-05): NNLS over a single val split can
overfit val noise — for example, on Caco-2 endpoints chemprop's val MAE
is mildly competitive with classical ML, NNLS gives chemprop a small
weight, but on test chemprop is substantially worse so the ensemble
*regresses* compared to classical-only. The fix is to **drop any model
whose individual val MAE is >max_loss_ratio× the best-model val MAE**
*before* running NNLS. With a permissive ratio (default 1.10 = "10 %
worse than the best") we keep the diversity benefit of the ensemble
while dropping models that are clearly noisy on this endpoint.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
from scipy.optimize import minimize

from .config import ENSEMBLE_MIN_WEIGHT

# Models with val MAE > MAX_LOSS_RATIO * best_val_MAE are excluded from the
# NNLS pool entirely. Default 1.10 = "drop everyone more than 10% worse than
# the best learner on val". Override via env var.
MAX_LOSS_RATIO = float(os.environ.get("MAX_LOSS_RATIO", "1.10"))


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
                         y_holdout: np.ndarray,
                         max_loss_ratio: float | None = None,
                         ) -> Dict[str, float]:
    """Given {model_name: pred_on_holdout} -> {model_name: weight}.

    If ``max_loss_ratio`` is given (e.g. 1.10), models whose individual val
    MAE is more than ``max_loss_ratio × best_val_MAE`` are excluded from
    the NNLS pool. Defaults to the package-level ``MAX_LOSS_RATIO`` constant.
    """
    if max_loss_ratio is None:
        max_loss_ratio = MAX_LOSS_RATIO

    names = list(holdout_preds.keys())
    weights = {n: 0.0 for n in names}

    if not names:
        return weights

    # 1) Per-model individual val MAE
    individual_mae = {
        n: float(np.mean(np.abs(np.asarray(holdout_preds[n]) - y_holdout)))
        for n in names
    }
    best_mae = min(individual_mae.values())
    threshold = best_mae * max_loss_ratio

    # 2) Filter to "competitive" models -- the ones at most max_loss_ratio
    #    worse than the leader on val.
    competitive = [n for n in names if individual_mae[n] <= threshold]
    if not competitive:
        # Pathological (NaNs, etc.); fall back to all
        competitive = list(names)

    # 3) NNLS-on-simplex over the competitive subset only
    P = np.column_stack([holdout_preds[n] for n in competitive])
    w = _nnls_simplex(P, y_holdout)

    # 4) Drop tiny weights and re-normalize within the competitive subset
    mask = w >= ENSEMBLE_MIN_WEIGHT
    if not mask.any():
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = np.where(mask, w, 0.0)
        s = w.sum()
        w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))

    for n, wi in zip(competitive, w):
        weights[n] = float(wi)
    return weights


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
