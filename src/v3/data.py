"""Data loading + endpoint-specific preprocessing.

DESIGN NOTE on zero-handling (lesson learned the hard way):

  An *earlier* version of this file applied Inductive Bio's recipe of
  replacing zeros with ``half_min`` (clearance/permeability) or 1e-6
  (PPB) before ``log10``. On this dataset that turned out to be
  catastrophic: KSOL's smallest non-zero value is 0.0029 μM, so applying
  ``log10(x * 1e-6)`` to it gives -8.5 — a huge outlier compared to the
  bulk of the distribution near -4. The model bled accuracy chasing
  those extremes and the test MA-RAE jumped from 0.756 (v1 recipe) to
  1.78 (Inductive Bio recipe).

  The simple ``log10((x + 1) * multiplier)`` recipe used by the official
  tutorial ("tutorial" below) gracefully handles zeros without creating
  outliers, and that's what we use by default. The fancy zero-handling
  is left in as an opt-in for future ablations only.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import (
    ENDPOINT_PREP, GROUND_TRUTH_CSV, SUBMISSION_COLUMNS,
    TEST_CSV, TRAIN_CSV,
)


def _half_min_floor(values: pd.Series) -> float:
    """Half of the smallest strictly-positive value (Inductive Bio recipe)."""
    pos = values[values > 0]
    if pos.empty:
        return 1e-6
    return float(pos.min() / 2.0)


def log_transform_endpoint(values: pd.Series, log_scale: bool, multiplier: float,
                           zero_handling: str) -> pd.Series:
    """Apply per-endpoint log transform.

    Default behaviour ("tutorial") = ``log10((x + 1) * multiplier)``,
    matching the official OpenADMET tutorial. The +1 is a clean
    zero-handler that doesn't create extreme outliers.

    Other modes are kept for ablation only and are NOT recommended:
      * half_min   : Inductive Bio recipe; replace 0 with half-min, log10
      * ppb_floor  : Inductive Bio recipe; replace 0 with 1e-6, log10
      * passthrough: no transform (used for LogD)
    """
    values = values.astype(float)
    if not log_scale or zero_handling == "passthrough":
        return values

    if zero_handling == "half_min":
        floor = _half_min_floor(values)
        clipped = values.where(values > 0, floor)
        return np.log10(clipped * multiplier)
    if zero_handling == "ppb_floor":
        clipped = values.where(values > 0, 1e-6)
        return np.log10(clipped * multiplier)
    # default: official tutorial recipe — zero-safe and outlier-free
    return np.log10((values + 1.0) * multiplier)


def inverse_log_endpoint(log_values: np.ndarray, assay: str) -> np.ndarray:
    """Inverse the log transform, mapping back to the original scale."""
    log_scale, multiplier, _, zero_handling = ENDPOINT_PREP[assay]
    if not log_scale or zero_handling == "passthrough":
        return log_values
    if zero_handling in ("half_min", "ppb_floor"):
        return 10 ** log_values / multiplier
    # default: official tutorial recipe inverse
    return 10 ** log_values / multiplier - 1.0


def load_train() -> pd.DataFrame:
    """Return train.csv with all original columns + per-endpoint log columns."""
    df = pd.read_csv(TRAIN_CSV)
    out = df[["Molecule Name", "SMILES"]].copy()
    for assay in SUBMISSION_COLUMNS:
        log_scale, multiplier, short, zh = ENDPOINT_PREP[assay]
        out[short] = log_transform_endpoint(df[assay], log_scale, multiplier, zh)
        out[assay] = df[assay]   # keep original
    return out


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_CSV)


def per_endpoint_views(train: pd.DataFrame
                       ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build per-endpoint training views, dropping NaNs for that endpoint.

    Returns ``{assay: (mask_idx, smiles_array, log_y_array)}`` where
    ``mask_idx`` is the integer row indices into ``train`` that are kept
    (lets us subset feature matrices later without re-aligning SMILES).
    """
    out = {}
    for assay in SUBMISSION_COLUMNS:
        _, _, short, _ = ENDPOINT_PREP[assay]
        idx = train.index[train[short].notna()].to_numpy()
        out[assay] = (
            idx,
            train.loc[idx, "SMILES"].to_numpy(),
            train.loc[idx, short].to_numpy(),
        )
    return out
