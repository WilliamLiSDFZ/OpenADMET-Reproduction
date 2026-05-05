"""Utility helpers: log transforms, endpoint config, metrics."""
from __future__ import annotations

from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---- Endpoint configuration (matches the official tutorial) ----
# Assay -> (do_log10_after_x_plus_1, multiplier_before_log, short_name)
ENDPOINT_CONVERSION_TABLE = """Assay,Log_Scale,Multiplier,Log_name
LogD,False,1,LogD
KSOL,True,1e-6,LogS
HLM CLint,True,1,Log_HLM_CLint
MLM CLint,True,1,Log_MLM_CLint
Caco-2 Permeability Papp A>B,True,1e-6,Log_Caco_Papp_AB
Caco-2 Permeability Efflux,True,1,Log_Caco_ER
MPPB,True,1,Log_Mouse_PPB
MBPB,True,1,Log_Mouse_BPB
MGMB,True,1,Log_Mouse_MPB
"""

# The submission file uses these column names (the original assay names)
SUBMISSION_COLUMNS = [
    "LogD",
    "KSOL",
    "MLM CLint",
    "HLM CLint",
    "Caco-2 Permeability Efflux",
    "Caco-2 Permeability Papp A>B",
    "MPPB",
    "MBPB",
    "MGMB",
]


def get_conversion_dataframes() -> Tuple[Dict[str, tuple], Dict[str, tuple]]:
    """Return (forward_dict, reverse_dict) for log transforming endpoints."""
    df = pd.read_csv(StringIO(ENDPOINT_CONVERSION_TABLE))
    # forward: assay name -> (log_scale, multiplier, short_name)
    forward = {row["Assay"]: (bool(row["Log_Scale"]), float(row["Multiplier"]), row["Log_name"])
               for _, row in df.iterrows()}
    # reverse: short_name -> (assay, log_scale, multiplier)
    reverse = {row["Log_name"]: (row["Assay"], bool(row["Log_Scale"]), float(row["Multiplier"]))
               for _, row in df.iterrows()}
    return forward, reverse


def log_transform(values: pd.Series, log_scale: bool, multiplier: float) -> pd.Series:
    """Apply log10((x + 1) * multiplier). For LogD, return values unchanged."""
    values = values.astype(float)
    if not log_scale:
        return values
    # Add 1 to avoid log(0). Match the official tutorial.
    return np.log10((values + 1.0) * multiplier)


def inverse_log_transform(values: np.ndarray, log_scale: bool, multiplier: float) -> np.ndarray:
    """Reverse the log transform back to original scale."""
    if not log_scale:
        return values
    return 10 ** values * (1.0 / multiplier) - 1.0


# ---- Metrics ----
def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RAE = sum |y_pred - y_true| / sum |y_true - mean(y_true)|.

    This is the per-endpoint RAE used by the challenge before being macro-averaged.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_pred - y_true))
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom < 1e-12:
        return float("inf")
    return float(num / denom)
