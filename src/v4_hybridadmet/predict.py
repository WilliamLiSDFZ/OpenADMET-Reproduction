"""Assemble per-endpoint test predictions from the per-group, per-seed
models trained by ``train.train_all``.

HybridADMET aggregation rule (from their methodology report):

    Predict each endpoint using ONLY the model trained for that endpoint's
    PRIMARY group, then average across the 5 seeds.

Mapping (see ``config.ENDPOINT_TO_GROUP``):
    LogD      -> logd_alone group (1 task)
    KSOL      -> ksol_centric group (6 tasks)
    HLM CLint -> metab_plus_mppb group (7 tasks)
    MPPB      -> metab_plus_mppb group (7 tasks)
    Caco-2 Efflux/Papp -> perm_only group (4 tasks)
    MLM CLint -> all_nine group
    MBPB      -> all_nine group
    MGMB      -> all_nine group
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from ..v3.data import inverse_log_endpoint


def build_submission(results: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-group, per-seed test predictions into a submission CSV.

    Args:
        results: dict returned by train.train_all -- {group_name: [seed_dicts]}
        test_df: original test.csv (for the Molecule Name column)

    Returns:
        DataFrame with ``Molecule Name`` + 9 endpoint columns (original units).
    """
    submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})

    for assay in cfg.SUBMISSION_COLUMNS:
        group_name = cfg.ENDPOINT_TO_GROUP[assay]
        group_results = results[group_name]
        # Each seed result has test_preds: (n_test, n_targets_in_group)
        targets = group_results[0]["targets"]
        # Find the column index in the group's output for this endpoint
        col_idx = targets.index(assay)
        # Average across seeds (HybridADMET = mean of 5 seeds)
        per_seed_preds = np.stack([r["test_preds"][:, col_idx]
                                    for r in group_results])
        avg_log_pred = per_seed_preds.mean(axis=0)
        # Inverse the log transform back to the original scale
        pred = inverse_log_endpoint(avg_log_pred, assay)
        # Non-LogD endpoints are non-negative
        log_scale = cfg.ENDPOINT_PREP[assay][0]
        if log_scale:
            pred = np.clip(pred, a_min=0.0, a_max=None)
        submission[assay] = pred
        print(f"  {assay:32s} <- {group_name:22s} (col {col_idx})  "
              f"pred mean={pred.mean():.3f}")

    return submission
