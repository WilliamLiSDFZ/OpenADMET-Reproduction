"""Assemble per-endpoint test predictions from the per-group, per-seed
models trained by ``train.train_all``.

HybridADMET aggregation rule (same as v4):

    Predict each endpoint using ONLY the model trained for that endpoint's
    PRIMARY group, then average across the 5 seeds.

Identical to v4_hybridadmet.predict.build_submission — we just keep a local
copy so v5 isn't accidentally affected if v4 evolves.
"""
from __future__ import annotations

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
        targets = group_results[0]["targets"]
        col_idx = targets.index(assay)
        # Average across seeds
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
