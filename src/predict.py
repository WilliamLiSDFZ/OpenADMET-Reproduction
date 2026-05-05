"""Predict on test.csv with the saved per-endpoint LightGBM models.

Output:
    output/submission.csv
        Molecule Name + 9 endpoint predictions, in the original (non-log) scale.
        Format matches the OpenADMET HuggingFace Space submission format.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import featurize_smiles_batch, feature_dim
from utils import (
    SUBMISSION_COLUMNS,
    get_conversion_dataframes,
    inverse_log_transform,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
FP_BITS = 2048
FP_RADIUS = 2


def main():
    test_csv = DATA_DIR / "test.csv"
    print(f"[1/3] Loading {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"      Loaded {len(test_df)} test molecules")

    # Featurize the test set (cache for repeated runs)
    cache_path = OUTPUT_DIR / "features_test.npz"
    if cache_path.exists():
        print(f"[2/3] Loading cached features from {cache_path}")
        loaded = np.load(cache_path)
        X_test = loaded["X"]
    else:
        print(f"[2/3] Featurizing test SMILES (dim={feature_dim(FP_BITS)})")
        X_test, _ = featurize_smiles_batch(
            test_df["SMILES"].tolist(), n_bits=FP_BITS, radius=FP_RADIUS,
        )
        np.savez_compressed(cache_path, X=X_test)

    # Predict per-endpoint and inverse-transform back to the original scale
    print("[3/3] Predicting with saved models")
    _, reverse = get_conversion_dataframes()
    out = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})

    for assay in SUBMISSION_COLUMNS:
        # find the matching model file by short_name
        # reverse maps short_name -> (assay, log_scale, multiplier)
        short = next((s for s, (a, _, _) in reverse.items() if a == assay), None)
        if short is None:
            print(f"      WARNING: no short_name for {assay}, skipping")
            out[assay] = np.nan
            continue
        model_path = MODELS_DIR / f"{short}.pkl"
        if not model_path.exists():
            print(f"      WARNING: missing {model_path}; filling with NaN")
            out[assay] = np.nan
            continue
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        log_scale, multiplier = bundle.get("log_scale"), bundle.get("multiplier")
        if log_scale is None:
            _, log_scale, multiplier = reverse[short]
        pred_log = model.predict(X_test)
        pred_orig = inverse_log_transform(pred_log, log_scale, multiplier)
        # Clamp non-physical negative values (concentrations / ratios are >= 0)
        if log_scale:
            pred_orig = np.clip(pred_orig, a_min=0.0, a_max=None)
        out[assay] = pred_orig
        print(f"      {assay:30s} -> mean={pred_orig.mean():.3f}, "
              f"min={pred_orig.min():.3f}, max={pred_orig.max():.3f}")

    sub_path = OUTPUT_DIR / "submission.csv"
    out.to_csv(sub_path, index=False)
    print(f"\nWrote submission file: {sub_path}")
    print(f"  rows = {len(out)}, columns = {list(out.columns)}")


if __name__ == "__main__":
    main()
