"""Train one LightGBM model per ADMET endpoint with 5-fold CV.

Outputs:
    output/cv_results.csv          per-fold R2/MAE/RAE
    output/cv_summary.csv          mean ± std per endpoint
    output/models/<short_name>.pkl one final model per endpoint (trained on full data)
    output/features_train.npz      cached features for reuse by predict.py
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from features import featurize_smiles_batch, feature_dim
from utils import (
    SUBMISSION_COLUMNS,
    get_conversion_dataframes,
    log_transform,
    relative_absolute_error,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------- Config -----------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
FP_BITS = 2048
FP_RADIUS = 2
N_SPLITS = int(os.environ.get("N_SPLITS", 5))
RANDOM_STATE = 42

LGBM_PARAMS = dict(
    n_estimators=int(os.environ.get("N_ESTIMATORS", 300)),
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=int(os.environ.get("N_JOBS", 2)),
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data ------------------------------------------------------
    train_csv = DATA_DIR / "train.csv"
    print(f"[1/5] Loading {train_csv}")
    train_df = pd.read_csv(train_csv)
    print(f"      Loaded {len(train_df)} compounds with columns: {list(train_df.columns)}")

    # 2) Apply log transform per endpoint ------------------------------
    forward, _ = get_conversion_dataframes()
    print("[2/5] Log transforming endpoints")
    log_df = train_df[["SMILES", "Molecule Name"]].copy()
    log_columns = []  # list of (assay, short_name)
    for assay in SUBMISSION_COLUMNS:
        if assay not in train_df.columns:
            print(f"      WARNING: {assay} not found in train.csv -- skipping")
            continue
        log_scale, multiplier, short = forward[assay]
        log_df[short] = log_transform(train_df[assay], log_scale, multiplier)
        log_columns.append((assay, short))
        n_valid = log_df[short].notna().sum()
        print(f"      {short:20s} (from {assay}): N={n_valid}")

    # 3) Featurize ALL training SMILES (cached & reused per endpoint) --
    cache_path = OUTPUT_DIR / "features_train.npz"
    if cache_path.exists():
        print(f"[3/5] Loading cached features from {cache_path}")
        loaded = np.load(cache_path)
        X_all = loaded["X"]
        assert X_all.shape[1] == feature_dim(FP_BITS), \
            "Cached feature dim mismatch -- delete the cache and rerun."
    else:
        print(f"[3/5] Featurizing {len(log_df)} training SMILES "
              f"(dim={feature_dim(FP_BITS)})")
        X_all, failed = featurize_smiles_batch(
            log_df["SMILES"].tolist(), n_bits=FP_BITS, radius=FP_RADIUS,
        )
        if failed:
            print(f"      {len(failed)} molecules failed to parse")
        np.savez_compressed(cache_path, X=X_all)
        print(f"      Cached features at {cache_path}")

    # 4) 5-fold CV per endpoint, then fit on full data -----------------
    print(f"[4/5] 5-fold CV per endpoint  (LightGBM, {LGBM_PARAMS['n_estimators']} trees)")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    summary_rows = []

    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise SystemExit(
            "LightGBM is not installed. Run: pip install lightgbm"
        ) from e

    for assay, short in log_columns:
        mask = log_df[short].notna().to_numpy()
        X = X_all[mask]
        y = log_df.loc[mask, short].to_numpy()
        if len(y) < N_SPLITS * 2:
            print(f"      Skipping {short}: only {len(y)} samples")
            continue

        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            model = LGBMRegressor(**LGBM_PARAMS)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            r2 = r2_score(y[va_idx], pred)
            mae = mean_absolute_error(y[va_idx], pred)
            rae = relative_absolute_error(y[va_idx], pred)
            cv_rows.append({"endpoint": short, "fold": fold,
                            "n_train": len(tr_idx), "n_val": len(va_idx),
                            "R2": r2, "MAE": mae, "RAE": rae})
            fold_metrics.append((r2, mae, rae))

        arr = np.array(fold_metrics)
        summary_rows.append({
            "endpoint": short,
            "assay": assay,
            "n_samples": int(len(y)),
            "R2_mean": arr[:, 0].mean(),  "R2_std": arr[:, 0].std(),
            "MAE_mean": arr[:, 1].mean(), "MAE_std": arr[:, 1].std(),
            "RAE_mean": arr[:, 2].mean(), "RAE_std": arr[:, 2].std(),
        })
        print(f"      {short:20s} R2={arr[:,0].mean():+.3f}±{arr[:,0].std():.3f}  "
              f"MAE={arr[:,1].mean():.3f}  RAE={arr[:,2].mean():.3f}  "
              f"(N={len(y)})")

        # Fit on full (filtered) training data and persist
        final_model = LGBMRegressor(**LGBM_PARAMS)
        final_model.fit(X, y)
        with open(MODELS_DIR / f"{short}.pkl", "wb") as f:
            pickle.dump({
                "model": final_model,
                "assay": assay,
                "short_name": short,
                "n_train": int(len(y)),
                "feature_dim": int(X.shape[1]),
                "fp_bits": FP_BITS,
                "fp_radius": FP_RADIUS,
            }, f)

    # 5) Write CV results ----------------------------------------------
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    macro_rae = summary_df["RAE_mean"].mean() if len(summary_df) else float("nan")
    macro_r2 = summary_df["R2_mean"].mean() if len(summary_df) else float("nan")
    print(f"[5/5] Macro-Averaged RAE on CV = {macro_rae:.3f}   "
          f"(approx. of leaderboard MA-RAE)")
    print(f"      Macro-Averaged R2  on CV = {macro_r2:+.3f}")
    summary_df.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)
    print(f"      Wrote {OUTPUT_DIR / 'cv_results.csv'} and "
          f"{OUTPUT_DIR / 'cv_summary.csv'}")
    print(f"      Saved {len(summary_rows)} models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
