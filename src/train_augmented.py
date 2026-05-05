"""Train LightGBM with external-data augmentation, then run official eval.

Strategy: per challenge endpoint, concatenate external rows (from
``external_data.load_augmentation``) onto the challenge training set,
attaching a sample_weight (challenge=1.0, measured=0.5, predicted=0.2).

End of script: predicts on test.csv -> writes submission_augmented.csv ->
calls the official `python -m eval` for a real MA-RAE.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from external_data import load_augmentation  # noqa: E402
from features import featurize_smiles_batch, feature_dim  # noqa: E402
from utils import (  # noqa: E402
    SUBMISSION_COLUMNS,
    get_conversion_dataframes,
    inverse_log_transform,
    log_transform,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = THIS_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "augmented"
EVAL_REPO = Path("/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval")
GROUND_TRUTH = DATA_DIR / "test_ground_truth.csv"
FP_BITS = 2048
FP_RADIUS = 2
RANDOM_STATE = 42
W_CHALLENGE = 1.0  # weight on the official training rows

LGBM_PARAMS = dict(
    n_estimators=int(os.environ.get("N_ESTIMATORS", 300)),
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=int(os.environ.get("N_JOBS", 4)),
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load challenge data ------------------------------------------------
    print(f"[1/6] Loading {DATA_DIR / 'train.csv'} and {DATA_DIR / 'test.csv'}")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"      Challenge train: {len(train_df)}    test: {len(test_df)}")

    forward, _ = get_conversion_dataframes()
    log_train = train_df[["SMILES", "Molecule Name"]].copy()
    log_columns = []
    for assay in SUBMISSION_COLUMNS:
        log_scale, multiplier, short = forward[assay]
        log_train[short] = log_transform(train_df[assay], log_scale, multiplier)
        log_columns.append((assay, short, log_scale, multiplier))

    # 2) Featurize challenge train + test (reuse caches from train.py / predict.py)
    print(f"[2/6] Featurizing challenge train/test (dim={feature_dim(FP_BITS)})")
    train_cache = ROOT / "output" / "features_train.npz"
    test_cache = ROOT / "output" / "features_test.npz"
    if train_cache.exists():
        print(f"      Reusing {train_cache}")
        X_train_challenge = np.load(train_cache)["X"]
    else:
        X_train_challenge, _ = featurize_smiles_batch(
            log_train["SMILES"].tolist(), n_bits=FP_BITS, radius=FP_RADIUS,
        )
        np.savez_compressed(train_cache, X=X_train_challenge)
    if test_cache.exists():
        print(f"      Reusing {test_cache}")
        X_test = np.load(test_cache)["X"]
    else:
        X_test, _ = featurize_smiles_batch(
            test_df["SMILES"].tolist(), n_bits=FP_BITS, radius=FP_RADIUS,
        )
        np.savez_compressed(test_cache, X=X_test)
    assert X_train_challenge.shape[1] == feature_dim(FP_BITS) == X_test.shape[1]

    # 3) Per-endpoint: assemble augmented training, fit, predict -----------
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise SystemExit("Install lightgbm first.") from e

    # Pre-computed external feature pool (run precompute_external_features.py first)
    pool_path = OUT_DIR / "features_external_pool.npz"
    if not pool_path.exists():
        raise SystemExit(
            f"Missing {pool_path}. Run precompute_external_features.py first."
        )
    pool = np.load(pool_path, allow_pickle=True)
    pool_smiles = pool["smiles"].astype(str)
    pool_X = pool["X"]
    pool_lookup = {s: i for i, s in enumerate(pool_smiles)}
    print(f"      Loaded {len(pool_smiles)} pre-fingerprinted external SMILES")

    submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})
    aug_summary_rows = []

    print("[3/6] Per-endpoint training (challenge + external)")
    for assay, short, log_scale, multiplier in log_columns:
        # Challenge rows for this endpoint
        mask = log_train[short].notna().to_numpy()
        Xc = X_train_challenge[mask]
        yc = log_train.loc[mask, short].to_numpy()
        wc = np.full(len(yc), W_CHALLENGE, dtype=float)

        # External rows
        ext = load_augmentation(short)
        if not ext.empty:
            # Drop external SMILES that overlap with train OR test (avoid leakage)
            forbid = set(train_df["SMILES"]).union(set(test_df["SMILES"]))
            keep = ~ext["SMILES"].isin(forbid)
            ext = ext[keep].reset_index(drop=True)

        n_ext = len(ext)
        if n_ext > 0:
            # Look up features in the precomputed pool (avoid re-fingerprinting)
            idx = [pool_lookup.get(s) for s in ext["SMILES"].tolist()]
            kept = [i for i, j in enumerate(idx) if j is not None]
            if len(kept) < n_ext:
                ext = ext.iloc[kept].reset_index(drop=True)
                idx = [j for j in idx if j is not None]
                n_ext = len(ext)
            Xe = pool_X[idx]
            ye = ext["label"].to_numpy()
            we = ext["weight"].to_numpy()
            X = np.vstack([Xc, Xe])
            y = np.concatenate([yc, ye])
            w = np.concatenate([wc, we])
        else:
            X, y, w = Xc, yc, wc

        aug_summary_rows.append({
            "endpoint": short,
            "n_challenge": int(mask.sum()),
            "n_external": int(n_ext),
            "n_total": int(len(y)),
            "external_sources": (
                ", ".join(sorted(ext["source"].unique())) if n_ext else "—"
            ),
        })

        if len(y) < 20:
            print(f"      {short:20s} too few samples, skipping")
            submission[assay] = np.nan
            continue

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X, y, sample_weight=w)
        pred_log = model.predict(X_test)
        pred = inverse_log_transform(pred_log, log_scale, multiplier)
        if log_scale:
            pred = np.clip(pred, a_min=0.0, a_max=None)
        submission[assay] = pred
        print(f"      {short:20s}  N_chall={mask.sum():5d}  +ext={n_ext:5d}  "
              f"pred_mean={pred.mean():.3f}")

    aug_df = pd.DataFrame(aug_summary_rows)
    aug_df.to_csv(OUT_DIR / "augmentation_summary.csv", index=False)
    print(f"      Wrote {OUT_DIR / 'augmentation_summary.csv'}")

    sub_path = OUT_DIR / "submission_augmented.csv"
    submission.to_csv(sub_path, index=False)
    print(f"[4/6] Wrote {sub_path}")

    # 5) Run official eval against ground truth ------------------------------
    if not GROUND_TRUTH.exists():
        raise SystemExit(
            f"Missing {GROUND_TRUTH}. Place the labeled test set there first."
        )
    metrics_path = OUT_DIR / "official_eval_augmented.csv"
    cmd = [
        sys.executable, "-m", "eval",
        str(sub_path),
        "--ground-truth", str(GROUND_TRUTH),
        "--output", str(metrics_path),
    ]
    print(f"[5/6] Running official eval -> {metrics_path}")
    print("      $ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=EVAL_REPO, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"eval exited with code {proc.returncode}")

    # 6) Pretty print + side-by-side comparison with the no-augmentation run -
    print(f"\n[6/6] Comparing against baseline (output/official_eval_metrics.csv)")
    aug_res = pd.read_csv(metrics_path)
    baseline_path = ROOT / "output" / "official_eval_metrics.csv"
    if baseline_path.exists():
        base_res = pd.read_csv(baseline_path)
        merged = base_res[["Endpoint", "mean_RAE", "mean_R2"]].merge(
            aug_res[["Endpoint", "mean_RAE", "mean_R2"]],
            on="Endpoint", suffixes=("_baseline", "_augmented"),
        )
        merged["ΔRAE"] = merged["mean_RAE_augmented"] - merged["mean_RAE_baseline"]
        merged["ΔR2"]  = merged["mean_R2_augmented"]  - merged["mean_R2_baseline"]
        # Round for readability
        for c in merged.columns:
            if c.startswith(("mean_", "Δ")):
                merged[c] = merged[c].round(3)
        comp_path = OUT_DIR / "comparison_vs_baseline.csv"
        merged.to_csv(comp_path, index=False)
        print(f"      Wrote {comp_path}")
        print(merged.to_string(index=False))
        macro = merged[merged["Endpoint"].str.contains("Macro", case=False, na=False)]
        if not macro.empty:
            row = macro.iloc[0]
            print()
            print(f"   *** baseline   MA-RAE = {row['mean_RAE_baseline']:.3f}")
            print(f"   *** augmented  MA-RAE = {row['mean_RAE_augmented']:.3f}  "
                  f"(Δ = {row['ΔRAE']:+.3f})")


if __name__ == "__main__":
    main()
