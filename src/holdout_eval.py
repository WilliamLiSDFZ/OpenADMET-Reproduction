"""End-to-end holdout evaluation that ends with a real MA-RAE number.

Why this script exists:
    The official labeled test set lives on HuggingFace and isn't reachable from
    this sandbox. To get a faithful MA-RAE we instead split train.csv 80/20,
    train on 80, predict on 20, then hand both files to the official scorer in
    `../../ExpansionRx-Challenge-Eval/eval/__main__.py`.

Outputs (under output/holdout/):
    fake_test_with_labels.csv   the held-out 20% (ground truth)
    fake_test_blinded.csv       same rows, only Molecule Name + SMILES
    fake_submission.csv         our predictions on the held-out 20%
    eval_metrics.csv            the scorer's per-endpoint table (incl. MA-RAE)
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

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
OUT_DIR = ROOT / "output" / "holdout"
EVAL_REPO = Path("/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval")
FP_BITS = 2048
FP_RADIUS = 2
HOLDOUT_FRAC = 0.20
RANDOM_STATE = 42

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
    print(f"[1/6] Loading {DATA_DIR / 'train.csv'}")
    full = pd.read_csv(DATA_DIR / "train.csv")
    print(f"      Loaded {len(full)} compounds")

    # Train / holdout split (random; 80 / 20).
    train_df, test_df = train_test_split(
        full, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"      Train: {len(train_df)}    Holdout: {len(test_df)}")

    # Save the holdout file the eval module expects (with labels) and a
    # blinded copy for symmetry with the real challenge format.
    test_df.to_csv(OUT_DIR / "fake_test_with_labels.csv", index=False)
    test_df[["Molecule Name", "SMILES"]].to_csv(
        OUT_DIR / "fake_test_blinded.csv", index=False
    )

    # 2) Log-transform every endpoint (LogD passes through).
    forward, _ = get_conversion_dataframes()
    log_train = train_df[["SMILES", "Molecule Name"]].copy()
    log_columns = []
    for assay in SUBMISSION_COLUMNS:
        log_scale, multiplier, short = forward[assay]
        log_train[short] = log_transform(train_df[assay], log_scale, multiplier)
        log_columns.append((assay, short))

    # 3) Featurize both partitions.
    print(f"[2/6] Featurizing train ({len(train_df)})")
    X_train, _ = featurize_smiles_batch(
        train_df["SMILES"].tolist(), n_bits=FP_BITS, radius=FP_RADIUS,
    )
    print(f"[3/6] Featurizing holdout ({len(test_df)})")
    X_test, _ = featurize_smiles_batch(
        test_df["SMILES"].tolist(), n_bits=FP_BITS, radius=FP_RADIUS,
    )
    assert X_train.shape[1] == feature_dim(FP_BITS) == X_test.shape[1]

    # 4) Per-endpoint LightGBM.
    print(f"[4/6] Training {len(log_columns)} endpoints "
          f"({LGBM_PARAMS['n_estimators']} trees each)")
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise SystemExit("Install lightgbm first: pip install lightgbm") from e

    submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})
    for assay, short in log_columns:
        log_scale, multiplier, _ = forward[assay]
        mask = log_train[short].notna().to_numpy()
        if mask.sum() < 20:
            print(f"      {assay}: only {mask.sum()} train samples -- skipping")
            submission[assay] = np.nan
            continue
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train[mask], log_train.loc[mask, short].to_numpy())
        pred_log = model.predict(X_test)
        pred = inverse_log_transform(pred_log, log_scale, multiplier)
        if log_scale:
            pred = np.clip(pred, a_min=0.0, a_max=None)
        submission[assay] = pred
        print(f"      {assay:30s} (N_train={mask.sum()})  pred mean={pred.mean():.3f}")

    sub_path = OUT_DIR / "fake_submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"      Wrote {sub_path}")

    # 5) Hand off to the official scorer.
    print(f"[5/6] Running official eval at {EVAL_REPO}")
    if not EVAL_REPO.exists():
        raise SystemExit(f"Cannot find {EVAL_REPO}; please update EVAL_REPO in the script.")

    metrics_path = OUT_DIR / "eval_metrics.csv"
    # Note: we deliberately skip --formatted; the upstream pretty-printer has
    # a known bug where it looks up 'mean_MA-RAE' instead of 'mean_RAE'.
    # We do the macro print ourselves below.
    cmd = [
        sys.executable, "-m", "eval",
        str(sub_path),
        "--ground-truth", str(OUT_DIR / "fake_test_with_labels.csv"),
        "--output", str(metrics_path),
    ]
    print("      $ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=EVAL_REPO, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"eval exited with code {proc.returncode}")

    # 6) Pretty print the macro line.
    print(f"\n[6/6] Read back {metrics_path}")
    res = pd.read_csv(metrics_path)
    # Format mean ± std as a single column for readability
    pretty = pd.DataFrame({"Endpoint": res["Endpoint"]})
    for m in ("MAE", "RAE", "R2", "Spearman R", "Kendall's Tau"):
        m_col, s_col = f"mean_{m}", f"std_{m}"
        if m_col in res.columns and s_col in res.columns:
            pretty[m] = [f"{v:.3f} ± {s:.3f}" for v, s in zip(res[m_col], res[s_col])]
    print(pretty.to_string(index=False))

    macro = res[res["Endpoint"].astype(str).str.contains("Macro", case=False, na=False)]
    if not macro.empty:
        ma_rae = macro.iloc[0]["mean_RAE"]
        std_ma_rae = macro.iloc[0]["std_RAE"]
        macro_r2 = macro.iloc[0]["mean_R2"]
        print()
        print(f"   *** Final MA-RAE on 20% holdout = {ma_rae:.3f} ± {std_ma_rae:.3f} ***")
        print(f"       (Macro R²              = {macro_r2:+.3f})")


if __name__ == "__main__":
    main()
