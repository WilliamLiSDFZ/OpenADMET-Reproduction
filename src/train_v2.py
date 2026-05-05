"""train_v2: combine the three top-3 improvements + run official eval.

  1. v2 features:  RDKit-2d + Morgan + Avalon  (+ optional Mordred)
  2. ADMET-AI distillation: 45 surrogate predictions concatenated as features
  3. SALI cliff masking per endpoint

Optional (controlled by env vars):
  ENABLE_AVALON_RDKIT_MORGAN=1   (default 1; v2 base features)
  ENABLE_DISTILL=1               (default 1)
  ENABLE_CLIFF_MASK=1            (default 1)
  EXT_PROFILE=none|all|selective (default 'none' -- v1's row-augmentation off
                                  to isolate v2 improvements)
  N_ESTIMATORS=int (default 300), N_JOBS=int (default 4)

Outputs (under output/v2/):
  submission_v2.csv
  official_eval_v2.csv          per-endpoint MAE/RAE/R²/...
  v2_vs_baseline.csv            apples-to-apples Δ vs the v1 baseline
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

# cliff_masking import only used as a fallback if no precomputed masks exist
from cliff_masking import detect_cliffs  # noqa: E402, F401
from external_data import load_augmentation  # noqa: E402
from utils import (  # noqa: E402
    SUBMISSION_COLUMNS,
    get_conversion_dataframes,
    inverse_log_transform,
    log_transform,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = THIS_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "v2"
EVAL_REPO = Path("/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval")
GROUND_TRUTH = DATA_DIR / "test_ground_truth.csv"
RANDOM_STATE = 42

ENABLE_DISTILL = bool(int(os.environ.get("ENABLE_DISTILL", "1")))
ENABLE_CLIFF_MASK = bool(int(os.environ.get("ENABLE_CLIFF_MASK", "1")))
EXT_PROFILE = os.environ.get("EXT_PROFILE", "none").lower()  # off by default

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


def _load(path: Path, key: str = "X") -> np.ndarray:
    return np.load(path, allow_pickle=True)[key]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"v2 config:  distill={ENABLE_DISTILL}  cliff={ENABLE_CLIFF_MASK}  "
          f"ext_profile={EXT_PROFILE}  n_est={LGBM_PARAMS['n_estimators']}")

    # 1) Load splits + featurized matrices ------------------------------
    print("[1/5] Loading challenge data + cached v2 features")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_train_base = _load(OUT_DIR / "features_train_v2.npz")
    X_test_base = _load(OUT_DIR / "features_test_v2.npz")
    print(f"      train={X_train_base.shape}, test={X_test_base.shape}")

    # 2) Concat distillation features ----------------------------------
    if ENABLE_DISTILL:
        D_train = _load(OUT_DIR / "distill_train.npz")
        D_test = _load(OUT_DIR / "distill_test.npz")
        X_train = np.concatenate([X_train_base, D_train], axis=1)
        X_test = np.concatenate([X_test_base, D_test], axis=1)
        print(f"      +distill: train={X_train.shape}, test={X_test.shape}")
    else:
        X_train, X_test = X_train_base, X_test_base

    # External pool (only used if EXT_PROFILE != 'none')
    if EXT_PROFILE != "none":
        ext_pool_X = _load(OUT_DIR / "features_external_v2.npz")
        ext_pool_smiles = _load(OUT_DIR / "features_external_v2.npz", key="smiles")
        ext_pool_lookup = {s: i for i, s in enumerate(ext_pool_smiles)}
        if ENABLE_DISTILL:
            ext_pool_distill = _load(OUT_DIR / "distill_external.npz")
            ext_pool_full = np.concatenate([ext_pool_X, ext_pool_distill], axis=1)
        else:
            ext_pool_full = ext_pool_X
        print(f"      external pool: {ext_pool_full.shape}")
        # Override the AUGMENTATION_LOADERS profile via env (already set)

    # 3) Log-transform endpoints + per-endpoint training --------------
    forward, _ = get_conversion_dataframes()
    log_train = train_df[["SMILES", "Molecule Name"]].copy()
    log_columns = []
    for assay in SUBMISSION_COLUMNS:
        log_scale, multiplier, short = forward[assay]
        log_train[short] = log_transform(train_df[assay], log_scale, multiplier)
        log_columns.append((assay, short, log_scale, multiplier))

    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise SystemExit("Install lightgbm.") from e

    submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})

    # Load precomputed cliff masks if available
    cliff_path = OUT_DIR / "cliff_masks.npz"
    cliff_masks_cache = None
    if ENABLE_CLIFF_MASK and cliff_path.exists():
        cliff_masks_cache = dict(np.load(cliff_path))
        print(f"      Using precomputed cliff masks: {cliff_path.name}")
    elif ENABLE_CLIFF_MASK:
        print("      WARN: cliff_masks.npz not found; running cliff detection on-the-fly")

    print("[2/5] Per-endpoint training")
    for assay, short, log_scale, multiplier in log_columns:
        mask = log_train[short].notna().to_numpy()

        # 3a) SALI cliff masking on the in-endpoint data
        if ENABLE_CLIFF_MASK and cliff_masks_cache is not None and short in cliff_masks_cache:
            cliff_keep_full = cliff_masks_cache[short].astype(bool)
            mask = mask & cliff_keep_full
            n_dropped_endpt = (cliff_keep_full == 0).sum()
        elif ENABLE_CLIFF_MASK and log_train[short].notna().sum() >= 100:
            keep, info = detect_cliffs(
                log_train.loc[log_train[short].notna(), "SMILES"].tolist(),
                log_train.loc[log_train[short].notna(), short].to_numpy(),
                z_thresh=2.5, cluster_size=50)
            valid_idx = np.where(log_train[short].notna().to_numpy())[0]
            full_keep = np.ones(len(log_train), dtype=bool)
            full_keep[valid_idx[~keep]] = False
            mask = mask & full_keep
            n_dropped_endpt = info.get("n_dropped", 0)
        else:
            n_dropped_endpt = 0

        Xc = X_train[mask]
        yc = log_train.loc[mask, short].to_numpy()

        wc = np.ones(len(yc), dtype=float)

        # 3b) Optional external augmentation (off by default)
        if EXT_PROFILE != "none":
            ext = load_augmentation(short)
            forbid = set(train_df["SMILES"]).union(set(test_df["SMILES"]))
            ext = ext[~ext["SMILES"].isin(forbid)].reset_index(drop=True)
            if not ext.empty:
                idx = [ext_pool_lookup.get(s) for s in ext["SMILES"].tolist()]
                kept = [i for i, j in enumerate(idx) if j is not None]
                ext = ext.iloc[kept].reset_index(drop=True)
                idx = [j for j in idx if j is not None]
                Xe = ext_pool_full[idx]
                ye = ext["label"].to_numpy()
                we = ext["weight"].to_numpy()
                Xc = np.vstack([Xc, Xe])
                yc = np.concatenate([yc, ye])
                wc = np.concatenate([wc, we])

        if len(yc) < 20:
            print(f"      {short}: too few samples; skipping")
            submission[assay] = np.nan
            continue

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xc, yc, sample_weight=wc)
        pred_log = model.predict(X_test)
        pred = inverse_log_transform(pred_log, log_scale, multiplier)
        if log_scale:
            pred = np.clip(pred, a_min=0.0, a_max=None)
        submission[assay] = pred
        print(f"      {short:20s}  N_train={len(yc):5d}  "
              f"(after cliff drop {n_dropped_endpt})  "
              f"pred_mean={pred.mean():.3f}")

    sub_path = OUT_DIR / "submission_v2.csv"
    submission.to_csv(sub_path, index=False)
    print(f"[3/5] Wrote {sub_path}")

    # 4) Run official eval -------------------------------------------
    metrics_path = OUT_DIR / "official_eval_v2.csv"
    cmd = [sys.executable, "-m", "eval", str(sub_path),
           "--ground-truth", str(GROUND_TRUTH),
           "--output", str(metrics_path)]
    print(f"[4/5] Official eval -> {metrics_path}")
    proc = subprocess.run(cmd, cwd=EVAL_REPO, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"eval exited {proc.returncode}")

    # 5) Compare to v1 baseline (output/augmented/eval_baseline_n200.csv) ---
    v2 = pd.read_csv(metrics_path)
    base_path = ROOT / "output" / "augmented" / "eval_baseline_n200.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        m = base[["Endpoint", "mean_RAE", "mean_R2"]].merge(
            v2[["Endpoint", "mean_RAE", "mean_R2"]],
            on="Endpoint", suffixes=("_v1", "_v2"))
        m["ΔRAE"] = (m["mean_RAE_v2"] - m["mean_RAE_v1"]).round(3)
        m["ΔR2"] = (m["mean_R2_v2"] - m["mean_R2_v1"]).round(3)
        for c in m.columns:
            if c.startswith("mean_"):
                m[c] = m[c].round(3)
        comp_path = OUT_DIR / "v2_vs_baseline.csv"
        m.to_csv(comp_path, index=False)
        print(f"\n[5/5] Comparison vs v1 baseline (n=200, no aug):")
        print(m.to_string(index=False))
        macro = m[m["Endpoint"].str.contains("Macro", case=False, na=False)]
        if not macro.empty:
            r = macro.iloc[0]
            print()
            print(f"   *** v1 baseline   MA-RAE = {r['mean_RAE_v1']:.3f}")
            print(f"   *** v2            MA-RAE = {r['mean_RAE_v2']:.3f}  "
                  f"(Δ = {r['ΔRAE']:+.3f})")


if __name__ == "__main__":
    main()
