"""train_v2_compose: ablation-friendly version of train_v2.

Lets you toggle each improvement individually so you can see which one helped.
Designed to use the v1 base features (RDKit + Morgan, 2265 dim) as the
backbone and add things on top, since adding Avalon to small-N endpoints
was a regression in the first v2 attempt.

Toggles via env vars (default value in parens):
  USE_AVALON=0|1     (0)  add Avalon fingerprints (1024 bits)
  USE_DISTILL=0|1    (0)  add the 45 ADMET-AI distillation features
  USE_CLIFF=0|1      (0)  drop SALI cliffs (precomputed at output/v2/cliff_masks.npz)
  EXT_PROFILE        (none) external row augmentation: none|all|selective
  N_ESTIMATORS       (200)
  TAG=<name>         optional run tag, used for output filenames

Outputs (under output/v2/):
  submission_<tag>.csv
  official_eval_<tag>.csv
  comparison_<tag>.csv      Δ vs v1 baseline (output/augmented/eval_baseline_n200.csv)
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
from utils import (  # noqa: E402
    SUBMISSION_COLUMNS,
    get_conversion_dataframes,
    inverse_log_transform,
    log_transform,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = THIS_DIR.parent
DATA_DIR = ROOT / "data"
V1_OUT = ROOT / "output"
V2_OUT = ROOT / "output" / "v2"
EVAL_REPO = Path("/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval")
GROUND_TRUTH = DATA_DIR / "test_ground_truth.csv"

USE_AVALON = bool(int(os.environ.get("USE_AVALON", "0")))
USE_DISTILL = bool(int(os.environ.get("USE_DISTILL", "0")))
USE_CLIFF = bool(int(os.environ.get("USE_CLIFF", "0")))
EXT_PROFILE = os.environ.get("EXT_PROFILE", "none").lower()
TAG = os.environ.get("TAG", "v2")

LGBM_PARAMS = dict(
    n_estimators=int(os.environ.get("N_ESTIMATORS", 200)),
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1,
    n_jobs=int(os.environ.get("N_JOBS", 4)),
)


def _load(p, key="X"):
    return np.load(p, allow_pickle=True)[key]


def main():
    V2_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Run tag: {TAG}")
    print(f"  use_avalon={USE_AVALON}  use_distill={USE_DISTILL}  "
          f"use_cliff={USE_CLIFF}  ext_profile={EXT_PROFILE}  "
          f"n_est={LGBM_PARAMS['n_estimators']}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Always start from v1 base (RDKit + Morgan, 2265 dim)
    X_train_base = _load(V1_OUT / "features_train.npz")
    X_test_base = _load(V1_OUT / "features_test.npz")
    print(f"  v1 base features: train={X_train_base.shape}, test={X_test_base.shape}")

    Xtr_parts, Xte_parts = [X_train_base], [X_test_base]

    if USE_AVALON:
        # Avalon-only feature: extract from v2 features by slicing
        # v2 layout = [RDKit(210), Morgan(2048), Avalon(1024)]
        v2_train = _load(V2_OUT / "features_train_v2.npz")
        v2_test = _load(V2_OUT / "features_test_v2.npz")
        # Take only the Avalon slice (indices 210+2048 .. 210+2048+1024)
        avalon_start = 210 + 2048
        avalon_end = avalon_start + 1024
        Xtr_parts.append(v2_train[:, avalon_start:avalon_end])
        Xte_parts.append(v2_test[:, avalon_start:avalon_end])
        print(f"  +Avalon: +{Xtr_parts[-1].shape[1]} features")

    if USE_DISTILL:
        Xtr_parts.append(_load(V2_OUT / "distill_train.npz"))
        Xte_parts.append(_load(V2_OUT / "distill_test.npz"))
        print(f"  +Distill: +{Xtr_parts[-1].shape[1]} features")

    X_train = np.concatenate(Xtr_parts, axis=1)
    X_test = np.concatenate(Xte_parts, axis=1)
    print(f"  Final feature dim: {X_train.shape[1]}")

    cliff_masks_cache = None
    if USE_CLIFF:
        cp = V2_OUT / "cliff_masks.npz"
        if cp.exists():
            cliff_masks_cache = dict(np.load(cp))
            print(f"  Loaded {len(cliff_masks_cache)} precomputed cliff masks")
        else:
            print("  WARN: USE_CLIFF=1 but cliff_masks.npz missing -- no masking")

    forward, _ = get_conversion_dataframes()
    log_train = train_df[["SMILES", "Molecule Name"]].copy()
    log_columns = []
    for assay in SUBMISSION_COLUMNS:
        log_scale, multiplier, short = forward[assay]
        log_train[short] = log_transform(train_df[assay], log_scale, multiplier)
        log_columns.append((assay, short, log_scale, multiplier))

    # External augmentation pool (loaded only if needed)
    ext_pool_full = None
    ext_pool_lookup = None
    if EXT_PROFILE != "none":
        if not (V2_OUT / "features_external_v2.npz").exists():
            raise SystemExit("Missing external v2 features.")
        ext_X = _load(V2_OUT / "features_external_v2.npz")
        ext_smis = _load(V2_OUT / "features_external_v2.npz", key="smiles")
        # Build same column composition as challenge features
        # ext_X has v2 layout (RDKit+Morgan+Avalon = 3289)
        # We need to align with our X_train features
        ext_parts = [ext_X[:, :2265]]  # RDKit + Morgan (v1 base)
        if USE_AVALON:
            ext_parts.append(ext_X[:, avalon_start:avalon_end])
        if USE_DISTILL:
            ext_parts.append(_load(V2_OUT / "distill_external.npz"))
        ext_pool_full = np.concatenate(ext_parts, axis=1)
        ext_pool_lookup = {s: i for i, s in enumerate(ext_smis)}
        os.environ["EXT_PROFILE"] = EXT_PROFILE  # let load_augmentation see it
        print(f"  External pool: {ext_pool_full.shape}")

    from lightgbm import LGBMRegressor
    submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})

    print(f"[Train] Per-endpoint")
    for assay, short, log_scale, multiplier in log_columns:
        mask = log_train[short].notna().to_numpy()
        # Cliff drop
        if cliff_masks_cache is not None and short in cliff_masks_cache:
            mask = mask & cliff_masks_cache[short].astype(bool)

        Xc = X_train[mask]
        yc = log_train.loc[mask, short].to_numpy()
        wc = np.ones(len(yc), dtype=float)

        # External augmentation
        if EXT_PROFILE != "none" and ext_pool_full is not None:
            ext = load_augmentation(short)
            forbid = set(train_df["SMILES"]).union(set(test_df["SMILES"]))
            ext = ext[~ext["SMILES"].isin(forbid)].reset_index(drop=True)
            if not ext.empty:
                idx = [ext_pool_lookup.get(s) for s in ext["SMILES"].tolist()]
                kept = [i for i, j in enumerate(idx) if j is not None]
                ext = ext.iloc[kept].reset_index(drop=True)
                idx = [j for j in idx if j is not None]
                Xc = np.vstack([Xc, ext_pool_full[idx]])
                yc = np.concatenate([yc, ext["label"].to_numpy()])
                wc = np.concatenate([wc, ext["weight"].to_numpy()])

        if len(yc) < 20:
            submission[assay] = np.nan
            continue
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(Xc, yc, sample_weight=wc)
        pred = inverse_log_transform(m.predict(X_test), log_scale, multiplier)
        if log_scale:
            pred = np.clip(pred, a_min=0.0, a_max=None)
        submission[assay] = pred

    sub_path = V2_OUT / f"submission_{TAG}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"[Eval] Wrote {sub_path}")

    metrics_path = V2_OUT / f"official_eval_{TAG}.csv"
    cmd = [sys.executable, "-m", "eval", str(sub_path),
           "--ground-truth", str(GROUND_TRUTH), "--output", str(metrics_path)]
    proc = subprocess.run(cmd, cwd=EVAL_REPO, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    res = pd.read_csv(metrics_path)
    base = pd.read_csv(V1_OUT / "augmented" / "eval_baseline_n200.csv")
    m = base[["Endpoint", "mean_RAE", "mean_R2"]].merge(
        res[["Endpoint", "mean_RAE", "mean_R2"]],
        on="Endpoint", suffixes=("_base", f"_{TAG}"))
    m[f"ΔRAE"] = (m[f"mean_RAE_{TAG}"] - m["mean_RAE_base"]).round(3)
    m[f"ΔR2"] = (m[f"mean_R2_{TAG}"] - m["mean_R2_base"]).round(3)
    for c in m.columns:
        if c.startswith("mean_"):
            m[c] = m[c].round(3)
    comp_path = V2_OUT / f"comparison_{TAG}.csv"
    m.to_csv(comp_path, index=False)
    print(m.to_string(index=False))
    macro = m[m["Endpoint"].str.contains("Macro", case=False, na=False)]
    if not macro.empty:
        r = macro.iloc[0]
        print()
        print(f"   *** baseline (v1 n=200)  MA-RAE = {r['mean_RAE_base']:.3f}")
        print(f"   *** {TAG}              MA-RAE = {r[f'mean_RAE_{TAG}']:.3f}  "
              f"(Δ = {r['ΔRAE']:+.3f})")


if __name__ == "__main__":
    main()
