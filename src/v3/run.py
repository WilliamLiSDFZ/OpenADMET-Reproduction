"""End-to-end v3 entry point.

  python -m v3.run                   # full pipeline
  python -m v3.run --skip-chemprop   # CPU-only ablation (LGBM/XGB/CB/RF/TabPFN)
  python -m v3.run --skip-tabpfn     # skip TabPFN
  python -m v3.run --classical-only  # just LGBM/XGB/CB/RF (≈ a strong v1)
  python -m v3.run --resume          # use saved per-model predictions where present

Outputs everything under ``output/v3/``:
    submission_v3.csv
    official_eval_v3.csv
    ensemble_weights.csv
    per_model_predictions/<model>_<endpoint>.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data import (
    inverse_log_endpoint, load_ground_truth, load_test, load_train,
    log_transform_endpoint, per_endpoint_views,
)
from .ensemble import apply_weights, fit_ensemble_weights
from .features import (
    cached_features, feature_dim, morgan_slice, rdkit_only_slice,
)
from .splits import time_window_split

PRED_DIR = cfg.V3_OUT / "per_model_predictions"


def _save_pred(model_name: str, endpoint_short: str, kind: str, arr: np.ndarray):
    """kind = 'val' (predictions on time-window validation) or 'test'."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{model_name}__{endpoint_short}__{kind}.npz",
                        X=arr)


def _load_pred(model_name: str, endpoint_short: str, kind: str):
    p = PRED_DIR / f"{model_name}__{endpoint_short}__{kind}.npz"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return np.load(p)["X"]
    except Exception:
        # Treat unreadable / empty cache as missing so we re-fit
        return None


def train_classical_models(train_df, test_df, X_full_train, X_full_test, args):
    """Per-endpoint LGBM / XGB / CatBoost / RF.  Time-window val predictions
    + final test predictions, both saved on disk."""
    from .models import (CatBoostModel, LightGBMModel, RandomForestModel,
                         XGBoostModel)

    # Time-window split for ensemble weight learning
    tr_idx, va_idx, _ = time_window_split(train_df["Molecule Name"].tolist(),
                                          train_pct=0.7, val_pct=0.15)
    print(f"  Time-window split: train={len(tr_idx)}, val={len(va_idx)}")

    classes = []
    if not args.skip_lgbm:     classes.append(LightGBMModel)
    if not args.skip_xgb:      classes.append(XGBoostModel)
    if not args.skip_catboost: classes.append(CatBoostModel)
    if not args.skip_rf:       classes.append(RandomForestModel)

    for assay in cfg.SUBMISSION_COLUMNS:
        log_scale, multiplier, short, zh = cfg.ENDPOINT_PREP[assay]
        y_full = log_transform_endpoint(train_df[assay], log_scale, multiplier, zh)
        keep = y_full.notna().to_numpy()

        for ModelCls in classes:
            mname = ModelCls.name
            cached_val = _load_pred(mname, short, "val")
            cached_test = _load_pred(mname, short, "test")
            if args.resume and cached_val is not None and cached_test is not None:
                continue

            print(f"  fit {mname:9s}  endpoint={short}")
            # ---- Time-window validation predictions ----
            tr_keep = np.intersect1d(tr_idx, np.where(keep)[0])
            va_keep = np.intersect1d(va_idx, np.where(keep)[0])
            if len(tr_keep) >= 30 and len(va_keep) >= 10:
                m = ModelCls()
                m.fit(X_full_train[tr_keep], y_full.iloc[tr_keep].to_numpy())
                vp = m.predict(X_full_train[va_keep])
                # Pad to full val_idx length (NaN where this endpoint was missing)
                pad = np.full(len(va_idx), np.nan)
                pos = np.searchsorted(va_idx, va_keep)
                pad[pos] = vp
                _save_pred(mname, short, "val", pad)

            # ---- Full-data → test predictions ----
            keep_full = np.where(keep)[0]
            m = ModelCls()
            m.fit(X_full_train[keep_full], y_full.iloc[keep_full].to_numpy())
            tp = m.predict(X_full_test)
            _save_pred(mname, short, "test", tp)


def train_tabpfn(train_df, test_df, X_full_train, X_full_test, args):
    if args.skip_tabpfn:
        return
    from .models.tabpfn_model import TabPFNModel
    s, e = rdkit_only_slice()
    Xtr_rd = X_full_train[:, s:e]
    Xte_rd = X_full_test[:, s:e]
    print(f"  TabPFN sees only RDKit-2d ({Xtr_rd.shape[1]} features)")

    tr_idx, va_idx, _ = time_window_split(train_df["Molecule Name"].tolist(),
                                          train_pct=0.7, val_pct=0.15)

    for assay in cfg.SUBMISSION_COLUMNS:
        log_scale, multiplier, short, zh = cfg.ENDPOINT_PREP[assay]
        y_full = log_transform_endpoint(train_df[assay], log_scale, multiplier, zh)
        keep = y_full.notna().to_numpy()

        if args.resume and _load_pred("tabpfn", short, "val") is not None \
                and _load_pred("tabpfn", short, "test") is not None:
            continue

        print(f"  fit tabpfn   endpoint={short}")
        tr_keep = np.intersect1d(tr_idx, np.where(keep)[0])
        va_keep = np.intersect1d(va_idx, np.where(keep)[0])

        if len(tr_keep) >= 30 and len(va_keep) >= 10:
            m = TabPFNModel()
            m.fit(Xtr_rd[tr_keep], y_full.iloc[tr_keep].to_numpy())
            vp = m.predict(Xtr_rd[va_keep])
            pad = np.full(len(va_idx), np.nan)
            pos = np.searchsorted(va_idx, va_keep)
            pad[pos] = vp
            _save_pred("tabpfn", short, "val", pad)

        keep_full = np.where(keep)[0]
        m = TabPFNModel()
        m.fit(Xtr_rd[keep_full], y_full.iloc[keep_full].to_numpy())
        tp = m.predict(Xte_rd)
        _save_pred("tabpfn", short, "test", tp)


def train_chemprop(train_df, test_df, args):
    if args.skip_chemprop:
        return

    short_names = [cfg.ENDPOINT_PREP[a][2] for a in cfg.SUBMISSION_COLUMNS]
    all_test_cached = all(
        _load_pred("chemprop", s, "test") is not None for s in short_names)
    all_val_cached = all(
        _load_pred("chemprop", s, "val") is not None for s in short_names)

    if args.resume and all_test_cached and all_val_cached:
        print("  Chemprop: all test+val preds cached, skipping")
        return

    # Time-window train/val split — chemprop must train ONLY on tr_idx so that
    # its predictions on va_idx are leak-free for ensemble weighting.
    tr_idx, va_idx, _ = time_window_split(
        train_df["Molecule Name"].tolist(), train_pct=0.7, val_pct=0.15)

    from .models.chemprop_model import train_all_clusters
    print(f"  Chemprop: training on {len(tr_idx)} time-window-train molecules, "
          f"holding out {len(va_idx)} for ensemble weighting")
    averaged = train_all_clusters(
        train_df, test_df,
        seeds=list(range(cfg.CHEMPROP_PARAMS["ensemble_size"])),
        train_indices=tr_idx,
        val_indices=va_idx,
    )
    # averaged is keyed "<short>__test" / "<short>__val"
    for key, vals in averaged.items():
        if key.endswith("__test"):
            short = key[: -len("__test")]
            _save_pred("chemprop", short, "test", vals)
        elif key.endswith("__val"):
            short = key[: -len("__val")]
            # The ensemble layer expects val pred arrays of length len(va_idx);
            # our chemprop val preds are already that length and aligned to va_idx.
            _save_pred("chemprop", short, "val", vals)


def fit_per_endpoint_ensemble(train_df, test_df):
    """Build the final ensemble per endpoint using time-window val predictions."""
    tr_idx, va_idx, _ = time_window_split(train_df["Molecule Name"].tolist(),
                                          train_pct=0.7, val_pct=0.15)
    weights_per_endpoint = {}
    submission = pd.DataFrame({"Molecule Name": test_df["Molecule Name"]})

    for assay in cfg.SUBMISSION_COLUMNS:
        log_scale, multiplier, short, zh = cfg.ENDPOINT_PREP[assay]
        y_full = log_transform_endpoint(train_df[assay], log_scale, multiplier, zh)
        keep = y_full.notna().to_numpy()
        va_keep = np.intersect1d(va_idx, np.where(keep)[0])

        # Collect per-model val + test preds
        val_preds = {}
        test_preds = {}
        for mname in ("lgbm", "xgb", "catboost", "rf", "tabpfn", "chemprop"):
            v = _load_pred(mname, short, "val")
            t = _load_pred(mname, short, "test")
            if t is None:
                continue
            test_preds[mname] = t
            if v is not None:
                # restrict to entries that aren't NaN (i.e. compounds that had
                # this endpoint measured)
                val_preds[mname] = v
        if not test_preds:
            print(f"  WARN no models for {short}")
            submission[assay] = np.nan
            continue

        # Fit weights against time-window val labels, fallback to equal weights
        if val_preds and len(va_keep) >= 10:
            usable = {k: v for k, v in val_preds.items() if not np.isnan(v).all()}
            if usable:
                pos = np.searchsorted(va_idx, va_keep)
                gt = y_full.iloc[va_keep].to_numpy()
                # Re-align val preds to only the kept indices
                stacks = []
                names = []
                for k, v in usable.items():
                    if not np.isnan(v[pos]).any():
                        stacks.append(v[pos])
                        names.append(k)
                if len(stacks) >= 1:
                    weights = fit_ensemble_weights(
                        {n: s for n, s in zip(names, stacks)}, gt
                    )
                    for k in test_preds:
                        weights.setdefault(k, 0.0)
                else:
                    weights = {k: 1.0 / len(test_preds) for k in test_preds}
            else:
                weights = {k: 1.0 / len(test_preds) for k in test_preds}
        else:
            weights = {k: 1.0 / len(test_preds) for k in test_preds}

        weights_per_endpoint[short] = weights
        log_pred = apply_weights(test_preds, weights)
        # Inverse-log per-endpoint
        pred = inverse_log_endpoint(log_pred, assay) if log_scale else log_pred
        if log_scale:
            pred = np.clip(pred, a_min=0.0, a_max=None)
        submission[assay] = pred
        print(f"  {short:20s} weights = "
              + " ".join(f"{k}:{v:.2f}" for k, v in weights.items() if v > 0))

    cfg.V3_OUT.mkdir(parents=True, exist_ok=True)
    sub_path = cfg.V3_OUT / "submission_v3.csv"
    submission.to_csv(sub_path, index=False)
    weights_path = cfg.V3_OUT / "ensemble_weights.csv"
    pd.DataFrame(
        [{"endpoint": ep, **w} for ep, w in weights_per_endpoint.items()]
    ).to_csv(weights_path, index=False)
    print(f"\n  Wrote {sub_path}")
    print(f"  Wrote {weights_path}")
    return sub_path


def run_official_eval(submission_csv: Path) -> Path | None:
    """Try the official scorer; if its repo isn't found, point the user at
    the manual command so the partial run isn't lost."""
    metrics_path = cfg.V3_OUT / "official_eval_v3.csv"
    if not Path(cfg.EVAL_REPO).is_dir():
        print(f"\n  WARN: official-eval repo not found at {cfg.EVAL_REPO}.")
        print("  The submission is already written, just score it manually:")
        print(f"    cd <your ExpansionRx-Challenge-Eval clone>")
        print(f"    python -m eval {submission_csv} \\")
        print(f"        --ground-truth {cfg.GROUND_TRUTH_CSV} \\")
        print(f"        --output {metrics_path}")
        print("\n  Or set EVAL_REPO=/path/to/ExpansionRx-Challenge-Eval and re-run.")
        return None

    cmd = [sys.executable, "-m", "eval", str(submission_csv),
           "--ground-truth", str(cfg.GROUND_TRUTH_CSV),
           "--output", str(metrics_path)]
    print("Running official eval ↓")
    print("  $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cfg.EVAL_REPO, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n  WARN: eval failed: {e}")
        print("  Submission file is still at:", submission_csv)
        return None
    res = pd.read_csv(metrics_path)
    print(res[["Endpoint", "mean_RAE", "mean_R2"]].to_string(index=False))
    macro = res[res["Endpoint"].str.contains("Macro", case=False, na=False)]
    if not macro.empty:
        r = macro.iloc[0]
        print(f"\n   *** v3 MA-RAE = {r['mean_RAE']:.3f}  R² = {r['mean_R2']:+.3f}")
    return metrics_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lgbm",     action="store_true")
    ap.add_argument("--skip-xgb",      action="store_true")
    ap.add_argument("--skip-catboost", action="store_true")
    ap.add_argument("--skip-rf",       action="store_true")
    ap.add_argument("--skip-tabpfn",   action="store_true")
    ap.add_argument("--skip-chemprop", action="store_true")
    ap.add_argument("--classical-only", action="store_true",
                    help="Skip both Chemprop AND TabPFN")
    ap.add_argument("--resume", action="store_true",
                    help="reuse cached per-model predictions on disk")
    ap.add_argument("--mordred", action="store_true",
                    help="include Mordred descriptors (~5 min extra featurization)")
    args = ap.parse_args()
    if args.classical_only:
        args.skip_chemprop = args.skip_tabpfn = True

    cfg.V3_OUT.mkdir(parents=True, exist_ok=True)
    print(f"v3 output dir: {cfg.V3_OUT}")

    print("\n[1/5] Loading data + featurizing")
    train_df = load_train()
    test_df = load_test()
    use_mord = args.mordred
    X_train = cached_features("train", train_df["SMILES"].tolist(), use_mordred=use_mord)
    X_test = cached_features("test", test_df["SMILES"].tolist(), use_mordred=use_mord)
    print(f"  train={X_train.shape}, test={X_test.shape}, dim={feature_dim(use_mord)}")

    print("\n[2/5] Classical learners (LGBM / XGB / CatBoost / RF)")
    train_classical_models(train_df, test_df, X_train, X_test, args)

    print("\n[3/5] TabPFN")
    train_tabpfn(train_df, test_df, X_train, X_test, args)

    print("\n[4/5] Chemprop multi-task (T4 GPU)")
    train_chemprop(train_df, test_df, args)

    print("\n[5/5] Per-endpoint weighted ensemble")
    sub_path = fit_per_endpoint_ensemble(train_df, test_df)
    if cfg.GROUND_TRUTH_CSV.exists():
        run_official_eval(sub_path)
    else:
        print("  (No ground truth at "
              f"{cfg.GROUND_TRUTH_CSV} -- skipping official eval)")


if __name__ == "__main__":
    main()
