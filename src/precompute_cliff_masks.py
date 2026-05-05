"""Precompute SALI cliff masks per endpoint and save them.

Run once. Subsequent train_v2 invocations load these cached masks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from cliff_masking import detect_cliffs  # noqa: E402
from utils import (  # noqa: E402
    SUBMISSION_COLUMNS,
    get_conversion_dataframes,
    log_transform,
)


ROOT = THIS_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "v2"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    forward, _ = get_conversion_dataframes()

    masks = {}
    summary = []
    for assay in SUBMISSION_COLUMNS:
        log_scale, multiplier, short = forward[assay]
        y = log_transform(train[assay], log_scale, multiplier)
        keep_idx = y.notna()
        smiles = train.loc[keep_idx, "SMILES"].tolist()
        y_valid = y.loc[keep_idx].to_numpy()
        if len(y_valid) < 100:
            mask = np.ones(len(y_valid), dtype=bool)
            info = {"n_total": int(len(y_valid)), "n_dropped": 0,
                    "n_clusters": 0, "skipped": True}
            print(f"  {short:20s}: too few ({len(y_valid)}); skip cliff")
        else:
            mask, info = detect_cliffs(smiles, y_valid, z_thresh=2.5,
                                       cluster_size=50)
            print(f"  {short:20s}: {info['n_dropped']:4d}/{info['n_total']} "
                  f"dropped ({100*info['n_dropped']/info['n_total']:.1f}%)")
        # Reconstruct a full-length mask aligned to train_df's index
        full = np.ones(len(train), dtype=bool)
        # rows with NaN target are kept-from-cliff-perspective (they'll be filtered by .dropna anyway)
        valid_idx = np.where(keep_idx.to_numpy())[0]
        full[valid_idx[~mask]] = False
        masks[short] = full
        summary.append({"endpoint": short, **info})

    np.savez_compressed(OUT_DIR / "cliff_masks.npz", **{k: v for k, v in masks.items()})
    pd.DataFrame(summary).to_csv(OUT_DIR / "cliff_summary.csv", index=False)
    print(f"\nSaved cliff masks to {OUT_DIR/'cliff_masks.npz'}")
    print(f"Summary: {OUT_DIR/'cliff_summary.csv'}")


if __name__ == "__main__":
    main()
