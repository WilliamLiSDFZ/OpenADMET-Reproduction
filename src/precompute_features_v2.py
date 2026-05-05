"""Pre-compute v2 features (RDKit-2d + Morgan + Avalon + Mordred) for
challenge train + test, plus the external SMILES pool. Run once.

Outputs (under output/v2/):
    features_train_v2.npz    challenge train features
    features_test_v2.npz     challenge test features
    features_external_v2.npz pool of unique external SMILES + their features
    smiles_index_v2.json     SMILES -> row index for the external pool
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from external_data import AUGMENTATION_LOADERS, load_augmentation  # noqa: E402
from features_v2 import featurize_smiles_batch_v2, feature_dim_v2  # noqa: E402

ROOT = THIS_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "v2"

# Run only the requested split (so each call fits inside our timeout budget):
#   USE_MORDRED=1   include the 1620 Mordred descriptors (slower, parallel via mordred)
#   ONLY=train|test|external|all   default 'all'
USE_MORDRED = bool(int(__import__("os").environ.get("USE_MORDRED", "0")))
ONLY = __import__("os").environ.get("ONLY", "all")


def _do_split(name: str, smiles_list, out_path: Path):
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return
    print(f"Featurizing {len(smiles_list)} {name} molecules"
          f" (use_mordred={USE_MORDRED})")
    X, _ = featurize_smiles_batch_v2(smiles_list, use_mordred=USE_MORDRED)
    np.savez_compressed(out_path, X=X, smiles=np.array(smiles_list))
    print(f"  Wrote {out_path} (shape={X.shape}, {out_path.stat().st_size/1024:.0f} KB)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Total feature dim: {feature_dim_v2(USE_MORDRED)}  (USE_MORDRED={USE_MORDRED}, ONLY={ONLY})")

    if ONLY in ("train", "all"):
        train = pd.read_csv(DATA_DIR / "train.csv")
        _do_split("train", train["SMILES"].tolist(),
                  OUT_DIR / "features_train_v2.npz")

    if ONLY in ("test", "all"):
        test = pd.read_csv(DATA_DIR / "test.csv")
        _do_split("test", test["SMILES"].tolist(),
                  OUT_DIR / "features_test_v2.npz")

    if ONLY in ("external", "all"):
        ext_path = OUT_DIR / "features_external_v2.npz"
        idx_path = OUT_DIR / "smiles_index_v2.json"
        if ext_path.exists():
            print(f"Already exists: {ext_path}")
        else:
            all_smiles = set()
            for ep in AUGMENTATION_LOADERS:
                df = load_augmentation(ep)
                if not df.empty:
                    all_smiles.update(df["SMILES"].tolist())
            smiles = sorted(all_smiles)
            _do_split("external", smiles, ext_path)
            with open(idx_path, "w") as f:
                json.dump({s: i for i, s in enumerate(smiles)}, f)
            print(f"  Wrote {idx_path}")


if __name__ == "__main__":
    main()
