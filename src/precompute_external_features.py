"""Pre-compute features for ALL unique external SMILES once.

Saves to output/augmented/features_external_pool.npz with:
    smiles : np.array of unique SMILES strings
    X      : (N, FEAT_DIM) float32 features
    smiles_to_idx : dict (str -> int) -- separate JSON for fast lookup

Run this once before train_augmented.py to amortize the 30+ second
fingerprinting cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from external_data import AUGMENTATION_LOADERS, load_augmentation  # noqa: E402
from features import featurize_smiles_batch  # noqa: E402

OUT_PATH = THIS_DIR.parent / "output" / "augmented" / "features_external_pool.npz"
INDEX_PATH = THIS_DIR.parent / "output" / "augmented" / "smiles_index.json"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Collect every unique SMILES from every endpoint loader
    all_smiles = set()
    for endpoint in AUGMENTATION_LOADERS:
        df = load_augmentation(endpoint)
        if not df.empty:
            all_smiles.update(df["SMILES"].tolist())
    smiles = sorted(all_smiles)
    print(f"Featurizing {len(smiles)} unique external SMILES")

    X, _ = featurize_smiles_batch(smiles, n_bits=2048, radius=2)
    print(f"Result shape: {X.shape}, dtype: {X.dtype}")

    np.savez_compressed(OUT_PATH, smiles=np.array(smiles), X=X)
    smi2idx = {s: i for i, s in enumerate(smiles)}
    with open(INDEX_PATH, "w") as f:
        json.dump(smi2idx, f)
    print(f"Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.1f} KB)")
    print(f"Wrote {INDEX_PATH}")


if __name__ == "__main__":
    main()
