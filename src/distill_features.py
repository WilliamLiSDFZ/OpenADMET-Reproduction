"""ADMET-AI distillation features.

The JCIM paper showed that **using ADMET-AI predictions as additional input
features** (concatenated to RDKit-2d / Mordred / fingerprints) gives a
statistically significant boost on every ADME endpoint. We can't run
admet-ai here (its checkpoints live on HuggingFace which is firewalled), so
we *distill* it: train tiny LightGBMs on the 2845 drugbank molecules where
admet_ai already published its predictions, then apply those distilled
models to every challenge SMILES to fabricate the same kind of feature
column.

End result: a (n_molecules, n_distill_endpoints=~50) float matrix that's
concatenated to v2 features at training time.

Cached at output/v2/distill_features_<train|test|external>.npz.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from features_v2 import featurize_smiles_batch_v2  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = THIS_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "v2"
DRUGBANK = DATA_DIR / "external" / "drugbank_admet_predictions.csv"

# ADMET-AI columns to distill (numeric, full coverage on drugbank). We
# include both regression-style (LogD, solubility, clearance, ...) and
# classification-style (probabilities) since LightGBM treats them all as
# real-valued features.
DISTILL_COLUMNS = [
    "logP", "tpsa", "QED", "Lipinski",
    "AMES", "BBB_Martins", "Bioavailability_Ma", "DILI", "HIA_Hou", "hERG",
    "PAMPA_NCATS", "Pgp_Broccatelli", "Skin_Reaction",
    "CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Substrate_CarbonMangels",
    "CYP2C9_Veith", "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Veith",
    "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Veith",
    "Carcinogens_Lagunin", "ClinTox",
    "Caco2_Wang", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ",
    "Half_Life_Obach", "HydrationFreeEnergy_FreeSolv", "LD50_Zhu",
    "Lipophilicity_AstraZeneca", "PPBR_AZ", "Solubility_AqSolDB", "VDss_Lombardo",
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

DISTILL_LGBM_PARAMS = dict(
    n_estimators=120,
    learning_rate=0.08,
    num_leaves=31,
    min_child_samples=8,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=4,
    random_state=0,
    verbose=-1,
    n_jobs=4,
)


def train_distill_models(verbose: bool = True):
    """Fit one LightGBM per ADMET-AI column on drugbank features."""
    from lightgbm import LGBMRegressor

    df = pd.read_csv(DRUGBANK)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    cols = [c for c in DISTILL_COLUMNS if c in df.columns]
    if verbose:
        print(f"Distilling {len(cols)} ADMET-AI columns from {len(df)} drugbank molecules")

    # Featurize drugbank SMILES once
    print("  Featurizing drugbank molecules (v2 features)...")
    X, _ = featurize_smiles_batch_v2(df["smiles"].tolist(), show_progress=False)
    print(f"  Shape: {X.shape}")

    models = {}
    for c in cols:
        y = df[c].astype(float).to_numpy()
        m = LGBMRegressor(**DISTILL_LGBM_PARAMS)
        m.fit(X, y)
        models[c] = m
    if verbose:
        print(f"  Fit {len(models)} distillation models")
    return models, cols


def distill_predict(models, cols, X: np.ndarray) -> np.ndarray:
    """Apply the dict-of-models to a feature matrix; return (n, n_endpoints)."""
    out = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        out[:, j] = models[c].predict(X)
    return out


def main():
    """Build distill features for train/test/external and save to disk."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models, cols = train_distill_models()

    for split in ("train", "test", "external"):
        feat_path = OUT_DIR / f"features_{split}_v2.npz"
        out_path = OUT_DIR / f"distill_{split}.npz"
        if not feat_path.exists():
            print(f"  Skipping {split}: {feat_path} not found")
            continue
        if out_path.exists():
            print(f"  Already exists: {out_path}")
            continue
        X = np.load(feat_path)["X"]
        D = distill_predict(models, cols, X)
        np.savez_compressed(out_path, X=D, columns=np.array(cols))
        print(f"  Wrote {out_path} (shape={D.shape})")


if __name__ == "__main__":
    main()
