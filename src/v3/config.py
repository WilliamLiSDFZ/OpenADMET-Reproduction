"""Central place for all paths and hyperparameters used across v3."""
from __future__ import annotations

import os
from pathlib import Path

# ----- Paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # the project root
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
GROUND_TRUTH_CSV = DATA_DIR / "test_ground_truth.csv"

V3_OUT = ROOT / "output" / "v3"


def _resolve_eval_repo() -> Path:
    """Locate the ExpansionRx-Challenge-Eval repo.

    Resolution order:
      1. ``EVAL_REPO`` env var (absolute path)
      2. Sibling directory of the project root  (../ExpansionRx-Challenge-Eval)
      3. Sibling directory at one level up        (../../ExpansionRx-Challenge-Eval)
      4. Common dev-machine paths (last-ditch fallback)

    Returns whatever it finds first; falls back to the sibling-of-project
    path even if it doesn't exist (so the error message is informative).
    """
    env = os.environ.get("EVAL_REPO")
    if env:
        return Path(env).expanduser().resolve()

    candidates = [
        ROOT.parent / "ExpansionRx-Challenge-Eval",
        ROOT.parent.parent / "ExpansionRx-Challenge-Eval",
        Path("/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval"),
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return candidates[0]   # informative default error if missing


EVAL_REPO = _resolve_eval_repo()

# ----- Endpoints ----------------------------------------------------------
SUBMISSION_COLUMNS = [
    "LogD",
    "KSOL",
    "MLM CLint",
    "HLM CLint",
    "Caco-2 Permeability Efflux",
    "Caco-2 Permeability Papp A>B",
    "MPPB",
    "MBPB",
    "MGMB",
]

# Per-endpoint preprocessing.
#
# Default zero-handling = "tutorial" (the official tutorial recipe:
# ``log10((x + 1) * multiplier)``).  We tried Inductive Bio's
# "half_min" and "ppb_floor" recipes and they made performance much
# worse on this dataset — see the design note in data.py for details.
ENDPOINT_PREP = {
    # assay name : (log_scale, multiplier, short_name, zero_handling)
    "LogD":                          (False, 1.0,  "LogD",             "passthrough"),
    "KSOL":                          (True,  1e-6, "LogS",             "tutorial"),
    "MLM CLint":                     (True,  1.0,  "Log_MLM_CLint",    "tutorial"),
    "HLM CLint":                     (True,  1.0,  "Log_HLM_CLint",    "tutorial"),
    "Caco-2 Permeability Efflux":    (True,  1.0,  "Log_Caco_ER",      "tutorial"),
    "Caco-2 Permeability Papp A>B":  (True,  1e-6, "Log_Caco_Papp_AB", "tutorial"),
    "MPPB":                          (True,  1.0,  "Log_Mouse_PPB",    "tutorial"),
    "MBPB":                          (True,  1.0,  "Log_Mouse_BPB",    "tutorial"),
    "MGMB":                          (True,  1.0,  "Log_Mouse_MPB",    "tutorial"),
}

# Task affinity grouping. Cluster compositions follow the OpenADME team
# (https://openadmet.ghost.io ‘Lessons Learned’) which used Spearman
# correlation on training labels to define multi-task groups for Chemprop.
TASK_GROUPS = {
    "solubility_binding": ["LogD", "KSOL", "MPPB", "MBPB", "MGMB"],
    "metabolism":          ["LogD", "MLM CLint", "HLM CLint"],
    "permeability":        ["LogD", "KSOL", "Caco-2 Permeability Papp A>B",
                            "Caco-2 Permeability Efflux"],
}
# LogD intentionally appears in every cluster -- it's the strongest single
# predictor of bulk physicochemical properties for the other endpoints.

# ----- Data splits --------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5

# ----- Featurization ------------------------------------------------------
FP_BITS = 2048
FP_RADIUS = 2
AVALON_BITS = 1024
USE_MORDRED = bool(int(os.environ.get("USE_MORDRED", "0")))   # off by default

# ----- LightGBM / XGBoost / CatBoost / RF base hyperparams ---------------
LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    min_child_samples=10, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbosity=0, tree_method="hist",
    n_jobs=-1,
)
CATBOOST_PARAMS = dict(
    iterations=500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, random_seed=RANDOM_STATE, verbose=0,
    thread_count=-1, allow_writing_files=False,
)
RF_PARAMS = dict(
    n_estimators=500, max_features="sqrt",
    min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1,
)

# ----- Chemprop ---------------------------------------------------------
CHEMPROP_PARAMS = dict(
    epochs=int(os.environ.get("CHEMPROP_EPOCHS", 50)),
    batch_size=int(os.environ.get("CHEMPROP_BATCH", 64)),
    init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
    depth=4, hidden_size=300, dropout=0.1,
    seed=RANDOM_STATE,
    ensemble_size=int(os.environ.get("CHEMPROP_ENS", 5)),
)

# ----- CheMeleon foundation-model pretraining (optional) -----------------
# Set the env var ``CHEMELEON`` to one of:
#   "1"  / "true" / "auto"   -> use chemprop's bundled CheMeleon downloader
#                               (chemprop>=2.2 ships it as ``mol_atom_bond.foundation``;
#                                older versions you must give an explicit path)
#   "/path/to/chemeleon.pt"  -> load this exact checkpoint
#   anything else / unset    -> do NOT use CheMeleon (default)
CHEMELEON = os.environ.get("CHEMELEON", "").strip()

# ----- TabPFN ------------------------------------------------------------
TABPFN_PARAMS = dict(
    # TabPFN v2 supports up to ~10000 samples and ~500 features. We feed
    # it RDKit-2d only (~210 features) to stay well under the limit.
    n_estimators=4,             # small ensemble of forward passes
    device=os.environ.get("TABPFN_DEVICE", "cuda"),  # or 'cpu' for CPU-only
)

# ----- Ensemble ----------------------------------------------------------
# Min weight in the ensemble (anything lower → drop the model). Solver: NNLS.
ENSEMBLE_MIN_WEIGHT = 0.02

# ----- Augmentation profile (proven helpful in v1 experiments) -----------
# 'none' / 'all' / 'selective'
EXT_PROFILE = os.environ.get("EXT_PROFILE", "selective")

# ----- Cliff masking -----------------------------------------------------
CLIFF_Z_THRESH = 2.5
# Apply only to endpoints with N >= 1500 (where v2 ablation showed it helps).
CLIFF_MIN_N = 1500
