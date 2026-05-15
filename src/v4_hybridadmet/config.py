"""All hyperparameters + paths for v4_hybridadmet. Mirrors HybridADMET's
spec where they reported it, with sensible PyTorch defaults elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

# Reuse v3's data definitions and endpoint metadata so we don't duplicate
# config drift across versions
from ..v3.config import (
    DATA_DIR, ENDPOINT_PREP, GROUND_TRUTH_CSV, RANDOM_STATE,
    SUBMISSION_COLUMNS, TEST_CSV, TRAIN_CSV,
)

ROOT = Path(__file__).resolve().parents[2]
V4_OUT = ROOT / "output" / "v4_hybridadmet"
EVAL_REPO = Path(os.environ.get(
    "EVAL_REPO",
    "/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval",
))

# ----- HybridADMET §"Training Strategy" — 9 per-endpoint multitask configs --
# After de-duplication these are 5 unique target-tuples.
PER_ENDPOINT_TARGETS = {
    "LogD":                          ["LogD"],
    "KSOL":                          ["LogD", "KSOL", "MLM CLint", "HLM CLint",
                                      "Caco-2 Permeability Efflux",
                                      "Caco-2 Permeability Papp A>B"],
    "MLM CLint":                     list(SUBMISSION_COLUMNS),     # all 9
    "HLM CLint":                     ["LogD", "KSOL", "MLM CLint", "HLM CLint",
                                      "Caco-2 Permeability Efflux",
                                      "Caco-2 Permeability Papp A>B", "MPPB"],
    "Caco-2 Permeability Efflux":    ["LogD", "KSOL",
                                      "Caco-2 Permeability Efflux",
                                      "Caco-2 Permeability Papp A>B"],
    "Caco-2 Permeability Papp A>B":  ["LogD", "KSOL",
                                      "Caco-2 Permeability Efflux",
                                      "Caco-2 Permeability Papp A>B"],
    "MPPB":                          ["LogD", "KSOL", "MLM CLint", "HLM CLint",
                                      "Caco-2 Permeability Efflux",
                                      "Caco-2 Permeability Papp A>B", "MPPB"],
    "MBPB":                          list(SUBMISSION_COLUMNS),
    "MGMB":                          list(SUBMISSION_COLUMNS),
}


def _build_groups():
    seen = {}
    for ep, t in PER_ENDPOINT_TARGETS.items():
        key = tuple(t)
        if key not in seen:
            seen[key] = f"group_{len(seen):02d}_n{len(t)}"
    ep_to_g = {e: seen[tuple(t)] for e, t in PER_ENDPOINT_TARGETS.items()}
    g_to_targets = {g: list(k) for k, g in seen.items()}
    g_to_primary = {g: [e for e, gg in ep_to_g.items() if gg == g] for g in g_to_targets}
    return ep_to_g, g_to_targets, g_to_primary


ENDPOINT_TO_GROUP, GROUP_TO_TARGETS, GROUP_TO_PRIMARY = _build_groups()
# Reproduces HybridADMET: 5 unique groups -- (1,), (6,), (9,), (7,), (4,)

# ----- Branch dimensions ----------------------------------------------------
DIM_UNIMOL = 512        # Uni-Mol2 84M variant output dim
DIM_PAMNET = 128        # PAMNet hidden dim (matches their QM9 setup)
DIM_FP_OUT = 256        # fingerprint MLP output before concat

# ----- Fingerprint sizes (must match features/fingerprints.py) -------------
MACCS_BITS = 167
ERG_BITS = 315
PUBCHEM_BITS = 881
FP_INPUT_DIM = MACCS_BITS + ERG_BITS + PUBCHEM_BITS    # 1363

# ----- Multi-task MLP head --------------------------------------------------
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.2

# ----- Uni-Mol2 -------------------------------------------------------------
UNIMOL_FINETUNE = bool(int(os.environ.get("UNIMOL_FINETUNE", "1")))
UNIMOL_MODEL_VARIANT = os.environ.get("UNIMOL_MODEL", "84M")   # or "164M"

# ----- PAMNet ---------------------------------------------------------------
PAMNET_CFG = dict(
    dim=DIM_PAMNET,
    n_layer=int(os.environ.get("V4_PAMNET_LAYERS", "4")),
    cutoff_l=float(os.environ.get("V4_PAMNET_CUTOFF_L", "5.0")),    # Å, local
    cutoff_g=float(os.environ.get("V4_PAMNET_CUTOFF_G", "12.0")),   # Å, global
    flow="source_to_target",
)

# ----- Training (HybridADMET defaults) --------------------------------------
EPOCHS = int(os.environ.get("V4_EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("V4_BATCH", "32"))
LR = float(os.environ.get("V4_LR", "1e-4"))         # AdamW
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
ENSEMBLE_SEEDS = int(os.environ.get("V4_ENS", "5"))  # HybridADMET = 5
CLIP_GRAD = 1.0

# ----- I/O ------------------------------------------------------------------
V4_OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = V4_OUT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = V4_OUT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
