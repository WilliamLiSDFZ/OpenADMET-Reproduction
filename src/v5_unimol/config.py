"""Hyperparameters + paths for v5_unimol. Strips v4_hybridadmet's
PAMNet/fingerprint settings; keeps everything else identical so the
v4 → v5 comparison is a clean ablation."""
from __future__ import annotations

import os
from pathlib import Path

# Reuse v3's data definitions (endpoint metadata, paths) so we don't fork
# config drift across versions.
from ..v3.config import (
    DATA_DIR, ENDPOINT_PREP, GROUND_TRUTH_CSV, RANDOM_STATE,
    SUBMISSION_COLUMNS, TEST_CSV, TRAIN_CSV,
)

ROOT = Path(__file__).resolve().parents[2]
V5_OUT = ROOT / "output" / "v5_unimol"
EVAL_REPO = Path(os.environ.get(
    "EVAL_REPO",
    "/sessions/bold-beautiful-faraday/mnt/python/ExpansionRx-Challenge-Eval",
))

# ----- HybridADMET §"Training Strategy" — 9 per-endpoint multitask configs --
# After de-duplication these are 5 unique target-tuples. **Identical to v4.**
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
# Reproduces HybridADMET dedup: 5 unique groups -- (1,), (6,), (9,), (7,), (4,)

# ----- Branch dimension -----------------------------------------------------
DIM_UNIMOL = 512        # Uni-Mol2 84M variant output dim

# ----- Multi-task MLP head --------------------------------------------------
# Same as v4 to keep the v4 vs v5 comparison clean.
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.2

# ----- Uni-Mol2 -------------------------------------------------------------
UNIMOL_FINETUNE = bool(int(os.environ.get("UNIMOL_FINETUNE", "1")))
UNIMOL_MODEL_VARIANT = os.environ.get("UNIMOL_MODEL", "84M")   # or "164M"

# ----- Training (same defaults as v4) ---------------------------------------
EPOCHS = int(os.environ.get("V5_EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("V5_BATCH", "32"))
LR = float(os.environ.get("V5_LR", "1e-4"))            # AdamW
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
ENSEMBLE_SEEDS = int(os.environ.get("V5_ENS", "5"))    # HybridADMET = 5
CLIP_GRAD = 1.0

# ----- I/O ------------------------------------------------------------------
V5_OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = V5_OUT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = V5_OUT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
