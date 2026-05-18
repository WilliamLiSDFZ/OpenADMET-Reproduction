"""Single-representation Dataset for v5_unimol.

vs v4_hybridadmet/data.py this drops PyG Data construction and PaDEL
fingerprint loading. We only need:

  1. Uni-Mol2 input pack (atom token ids + 3D coords + pairwise dists +
     edge-type matrix), built lazily inside ``collate_unimol`` since the
     dictionary file (``mol.dict.txt``) only gets downloaded after the first
     UniMolModel construction.
  2. Multi-task target vector (NaN-aware mask) — identical to v4.

3D conformers come from RDKit ETKDGv3 + MMFF (same recipe as v4 / Uni-Mol2's
own preprocessing). Cached to a single pickle so a Docker restart doesn't
re-embed all 5326 molecules.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as cfg
# Reuse v4's molecule featurization helpers — they're the right code,
# we just don't need the PyG part. We call smiles_to_3d directly and skip
# smiles_to_pyg_data.
from ..v4_hybridadmet.data import (
    pack_unimol_input,    # Uni-Mol2 input dict (token, coord, dist, edge_type)
    smiles_to_3d,         # SMILES -> (atomic numbers, 3D coords) via ETKDG+MMFF
)
# Reuse v3's log-transform; matches "follow the organizer's tutorial pipeline"
# instruction in HybridADMET §Preprocessing.
from ..v3.data import log_transform_endpoint


# ----- Build per-molecule feature pack (no PyG, no fp) ----------------------
def build_molecule_features(smiles: str, seed: int = 42) -> dict | None:
    """Return a minimal pack: just SMILES + atomic numbers + 3D coords.

    Uni-Mol2 packing is deferred to collate time because pack_unimol_input
    needs the atom dictionary (only available after first UniMolModel construct).
    """
    atoms, coords = smiles_to_3d(smiles, seed=seed)
    if atoms is None:
        return None
    return {
        "smiles": smiles,
        "atoms": atoms,
        "coords": coords,
    }


def precompute_features(smiles_list: List[str],
                        cache_path: Path | None = None,
                        seed: int = 42,
                        show_progress: bool = True) -> List[dict | None]:
    """Precompute 3D conformer for each SMILES. Cached to ``cache_path``.

    Per-mol cost is ~0.3 s, so 5326 molecules ≈ 25 min. Caching saves all
    the time on Docker restarts.
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"  Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(smiles_list, desc="3D conformer embed (ETKDG+MMFF)")
        except ImportError:
            it = smiles_list
    else:
        it = smiles_list
    feats = [build_molecule_features(s, seed=seed) for s in it]
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(feats, f)
        print(f"  Cached {cache_path}")
    return feats


# ----- PyTorch Dataset ------------------------------------------------------
class UniMolDataset(Dataset):
    """Returns (mol_feature_dict, target, mask) per sample.

    Drops fingerprints relative to v4's HybridADMETDataset.

    Args:
        molecule_features: list of dicts from ``precompute_features``
                           (may include None for SMILES that failed embed).
        targets:           (N, n_tasks) float32 array (NaN -> filled by build_targets).
        target_mask:       (N, n_tasks) float32 array (1 where target is valid).
    """
    def __init__(self, molecule_features, targets, target_mask):
        self.molfs = molecule_features
        self.y     = torch.as_tensor(targets, dtype=torch.float32)
        self.mask  = torch.as_tensor(target_mask, dtype=torch.float32)
        # Indices where 3D conformer generation succeeded
        self.valid_indices = [i for i, m in enumerate(self.molfs) if m is not None]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        return {
            "molf": self.molfs[i],
            "y": self.y[i],
            "mask": self.mask[i],
        }


def collate_unimol(batch_items, device=None):
    """Custom collate: builds the padded Uni-Mol2 batch + stacked y/mask.

    Same Uni-Mol2 packing as v4's collate_hybrid; we just don't build a
    PyGBatch or stack fingerprints. Cheaper per-batch as a bonus.
    """
    # Build per-molecule Uni-Mol2 dict NOW (lazy, after weights downloaded)
    unimol_dicts = [pack_unimol_input(it["molf"]["atoms"], it["molf"]["coords"])
                    for it in batch_items]
    max_n = max(d["n_atoms"] for d in unimol_dicts)
    B = len(unimol_dicts)
    src_tokens   = torch.zeros((B, max_n), dtype=torch.long)
    src_coord    = torch.zeros((B, max_n, 3), dtype=torch.float32)
    src_distance = torch.zeros((B, max_n, max_n), dtype=torch.float32)
    src_edge_type = torch.zeros((B, max_n, max_n), dtype=torch.long)
    for b, d in enumerate(unimol_dicts):
        n = d["n_atoms"]
        src_tokens[b, :n] = d["src_tokens"]
        src_coord[b, :n] = d["src_coord"]
        src_distance[b, :n, :n] = d["src_distance"]
        src_edge_type[b, :n, :n] = d["src_edge_type"]
    unimol_batch = {
        "src_tokens": src_tokens,
        "src_coord": src_coord,
        "src_distance": src_distance,
        "src_edge_type": src_edge_type,
    }

    y    = torch.stack([it["y"]    for it in batch_items])
    mask = torch.stack([it["mask"] for it in batch_items])

    if device is not None:
        unimol_batch = {k: v.to(device) for k, v in unimol_batch.items()}
        y, mask = y.to(device), mask.to(device)

    return {
        "unimol_batch": unimol_batch,
        "y": y,
        "mask": mask,
    }


# ----- Build targets/mask matrices ------------------------------------------
def build_targets(train_df: pd.DataFrame, target_endpoints: List[str]):
    """Returns (targets (N, T), mask (N, T)) for the given endpoint subset,
    log-transformed via the tutorial recipe. Identical to v4.build_targets.
    """
    n = len(train_df)
    T = len(target_endpoints)
    targets = np.zeros((n, T), dtype=np.float32)
    mask = np.zeros((n, T), dtype=np.float32)
    for j, assay in enumerate(target_endpoints):
        log_scale, mult, _short, zh = cfg.ENDPOINT_PREP[assay]
        y = log_transform_endpoint(train_df[assay], log_scale, mult, zh)
        keep = y.notna().to_numpy()
        targets[keep, j] = y.to_numpy()[keep]
        mask[keep, j] = 1.0
    return targets, mask
