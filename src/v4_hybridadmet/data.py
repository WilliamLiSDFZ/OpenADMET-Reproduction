"""Multi-representation Dataset.

For each molecule we precompute and cache:

  1. Uni-Mol2 input dict (atom tokens + 3D coords + pairwise dists)
  2. torch_geometric Data (atomic numbers + 3D coords + edge_index for PAMNet)
  3. fingerprint vector (1363 floats)
  4. multi-task target vector (NaN-aware mask)

3D conformers are generated once via RDKit ETKDGv3 (the same recipe Uni-Mol2
uses in its own preprocessing). Both Uni-Mol2 and PAMNet share the same
coordinates.

Generation is expensive (5326 molecules × ~0.3 s each ≈ 25 min); we cache
the whole feature pack to ``cache/molecule_features.pkl``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import Dataset

from . import config as cfg
# Reuse v3's log-transform; mirrors HybridADMET's "follow the organizer's
# tutorial pipeline" instruction (= log10((x+1)*multiplier)).
from ..v3.data import log_transform_endpoint

RDLogger.DisableLog("rdApp.*")


# ----- 3D conformer generation ----------------------------------------------
def smiles_to_3d(smiles: str, seed: int = 42):
    """SMILES -> (atomic_numbers[N], coords[N,3]) via RDKit ETKDGv3.
    Returns (None, None) on failure.
    """
    if not isinstance(smiles, str) or not smiles:
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    try:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            # Sometimes ETKDG fails; fall back to a less constrained method
            cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
        if cid == -1:
            return None, None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return None, None
    atoms = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    conf = mol.GetConformer()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return atoms, coords


def smiles_to_pyg_data(smiles: str, atoms, coords):
    """Build a torch_geometric.data.Data for PAMNet.

    PAMNet wants ``x`` = atomic numbers, ``pos`` = 3D coords, ``edge_index``
    = covalent bonds (will be augmented with radius-graph at forward time).
    """
    from torch_geometric.data import Data
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    mol = Chem.AddHs(mol)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.zeros((2, 0), dtype=torch.long))
    return Data(
        x=torch.tensor(atoms, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        edge_index=edge_index,
    )


# ----- Uni-Mol2 input packing -----------------------------------------------
def pack_unimol_input(atoms: np.ndarray, coords: np.ndarray) -> dict:
    """Build the per-molecule dict Uni-Mol2's collator expects.

    The exact keys depend on the unimol_tools version. We provide the
    minimum subset (``src_tokens``, ``src_coord``, ``src_distance``,
    ``src_edge_type``) that the latest stable 2024-2025 release uses.
    """
    n = len(atoms)
    # Atom type tokens: use raw atomic numbers (Uni-Mol2's dict maps Z -> token).
    src_tokens = torch.tensor(atoms, dtype=torch.long)
    src_coord  = torch.tensor(coords, dtype=torch.float32)
    # Pairwise distance matrix (n, n)
    diff = src_coord.unsqueeze(0) - src_coord.unsqueeze(1)
    src_distance = (diff * diff).sum(dim=-1).clamp(min=1e-12).sqrt()
    # Edge type (atom-pair type): outer product of token ids
    src_edge_type = src_tokens.unsqueeze(0) * src_tokens.unsqueeze(1)
    return {
        "src_tokens": src_tokens,
        "src_coord": src_coord,
        "src_distance": src_distance,
        "src_edge_type": src_edge_type,
        "n_atoms": n,
    }


# ----- Build per-molecule feature pack --------------------------------------
def build_molecule_features(smiles: str, seed: int = 42) -> dict | None:
    atoms, coords = smiles_to_3d(smiles, seed=seed)
    if atoms is None:
        return None
    pyg_data = smiles_to_pyg_data(smiles, atoms, coords)
    unimol_in = pack_unimol_input(atoms, coords)
    return {
        "smiles": smiles,
        "atoms": atoms,
        "coords": coords,
        "pyg": pyg_data,
        "unimol": unimol_in,
    }


def precompute_features(smiles_list: List[str],
                        cache_path: Path | None = None,
                        seed: int = 42,
                        show_progress: bool = True) -> List[dict | None]:
    if cache_path is not None and Path(cache_path).exists():
        print(f"  Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(smiles_list, desc="3D + PyG + Uni-Mol2 pack")
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
class HybridADMETDataset(Dataset):
    """Returns (mol_feature_dict, fingerprint, target, mask) per sample.

    Args:
        molecule_features: list of dicts from ``precompute_features``
        fingerprints:      (N, 1363) float32 array
        targets:           (N, n_tasks) float32 array (NaN where missing)
        target_mask:       (N, n_tasks) float32 array (1 where target is valid)
    """
    def __init__(self, molecule_features, fingerprints, targets, target_mask):
        self.molfs = molecule_features
        self.fp    = torch.as_tensor(fingerprints, dtype=torch.float32)
        self.y     = torch.as_tensor(targets, dtype=torch.float32)
        self.mask  = torch.as_tensor(target_mask, dtype=torch.float32)
        # Find indices where 3D conformer generation failed; we exclude these
        self.valid_indices = [i for i, m in enumerate(self.molfs) if m is not None]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        return {
            "molf": self.molfs[i],
            "fp": self.fp[i],
            "y": self.y[i],
            "mask": self.mask[i],
        }


def collate_hybrid(batch_items, device=None):
    """Custom collate: builds a PyG Batch + a Uni-Mol2 padded batch + stacked
    fingerprint/target tensors.
    """
    from torch_geometric.data import Batch as PyGBatch

    pyg_list = [it["molf"]["pyg"] for it in batch_items]
    pyg_batch = PyGBatch.from_data_list(pyg_list)

    # Uni-Mol2 collator: pad along the atom dim.
    unimol_dicts = [it["molf"]["unimol"] for it in batch_items]
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

    fp   = torch.stack([it["fp"]   for it in batch_items])
    y    = torch.stack([it["y"]    for it in batch_items])
    mask = torch.stack([it["mask"] for it in batch_items])

    if device is not None:
        pyg_batch = pyg_batch.to(device)
        unimol_batch = {k: v.to(device) for k, v in unimol_batch.items()}
        fp, y, mask = fp.to(device), y.to(device), mask.to(device)

    return {
        "pamnet_data": pyg_batch,
        "unimol_batch": unimol_batch,
        "fp": fp,
        "y": y,
        "mask": mask,
    }


# ----- Build targets/mask matrices ------------------------------------------
def build_targets(train_df: pd.DataFrame, target_endpoints: List[str]):
    """Returns (targets (N, T), mask (N, T)) for the given endpoint subset,
    log-transformed via the tutorial recipe.
    """
    n = len(train_df)
    T = len(target_endpoints)
    targets = np.zeros((n, T), dtype=np.float32)
    mask = np.zeros((n, T), dtype=np.float32)
    for j, assay in enumerate(target_endpoints):
        log_scale, mult, _short, zh = cfg.ENDPOINT_PREP[assay]
        y = log_transform_endpoint(train_df[assay], log_scale, mult, zh)
        # NaN -> 0 target + 0 mask
        keep = y.notna().to_numpy()
        targets[keep, j] = y.to_numpy()[keep]
        mask[keep, j] = 1.0
    return targets, mask
