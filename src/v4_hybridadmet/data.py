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


# ----- Uni-Mol2 dictionary loader -------------------------------------------
# Uni-Mol2 ships a fairseq-format atom dictionary at
# ``unimol_tools/weights/mol.dict.txt``. The encoder's atom-type embedding
# table is sized by ``len(dictionary)`` and the GBF (Gaussian basis function)
# embedding by ``len(dict)²``. If we feed raw atomic numbers as token ids,
# H/C/N/O happen to work but P/S/Cl/Br/I overflow the table → CUDA assertion.
_UNIMOL_DICT_CACHE = None


def _load_unimol_dict():
    """Return Uni-Mol2's atom Dictionary (fairseq-style). Cached.

    Falls back to manual parsing if unimol_tools doesn't expose its
    Dictionary class at any of the known import paths.
    """
    global _UNIMOL_DICT_CACHE
    if _UNIMOL_DICT_CACHE is not None:
        return _UNIMOL_DICT_CACHE

    import importlib
    from pathlib import Path
    import unimol_tools

    # Locate mol.dict.txt — shipped with unimol_tools weights/
    pkg_dir = Path(unimol_tools.__file__).parent
    candidates = [pkg_dir / "weights" / "mol.dict.txt",
                  pkg_dir / "data" / "mol.dict.txt",
                  pkg_dir / "mol.dict.txt"]
    dict_path = next((p for p in candidates if p.exists()), None)

    if dict_path is None:
        # The dict gets auto-downloaded the first time UniMolModel is built.
        # If we're called before that, trigger it now.
        print("  mol.dict.txt missing -- triggering unimol_tools download…")
        try:
            from unimol_tools.weights.weighthub import weight_download
            for fname in ("mol.dict.txt", "mol_pre_all_h_220816.pt"):
                try:
                    weight_download(str(pkg_dir / "weights"), fname)
                except Exception as e:  # noqa: BLE001
                    print(f"    weight_download({fname}) failed: "
                          f"{type(e).__name__}: {e}")
        except ImportError:
            # Last resort: instantiate UniMolModel to make it fetch
            try:
                from unimol_tools.models.unimol import UniMolModel
                _ = UniMolModel(output_dim=512)
                del _
            except Exception as e:  # noqa: BLE001
                raise FileNotFoundError(
                    f"mol.dict.txt not found and auto-download failed: {e}"
                ) from e
        dict_path = next((p for p in candidates if p.exists()), None)
        if dict_path is None:
            raise FileNotFoundError(
                f"mol.dict.txt still missing after download attempt in {pkg_dir}")

    # Try unimol_tools' own Dictionary class across known API paths
    for mod_name in (
        "unimol_tools.data.dictionary",
        "unimol_tools.data",
        "unimol_tools.utils.dictionary",
        "unimol_tools.utils",
    ):
        try:
            mod = importlib.import_module(mod_name)
            Dictionary = getattr(mod, "Dictionary", None)
            if Dictionary is None:
                continue
            d = Dictionary.load(str(dict_path))
            _UNIMOL_DICT_CACHE = d
            print(f"  Loaded Uni-Mol2 dictionary via {mod_name}.Dictionary "
                  f"({len(d)} tokens)")
            return d
        except (ImportError, AttributeError, Exception):  # noqa: BLE001
            continue

    # Fallback: manual fairseq-style parsing
    class _ManualDict:
        def __init__(self, symbols, specials):
            self._idx = dict(specials)
            n = len(specials)
            for s in symbols:
                if s not in self._idx:
                    self._idx[s] = n
                    n += 1
            self._inv = {v: k for k, v in self._idx.items()}

        def index(self, sym):
            return self._idx.get(sym, self._idx.get("[UNK]", 3))

        def __len__(self):
            return max(self._idx.values()) + 1

        def cls(self): return self._idx.get("[CLS]", 0)
        def unk(self): return self._idx.get("[UNK]", 3)
        def pad(self): return self._idx.get("[PAD]", 1)

    with open(dict_path) as f:
        symbols = [line.strip().split()[0] for line in f if line.strip()]
    specials = {"[CLS]": 0, "[PAD]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4}
    d = _ManualDict(symbols, specials)
    print(f"  Loaded Uni-Mol2 dictionary via manual fallback "
          f"({len(d)} tokens, specials={list(specials.keys())})")
    _UNIMOL_DICT_CACHE = d
    return d


def _atom_symbol(z: int) -> str:
    from rdkit.Chem import GetPeriodicTable
    return GetPeriodicTable().GetElementSymbol(int(z))


# ----- Uni-Mol2 input packing -----------------------------------------------
def pack_unimol_input(atoms: np.ndarray, coords: np.ndarray) -> dict:
    """Build the per-molecule dict Uni-Mol2's encoder expects.

    Uni-Mol2 input convention:
      * Prepend a ``[CLS]`` token at position 0 (used as molecule-level pool).
      * ``src_tokens`` are indices INTO mol.dict.txt, NOT raw atomic numbers.
      * ``src_edge_type[i,j] = src_tokens[i] * n_dict + src_tokens[j]``
        — i.e. the multiplier is the dictionary size, so edge types fit
        inside an embedding table of size ``n_dict²``.
      * The CLS row in ``src_coord`` is placed at the molecule's centroid
        (this is what the Uni-Mol2 authors do).
    """
    d = _load_unimol_dict()
    n_dict = len(d)
    cls_idx = d.cls() if hasattr(d, "cls") else d.index("[CLS]")
    unk_idx = d.unk() if hasattr(d, "unk") else d.index("[UNK]")

    tokens = [cls_idx]
    for z in atoms:
        sym = _atom_symbol(int(z))
        try:
            tokens.append(d.index(sym))
        except (KeyError, ValueError):
            tokens.append(unk_idx)

    n = len(tokens)                                    # = len(atoms) + 1
    src_tokens = torch.tensor(tokens, dtype=torch.long)

    # CLS coordinate at molecule centroid (Uni-Mol2 convention)
    coords_arr = np.asarray(coords, dtype=np.float32)
    cls_coord = coords_arr.mean(axis=0, keepdims=True)
    src_coord = torch.tensor(
        np.concatenate([cls_coord, coords_arr], axis=0), dtype=torch.float32)

    diff = src_coord.unsqueeze(0) - src_coord.unsqueeze(1)
    src_distance = (diff * diff).sum(dim=-1).clamp(min=1e-12).sqrt()

    src_edge_type = src_tokens.unsqueeze(0) * n_dict + src_tokens.unsqueeze(1)

    return {
        "src_tokens": src_tokens,
        "src_coord": src_coord,
        "src_distance": src_distance,
        "src_edge_type": src_edge_type,
        "n_atoms": n,
    }


# ----- Build per-molecule feature pack --------------------------------------
# NOTE: We intentionally DO NOT call ``pack_unimol_input`` here. That function
# needs the Uni-Mol2 atom dictionary (``mol.dict.txt``), which only gets
# downloaded to ``unimol_tools/weights/`` the first time UniMolModel is
# instantiated — which happens later in train_one, not at precompute time.
# Building the Uni-Mol2 batch dict lazily inside collate_hybrid avoids
# this ordering problem AND keeps the on-disk cache slim.
def build_molecule_features(smiles: str, seed: int = 42) -> dict | None:
    atoms, coords = smiles_to_3d(smiles, seed=seed)
    if atoms is None:
        return None
    pyg_data = smiles_to_pyg_data(smiles, atoms, coords)
    return {
        "smiles": smiles,
        "atoms": atoms,
        "coords": coords,
        "pyg": pyg_data,
        # "unimol" packed lazily at collate time
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

    # Uni-Mol2 collator: build per-molecule input dict NOW (lazy), so we don't
    # depend on the Uni-Mol2 weights being downloaded at precompute time.
    # If a cached pickle from an old version still has a "unimol" key, ignore it.
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
