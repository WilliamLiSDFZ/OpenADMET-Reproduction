"""Featurization: RDKit-2d + Morgan + Avalon (+ optional Mordred).

Caches features under ``output/v3/features_*.npz`` for cheap re-use.
``RDKIT_ONLY`` flag returns just the 210-dim RDKit-2d slice — used by
TabPFN, which has a hard ~500-feature input limit.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .config import AVALON_BITS, FP_BITS, FP_RADIUS, V3_OUT

_DESC_NAMES = [d[0] for d in Descriptors.descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
N_RDKIT = len(_DESC_NAMES)
_MORDRED_CALC = None  # lazy


def _mordred():
    global _MORDRED_CALC
    if _MORDRED_CALC is None:
        from mordred import Calculator, descriptors
        _MORDRED_CALC = Calculator(descriptors, ignore_3D=True)
    return _MORDRED_CALC


def _safe_mol(s: str):
    if not isinstance(s, str) or not s:
        return None
    return Chem.MolFromSmiles(s)


def _rdkit_desc(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(N_RDKIT, dtype=np.float32)
    try:
        v = list(_DESC_CALC.CalcDescriptors(mol))
    except Exception:
        return np.zeros(N_RDKIT, dtype=np.float32)
    arr = np.asarray(v, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def _morgan(mol) -> np.ndarray:
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _avalon(mol) -> np.ndarray:
    arr = np.zeros(AVALON_BITS, dtype=np.uint8)
    if mol is None:
        return arr
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=AVALON_BITS)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(smiles: str, use_mordred: bool = False) -> np.ndarray:
    mol = _safe_mol(smiles)
    parts = [_rdkit_desc(mol), _morgan(mol).astype(np.float32),
             _avalon(mol).astype(np.float32)]
    if use_mordred:
        if mol is None:
            parts.append(np.zeros(len(_mordred().descriptors), dtype=np.float32))
        else:
            try:
                vals = list(_mordred()(mol).fill_missing(0.0)._values)
            except Exception:
                vals = [0.0] * len(_mordred().descriptors)
            arr = np.asarray(vals, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(arr.astype(np.float32))
    return np.concatenate(parts)


def feature_dim(use_mordred: bool = False) -> int:
    n = N_RDKIT + FP_BITS + AVALON_BITS
    if use_mordred:
        n += len(_mordred().descriptors)
    return n


def rdkit_only_slice() -> Tuple[int, int]:
    """Indices [start, end) of the 210-dim RDKit-2d block (for TabPFN)."""
    return 0, N_RDKIT


def morgan_slice() -> Tuple[int, int]:
    return N_RDKIT, N_RDKIT + FP_BITS


def featurize_batch(smiles_list, use_mordred: bool = False, show_progress: bool = True
                    ) -> Tuple[np.ndarray, list]:
    """Return (X, failed_indices). Mordred uses parallel pandas if enabled."""
    failed = []
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(smiles_list, desc="Featurize")
        except ImportError:
            it = smiles_list
    else:
        it = smiles_list

    base = []
    for i, s in enumerate(it):
        mol = _safe_mol(s)
        if mol is None:
            failed.append(i)
        base.append(np.concatenate([_rdkit_desc(mol),
                                    _morgan(mol).astype(np.float32),
                                    _avalon(mol).astype(np.float32)]))
    base = np.asarray(base, dtype=np.float32)

    if not use_mordred:
        return base, failed

    print(f"Computing Mordred ({len(smiles_list)} molecules, parallel)...")
    mols = [_safe_mol(s) for s in smiles_list]
    df = _mordred().pandas(mols, nproc=4, quiet=True).fill_missing(0.0)
    mord = np.nan_to_num(df.to_numpy().astype(np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)
    return np.concatenate([base, mord], axis=1), failed


def cached_features(name: str, smiles_list, use_mordred: bool = False) -> np.ndarray:
    """Featurize with on-disk caching keyed by ``name`` + ``use_mordred``."""
    suffix = "_mord" if use_mordred else ""
    cache = V3_OUT / f"features_{name}{suffix}.npz"
    if cache.exists():
        loaded = np.load(cache, allow_pickle=True)
        if loaded["X"].shape[0] == len(smiles_list):
            return loaded["X"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    X, _ = featurize_batch(smiles_list, use_mordred=use_mordred)
    np.savez_compressed(cache, X=X, smiles=np.array(smiles_list))
    return X
