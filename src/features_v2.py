"""Enhanced feature extraction (v2): RDKit-2d + Mordred + Morgan + Avalon.

This is the "augmented classical descriptors" recipe that the JCIM paper
(Fischer, Southiratn, Triki, Cedeno 2025) found best for ADME tasks. The
biggest single win comes from concatenating Mordred + Avalon onto the v1
RDKit-2d + Morgan stack.

Cached feature dim is large (~3900) but tree models handle it fine, and
LightGBM with feature_fraction=0.8 implicitly does feature selection.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# Silence RDKit "DEPRECATION WARNING: please use MorganGenerator" spam.
RDLogger.DisableLog("rdApp.*")

# ---- RDKit 2D descriptors (~210 columns) -------------------------------
_DESC_NAMES: List[str] = [d[0] for d in Descriptors.descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_MORGAN_GEN_2_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---- Morgan fingerprint (radius=2, 2048 bits) ---------------------------
MORGAN_BITS = 2048
MORGAN_RADIUS = 2

# ---- Avalon fingerprint (1024 bits in JCIM paper) -----------------------
AVALON_BITS = 1024

# ---- Mordred descriptors (~1600, 2D-only, much faster than 3D) ----------
_MORDRED_CALC = None  # lazy init -- import is heavy


def _mordred() -> "Calculator":
    global _MORDRED_CALC
    if _MORDRED_CALC is None:
        from mordred import Calculator, descriptors
        _MORDRED_CALC = Calculator(descriptors, ignore_3D=True)
    return _MORDRED_CALC


# ---- Per-molecule featurization ----------------------------------------
def smiles_to_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def calc_rdkit_descriptors(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(len(_DESC_NAMES), dtype=np.float32)
    try:
        vals = list(_DESC_CALC.CalcDescriptors(mol))
    except Exception:
        return np.zeros(len(_DESC_NAMES), dtype=np.float32)
    arr = np.asarray(vals, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def calc_morgan_fp(mol, n_bits: int = MORGAN_BITS,
                   radius: int = MORGAN_RADIUS) -> np.ndarray:
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    if (radius, n_bits) == (2, 2048):
        gen = _MORGAN_GEN_2_2048
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_avalon_fp(mol, n_bits: int = AVALON_BITS) -> np.ndarray:
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    try:
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    except Exception:
        return np.zeros(n_bits, dtype=np.uint8)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_mordred(mol) -> np.ndarray:
    """Full 2D Mordred descriptor vector (~1600 floats)."""
    calc = _mordred()
    if mol is None:
        return np.zeros(len(calc.descriptors), dtype=np.float32)
    try:
        vals = calc(mol).fill_missing(0.0)._values
    except Exception:
        return np.zeros(len(calc.descriptors), dtype=np.float32)
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def featurize_smiles_v2(smiles: str, use_mordred: bool = False) -> np.ndarray:
    """RDKit-2d + Morgan + Avalon (+ optional Mordred) concat -> float32."""
    mol = smiles_to_mol(smiles)
    desc = calc_rdkit_descriptors(mol)
    morgan = calc_morgan_fp(mol).astype(np.float32)
    avalon = calc_avalon_fp(mol).astype(np.float32)
    parts = [desc, morgan, avalon]
    if use_mordred:
        parts.append(calc_mordred(mol))
    return np.concatenate(parts)


def feature_dim_v2(use_mordred: bool = False) -> int:
    n = len(_DESC_NAMES) + MORGAN_BITS + AVALON_BITS
    if use_mordred:
        n += len(_mordred().descriptors)
    return n


def featurize_smiles_batch_v2(smiles_list, show_progress: bool = True,
                              use_mordred: bool = False, nproc: int = 4,
                              ) -> Tuple[np.ndarray, list]:
    """Featurize a batch. When use_mordred=True, Mordred is computed in
    parallel via mordred.Calculator.pandas; everything else is serial.
    """
    failed = []

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Featurize (RDKit+Morgan+Avalon)")
        except ImportError:
            iterator = smiles_list
    else:
        iterator = smiles_list

    base_parts = []
    for i, smi in enumerate(iterator):
        mol = smiles_to_mol(smi)
        if mol is None:
            failed.append(i)
        desc = calc_rdkit_descriptors(mol)
        morgan = calc_morgan_fp(mol).astype(np.float32)
        avalon = calc_avalon_fp(mol).astype(np.float32)
        base_parts.append(np.concatenate([desc, morgan, avalon]))
    base = np.asarray(base_parts, dtype=np.float32)

    if not use_mordred:
        return base, failed

    # Mordred in parallel via its built-in pandas interface
    print(f"Computing Mordred ({len(smiles_list)} mols, nproc={nproc})...")
    mols = [smiles_to_mol(s) for s in smiles_list]
    calc = _mordred()
    mord_df = calc.pandas(mols, nproc=nproc, quiet=True).fill_missing(0.0)
    mord = mord_df.to_numpy().astype(np.float32)
    mord = np.nan_to_num(mord, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([base, mord], axis=1), failed


if __name__ == "__main__":
    print(f"Total feature dim: {feature_dim_v2()}")
    test = "CCO"
    v = featurize_smiles_v2(test)
    print(f"Test SMILES '{test}' -> shape {v.shape}, "
          f"first 5 values: {v[:5]}")
