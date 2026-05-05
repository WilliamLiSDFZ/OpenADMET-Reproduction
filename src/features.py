"""Feature extraction: RDKit physico-chemical descriptors + Morgan fingerprints."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# Build the RDKit descriptor calculator once per process.
_DESC_NAMES: List[str] = [d[0] for d in Descriptors.descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)


def smiles_to_mol(smiles: str):
    """Parse SMILES -> Mol. Returns None on failure."""
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def calc_rdkit_descriptors(mol) -> np.ndarray:
    """Return the 200+ RDKit descriptors as a 1-D float array.

    Any NaN / inf produced by descriptors on weird molecules are replaced by 0.
    """
    if mol is None:
        return np.zeros(len(_DESC_NAMES), dtype=np.float32)
    try:
        vals = list(_DESC_CALC.CalcDescriptors(mol))
    except Exception:
        return np.zeros(len(_DESC_NAMES), dtype=np.float32)
    arr = np.asarray(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def calc_morgan_fp(mol, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Return a Morgan fingerprint as a uint8 array of length n_bits."""
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Concatenate RDKit descriptors and Morgan fingerprint into one vector."""
    mol = smiles_to_mol(smiles)
    desc = calc_rdkit_descriptors(mol)
    fp = calc_morgan_fp(mol, n_bits=n_bits, radius=radius).astype(np.float32)
    return np.concatenate([desc, fp])


def featurize_smiles_batch(smiles_list, n_bits: int = 2048, radius: int = 2,
                           show_progress: bool = True) -> Tuple[np.ndarray, list]:
    """Featurize a list of SMILES. Returns (feature_matrix, list_of_failed_indices)."""
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Featurizing")
        except ImportError:
            iterator = smiles_list
    else:
        iterator = smiles_list

    feats = []
    failed = []
    for i, smi in enumerate(iterator):
        vec = featurize_smiles(smi, n_bits=n_bits, radius=radius)
        if smiles_to_mol(smi) is None:
            failed.append(i)
        feats.append(vec)
    return np.asarray(feats, dtype=np.float32), failed


def feature_dim(n_bits: int = 2048) -> int:
    """Total dimension of the concatenated feature vector."""
    return len(_DESC_NAMES) + n_bits
