"""5-fold Bemis-Murcko scaffold split — identical to v4_hybridadmet.splits.

Imported via ``from .splits import scaffold_kfold``. We could just re-export
v4's version, but keeping a local copy means v5 isn't accidentally affected
if v4 gets edited later.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def _scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(
            smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold(smiles: List[str], n_splits: int = 5, seed: int = 42
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) for each of K scaffold-balanced folds."""
    scafs = np.array([_scaffold(s) for s in smiles])
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(scafs.tolist()))
    rng.shuffle(unique)
    fold_of_scaf = {s: i % n_splits for i, s in enumerate(unique)}
    fold_of_idx = np.array([fold_of_scaf[s] for s in scafs])
    for k in range(n_splits):
        val = np.where(fold_of_idx == k)[0]
        tr = np.where(fold_of_idx != k)[0]
        yield tr, val
