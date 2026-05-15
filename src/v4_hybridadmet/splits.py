"""5-fold Bemis-Murcko scaffold split — the exact splitter HybridADMET
used ("Splitting. We use a 5-fold scaffold split.").

A scaffold split groups molecules by their Bemis-Murcko core scaffold so
that no scaffold appears in both train and val of the same fold — this
better reflects generalisation to novel chemical series than random K-fold.
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
    unique = list(dict.fromkeys(scafs.tolist()))   # preserves insertion order
    rng.shuffle(unique)
    fold_of_scaf = {s: i % n_splits for i, s in enumerate(unique)}
    fold_of_idx = np.array([fold_of_scaf[s] for s in scafs])
    for k in range(n_splits):
        val = np.where(fold_of_idx == k)[0]
        tr = np.where(fold_of_idx != k)[0]
        yield tr, val
