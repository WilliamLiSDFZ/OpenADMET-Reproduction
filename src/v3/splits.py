"""Cross-validation / hold-out splits.

The OpenADMET-ExpansionRx test set is taken from a *later time slice* of
the same lead-optimization campaign (molecule IDs E-0020101+ vs E-00xxxxx
in train), so a random K-fold over training data is overoptimistic. We
implement two more realistic splitters in addition to random K-fold:

  * **scaffold split** -- by Bemis-Murcko scaffold; used for in-domain
    generalization.
  * **time window split** -- sort by ``Molecule Name`` (a time-correlated
    integer); train on the early X%, validate on the next Y%. Slides forward.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold


def random_kfold(n: int, n_splits: int = 5, seed: int = 42
                 ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    yield from KFold(n_splits=n_splits, shuffle=True,
                     random_state=seed).split(np.arange(n))


def _scaffold(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi,
                                                             includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold(smiles: List[str], n_splits: int = 5, seed: int = 42
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """K-fold on Bemis-Murcko scaffolds, mimicking JCIM 2025 setup."""
    n = len(smiles)
    scafs = np.array([_scaffold(s) for s in smiles])
    rng = np.random.default_rng(seed)

    # Group indices by scaffold
    unique = list(set(scafs.tolist()))
    rng.shuffle(unique)
    fold_of_scaf = {s: i % n_splits for i, s in enumerate(unique)}
    fold_of_idx = np.array([fold_of_scaf[s] for s in scafs])

    for k in range(n_splits):
        val = np.where(fold_of_idx == k)[0]
        tr = np.where(fold_of_idx != k)[0]
        yield tr, val


def _id_int(name: str) -> int:
    """Extract the integer suffix from molecule names like 'E-0001321'."""
    digits = "".join(c for c in str(name) if c.isdigit())
    return int(digits) if digits else 0


def time_window_split(molecule_names, train_pct: float = 0.7,
                      val_pct: float = 0.15
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by molecule ID, then take train / val / test fractions sequentially.

    Mimics the train/test convention of the OpenADMET challenge itself
    (train = E-0001321..E-0020100, test = E-0020101+). Used to pick
    ensemble weights without peeking at the real test set.
    """
    ids = np.array([_id_int(m) for m in molecule_names])
    order = np.argsort(ids, kind="stable")
    n = len(order)
    n_tr = int(round(train_pct * n))
    n_va = int(round(val_pct * n))
    tr = order[:n_tr]
    va = order[n_tr:n_tr + n_va]
    te = order[n_tr + n_va:]
    return tr, va, te


def time_sliding_window_kfold(molecule_names, n_splits: int = 5,
                              first_train_pct: float = 0.5,
                              step_pct: float = 0.1, val_pct: float = 0.1
                              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Sliding-window CV mimicking the MATCHA / OpenADME team's procedure.

    Pass 1: train on ID-sorted [0..50%], validate on [50%..60%]
    Pass 2: train on [0..60%], validate on [60%..70%]   ... and so on.
    """
    ids = np.array([_id_int(m) for m in molecule_names])
    order = np.argsort(ids, kind="stable")
    n = len(order)
    for k in range(n_splits):
        end_train = int(round((first_train_pct + k * step_pct) * n))
        end_val = end_train + int(round(val_pct * n))
        if end_val > n:
            end_val = n
        tr = order[:end_train]
        va = order[end_train:end_val]
        if len(va) == 0:
            return
        yield tr, va
