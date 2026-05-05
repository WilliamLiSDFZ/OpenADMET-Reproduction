"""SALI-based activity / property cliff detection.

Reproduces the procedure from Fischer et al. (J. Chem. Inf. Model. 2025):

    SALI(i, j) = |A_i - A_j| / (1.05 - Tanimoto(i, j))

where A is the log-space target. Compounds whose maximum SALI z-score
(within their structural cluster) lies more than ``z_thresh`` standard
deviations from the population mean are flagged as cliffs and dropped
from training.

Returns a boolean mask the SAME length as the input arrays:
    True = keep, False = drop.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering


def _morgan_bitvec(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def _morgan_to_array(fp, n_bits: int = 2048):
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def detect_cliffs(smiles, y, z_thresh: float = 2.5, n_clusters: int | None = None,
                  cluster_size: int = 50, verbose: bool = False
                  ) -> Tuple[np.ndarray, dict]:
    """Flag activity / property cliffs.

    Args:
        smiles: 1-D iterable of SMILES strings (length N).
        y:      1-D iterable of target values in log space (length N).
        z_thresh: |z-score| above which a compound is a cliff. Default 2.5.
        n_clusters: explicit number of clusters; if None, choose so that
                    each cluster has roughly ``cluster_size`` molecules.
        cluster_size: target average cluster size when n_clusters is None.

    Returns:
        keep_mask: boolean array of length N (True = keep).
        info: dict with summary stats.
    """
    smiles = list(smiles)
    y = np.asarray(y, dtype=float)
    n = len(smiles)
    assert len(y) == n

    # Fingerprints (skip molecules that fail to parse)
    fps = [_morgan_bitvec(s) for s in smiles]
    valid = np.array([fp is not None for fp in fps])
    if not valid.all():
        # only do cliff detection on valid molecules; invalid ones we keep as-is
        pass

    if n_clusters is None:
        n_clusters = max(2, n // cluster_size)

    # Agglomerative clustering on Morgan-bit space (Hamming-equivalent on uint8)
    fp_arr = np.array([_morgan_to_array(fp) if fp is not None else np.zeros(2048, dtype=np.uint8)
                       for fp in fps])
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="average",
                                      metric="hamming")
    labels = cluster.fit_predict(fp_arr)

    # Per-cluster, compute pairwise SALI and take max per-row as the cliff signal
    max_sali = np.zeros(n, dtype=float)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        # Tanimoto similarity matrix within cluster
        cluster_fps = [fps[i] for i in idx]
        # Filter out None
        cl_idx = [i for i, fp in zip(idx, cluster_fps) if fp is not None]
        cluster_fps = [fp for fp in cluster_fps if fp is not None]
        if len(cluster_fps) < 2:
            continue
        sims = np.zeros((len(cluster_fps), len(cluster_fps)), dtype=float)
        for k, fp in enumerate(cluster_fps):
            row = DataStructs.BulkTanimotoSimilarity(fp, cluster_fps)
            sims[k] = row
        # SALI: |y_i - y_j| / (1.05 - sim)  (regularized denominator)
        y_local = y[cl_idx]
        diff = np.abs(y_local[:, None] - y_local[None, :])
        denom = 1.05 - sims
        sali_mat = diff / denom
        np.fill_diagonal(sali_mat, 0.0)
        for k, abs_idx in enumerate(cl_idx):
            max_sali[abs_idx] = max(max_sali[abs_idx], sali_mat[k].max())

    # Z-score over the population that has any pairs at all
    nonzero = max_sali > 0
    if nonzero.sum() < 5:
        # Not enough signal; keep everyone
        return np.ones(n, dtype=bool), {"n_total": n, "n_dropped": 0,
                                        "n_clusters": int(n_clusters)}
    mu = max_sali[nonzero].mean()
    sd = max_sali[nonzero].std()
    z = np.zeros(n, dtype=float)
    z[nonzero] = (max_sali[nonzero] - mu) / (sd if sd > 0 else 1.0)
    cliff = z > z_thresh
    keep = ~cliff
    if verbose:
        print(f"  cliffs: {cliff.sum()}/{n} dropped "
              f"(z > {z_thresh}, n_clusters={n_clusters})")
    return keep, {"n_total": int(n), "n_dropped": int(cliff.sum()),
                  "n_clusters": int(n_clusters), "mean_sali": float(mu),
                  "std_sali": float(sd)}


if __name__ == "__main__":
    # Quick smoke test on the challenge LogD column
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "train.csv")
    sub = df[["SMILES", "LogD"]].dropna().reset_index(drop=True).head(500)
    keep, info = detect_cliffs(sub["SMILES"].tolist(), sub["LogD"].tolist(),
                               verbose=True)
    print(info)
    print(f"Remaining: {keep.sum()}/{len(keep)}")
