"""Fingerprint branch's input: MACCS keys + ErG + PubChem (881-bit) — the
exact three fingerprints HybridADMET concatenated in their "fingerprint
branch". HybridADMET note that this branch "yields a notable improvement
for Efflux and Papp", so we don't skimp on any of the three.

  MACCS  : RDKit ``MACCSkeys.GenMACCSKeys``                — 167 bits
  ErG    : RDKit ``rdReducedGraphs.GetErGFingerprint``     — 315 bits
  PubChem: ``padelpy.from_smiles`` (Java PaDEL backend)    — 881 bits
           ➔ Requires Java (``apt install default-jre`` on Ubuntu).
           padelpy launches the JVM once per call so we batch large lists.

Total feature dim:  167 + 315 + 881  =  1363
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdReducedGraphs

RDLogger.DisableLog("rdApp.*")

MACCS_BITS = 167
ERG_BITS = 315
PUBCHEM_BITS = 881
FP_DIM = MACCS_BITS + ERG_BITS + PUBCHEM_BITS    # 1363


# ----- MACCS + ErG (fast, RDKit-only) ---------------------------------------
def _maccs(mol) -> np.ndarray:
    arr = np.zeros(MACCS_BITS, dtype=np.float32)
    if mol is None:
        return arr
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr[:MACCS_BITS].astype(np.float32)


def _erg(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(ERG_BITS, dtype=np.float32)
    try:
        v = rdReducedGraphs.GetErGFingerprint(mol)
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape[0] < ERG_BITS:
            arr = np.concatenate([arr,
                                  np.zeros(ERG_BITS - arr.shape[0], dtype=np.float32)])
        return arr[:ERG_BITS]
    except Exception:
        return np.zeros(ERG_BITS, dtype=np.float32)


# ----- PubChem 881-bit (via padelpy / PaDEL Java backend) -------------------
def _pubchem_batch(smiles_list: List[str], chunk_size: int = 200) -> np.ndarray:
    """Run PaDEL-Descriptor PubChem fingerprint on a list of SMILES.

    PaDEL launches a JVM per ``from_smiles`` call (~2 s startup), so we
    feed it in chunks of ~200 SMILES to amortise. Returns (N, 881) float32.
    """
    from padelpy import from_smiles    # heavy import; defer to call time

    out = np.zeros((len(smiles_list), PUBCHEM_BITS), dtype=np.float32)
    for start in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[start:start + chunk_size]
        try:
            results = from_smiles(chunk, fingerprints=True, descriptors=False,
                                  timeout=600)
        except Exception as e:    # PaDEL sometimes chokes on weird SMILES
            print(f"  PaDEL failed on chunk {start}..{start+len(chunk)}: {e}")
            # fall back: try one-at-a-time
            results = []
            for s in chunk:
                try:
                    results.append(from_smiles(s, fingerprints=True,
                                               descriptors=False, timeout=120))
                except Exception:
                    results.append({f"PubchemFP{i}": 0 for i in range(PUBCHEM_BITS)})
        for j, mol_fp in enumerate(results):
            for k in range(PUBCHEM_BITS):
                v = mol_fp.get(f"PubchemFP{k}", 0)
                try:
                    out[start + j, k] = float(v)
                except (TypeError, ValueError):
                    out[start + j, k] = 0.0
    return out


def compute_fingerprints_batch(smiles_list: List[str],
                               cache_path: Path | None = None,
                               show_progress: bool = True) -> np.ndarray:
    """(N, 1363) float32 = MACCS(167) ⊕ ErG(315) ⊕ PubChem(881).

    Caches the concatenated matrix to ``cache_path`` (.npy) if given.
    PaDEL is slow; we cache so the second run is instant.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape == (len(smiles_list), FP_DIM):
            return cached

    # MACCS + ErG (fast, ~1 ms per molecule)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(smiles_list, desc="MACCS+ErG")
        except ImportError:
            it = smiles_list
    else:
        it = smiles_list
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in it]
    maccs = np.stack([_maccs(m) for m in mols])
    erg   = np.stack([_erg(m)   for m in mols])

    # PubChem via PaDEL (slow Java call, ~10 ms/molecule batched)
    print(f"  Computing PubChem 881-bit FP for {len(smiles_list)} molecules via PaDEL...")
    pubchem = _pubchem_batch(smiles_list)

    out = np.concatenate([maccs, erg, pubchem], axis=1).astype(np.float32)
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        print(f"  Cached {cache_path} ({out.nbytes/1024:.0f} KB)")
    return out
