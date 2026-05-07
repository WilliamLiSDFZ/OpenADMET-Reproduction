"""Chemprop v2 multi-task MPNN wrapper.

This is the *single biggest performance lever* identified in the
OpenADMET-ExpansionRx top-10 reports — every team that finished in the
top 10 used Chemprop or a comparable graph neural net.

We train ONE multi-task model per task-affinity cluster (see
``config.TASK_GROUPS``), so a molecule that has values for endpoints in
several clusters contributes to all of them. Missing endpoint values are
masked out of the loss naturally by Chemprop v2.

GPU only — needs PyTorch + chemprop>=2.0. On a T4:
  cluster A (5 endpoints, ~5300 mols)  ≈ 25–35 min   for 50 epochs
  cluster B (3 endpoints, ~5300 mols)  ≈ 20–30 min
  cluster C (4 endpoints, ~5300 mols)  ≈ 25–35 min
  ensemble of 5 seeds × 3 clusters     ≈ 4-6 hours total
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import CHEMPROP_PARAMS, ENDPOINT_PREP, RANDOM_STATE, TASK_GROUPS, V3_OUT
from ..data import log_transform_endpoint


def _check_chemprop():
    try:
        import chemprop  # noqa: F401
        from lightning import pytorch as pl  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Chemprop pipeline requires chemprop>=2.0 and pytorch-lightning.\n"
            "Install on your T4 host:\n"
            "    pip install chemprop torch lightning\n"
            f"Original import error: {e}"
        )


CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_CACHE = ".chemprop/chemeleon_mp.pt"   # under user's home


def _download_chemeleon_ckpt() -> "Path | None":
    """Mirror of chemprop CLI's CHEMELEON download path: ``~/.chemprop/chemeleon_mp.pt``.
    Downloads from Zenodo on first use, then caches.

    Returns the cached path or None on failure.
    """
    from pathlib import Path
    from urllib.request import urlretrieve

    ckpt_dir = Path.home() / ".chemprop"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "chemeleon_mp.pt"
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path
    print(f"  CheMeleon: downloading from Zenodo to {model_path} (~30 MB)…")
    try:
        urlretrieve(CHEMELEON_URL, model_path)
        print(f"  CheMeleon: downloaded ({model_path.stat().st_size/1e6:.1f} MB)")
        return model_path
    except Exception as e:  # noqa: BLE001
        print(f"  CheMeleon: download failed ({type(e).__name__}: {e})")
        return None


def get_chemeleon_message_passing(BondMessagePassing):
    """Return a BondMessagePassing block initialised with CheMeleon weights.

    Returns ``None`` (and logs a warning) if CHEMELEON is disabled, the
    download fails, or the checkpoint can't be loaded for any reason —
    the caller should then construct a randomly-initialised block.

    Reads the env var ``CHEMELEON`` (``config.CHEMELEON``):
      ""/"0"/"false"/"no"            -> disabled, return None
      "1"/"true"/"auto"              -> auto-download from Zenodo
      "<path/to/chemeleon_mp.pt>"    -> load that checkpoint directly
    """
    from pathlib import Path
    import torch

    from .. import config as _cfg
    spec = _cfg.CHEMELEON
    if not spec or spec.lower() in ("0", "false", "no"):
        return None

    auto = spec.lower() in ("1", "true", "auto")
    if auto:
        ckpt_path = _download_chemeleon_ckpt()
        if ckpt_path is None:
            return None
    else:
        ckpt_path = Path(spec).expanduser()
        if not ckpt_path.exists():
            print(f"  CheMeleon: file not found at {ckpt_path}")
            return None

    # Try the safer weights_only=True load first (matches chemprop's CLI),
    # fall back to the unsafe one if needed.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"  CheMeleon: load failed ({type(e).__name__}: {e})")
        return None

    if not isinstance(ckpt, dict) or "hyper_parameters" not in ckpt or "state_dict" not in ckpt:
        print(f"  CheMeleon: unexpected checkpoint structure at {ckpt_path}")
        return None

    try:
        mp = BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
    except Exception as e:  # noqa: BLE001
        print(f"  CheMeleon: state_dict load failed ({type(e).__name__}: {e})")
        return None

    print(f"  CheMeleon: ✓ loaded from {ckpt_path} "
          f"(hidden_size={ckpt['hyper_parameters'].get('d_h', '?')})")
    return mp


def _maybe_v2_featurizer(Featurizer):
    """CheMeleon was pretrained with the V2 atom-featurizer, so when we use
    CheMeleon we MUST use V2 too (chemprop CLI enforces this with an error).

    Returns a featurizer instance — V2 if CheMeleon is requested, default
    otherwise.
    """
    from .. import config as _cfg
    if not _cfg.CHEMELEON or _cfg.CHEMELEON.lower() in ("0", "false", "no"):
        return Featurizer()

    # chemprop's V2 atom featurizer; live in chemprop.featurizers.atom
    try:
        import importlib
        atom_mod = importlib.import_module("chemprop.featurizers.atom")
        AF = getattr(atom_mod, "MultiHotAtomFeaturizer", None)
        if AF is not None and hasattr(AF, "v2"):
            return Featurizer(atom_featurizer=AF.v2())
    except Exception:
        pass
    # Fallback: default featurizer (most chemprop 2.2+ versions default to V2)
    return Featurizer()


def _resolve_chemprop_api():
    """Locate Chemprop's API objects across v2.0–v2.x naming changes.

    Returns a dict of resolved classes/functions; raises with a clear
    message if any required piece is missing.
    """
    import importlib
    found = {}
    # Featurizer moved between chemprop.data and chemprop.featurizers
    for candidate in (
        ("chemprop.featurizers",          "SimpleMoleculeMolGraphFeaturizer"),
        ("chemprop.featurizers.molgraph", "SimpleMoleculeMolGraphFeaturizer"),
        ("chemprop.data",                 "SimpleMoleculeMolGraphFeaturizer"),
    ):
        mod_name, attr = candidate
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, attr):
                found["Featurizer"] = getattr(mod, attr)
                break
        except ImportError:
            continue
    if "Featurizer" not in found:
        raise SystemExit(
            "Could not locate SimpleMoleculeMolGraphFeaturizer in chemprop. "
            "Tried: chemprop.featurizers, chemprop.featurizers.molgraph, "
            "chemprop.data. Run `pip show chemprop` and tell me the "
            "version so I can fix the wrapper."
        )

    # MoleculeDatapoint / Dataset / collate_batch — historically all in chemprop.data
    from chemprop import data as _data
    for attr in ("MoleculeDatapoint", "MoleculeDataset", "collate_batch"):
        if not hasattr(_data, attr):
            raise SystemExit(
                f"chemprop.data is missing {attr!r} -- API change.")
        found[attr] = getattr(_data, attr)

    # BondMessagePassing etc. live under chemprop.nn
    from chemprop import nn as _nn, models as _models
    for attr in ("BondMessagePassing", "MeanAggregation", "RegressionFFN"):
        if not hasattr(_nn, attr):
            raise SystemExit(f"chemprop.nn is missing {attr!r} -- API change.")
        found[attr] = getattr(_nn, attr)
    found["MPNN"] = _models.MPNN

    try:
        from chemprop.nn.metrics import MAE as _MAE
    except ImportError:
        raise SystemExit("Could not import chemprop.nn.metrics.MAE")
    found["MAE"] = _MAE
    return found


def _build_multitask_csv(train_df: pd.DataFrame, endpoints: List[str],
                         out_path: Path) -> List[str]:
    """Write a Chemprop-friendly CSV: smiles + N target columns (NaN ok)."""
    df = pd.DataFrame({"smiles": train_df["SMILES"].values})
    for assay in endpoints:
        log_scale, multiplier, short, zh = ENDPOINT_PREP[assay]
        df[short] = log_transform_endpoint(train_df[assay], log_scale, multiplier, zh)
    df.to_csv(out_path, index=False)
    short_cols = [ENDPOINT_PREP[a][2] for a in endpoints]
    return short_cols


def train_multitask(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    cluster_name: str, endpoints: List[str],
                    seed: int = RANDOM_STATE,
                    epochs: int | None = None,
                    train_indices=None,
                    val_indices=None,
                    ) -> Dict[str, np.ndarray]:
    """Train one multi-task Chemprop model on the given endpoint cluster.

    If ``train_indices`` is given, fits ONLY on those rows of ``train_df``
    (this is what we want to keep the time-window val truly held out — the
    earlier "train on everything then predict val" path leaked because every
    val molecule was already in chemprop's train set).

    If ``val_indices`` is given, also returns predictions on those rows so
    the per-endpoint NNLS ensemble can weight chemprop fairly.

    Returns a dict with two-level keys:
        ``"<short>__test"`` -> 1-D array of length len(test_df)
        ``"<short>__val"``  -> 1-D array of length len(val_indices)  (only when
                               val_indices is provided)
    Predictions are in the same log space as the targets.
    """
    _check_chemprop()
    import torch
    from lightning import pytorch as pl
    from torch.utils.data import DataLoader

    api = _resolve_chemprop_api()
    Featurizer       = api["Featurizer"]
    MoleculeDatapoint = api["MoleculeDatapoint"]
    MoleculeDataset   = api["MoleculeDataset"]
    collate_batch     = api["collate_batch"]
    BondMessagePassing = api["BondMessagePassing"]
    MeanAggregation    = api["MeanAggregation"]
    RegressionFFN      = api["RegressionFFN"]
    MPNN               = api["MPNN"]
    MAE                = api["MAE"]
    # UnscaleTransform's signature changed across chemprop versions; we
    # don't actually need to unscale (target values are already in log
    # space), so we simply omit output_transform and let RegressionFFN
    # use its identity default.

    epochs = epochs or CHEMPROP_PARAMS["epochs"]
    work_dir = V3_OUT / "chemprop" / f"{cluster_name}_seed{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)

    train_csv = work_dir / "train.csv"
    short_cols = _build_multitask_csv(train_df, endpoints, train_csv)

    # ---- Build Chemprop dataset --------------------------------------------
    # Restrict the training pool to the requested indices so the time-window
    # val rows are held out (no leakage into the ensemble-weight learning).
    if train_indices is not None:
        train_pool = train_df.iloc[train_indices].reset_index(drop=True)
    else:
        train_pool = train_df

    smis = train_pool["SMILES"].tolist()
    test_smis = test_df["SMILES"].tolist()

    targets = np.full((len(smis), len(short_cols)), np.nan, dtype=np.float32)
    for j, c in enumerate(short_cols):
        log_scale, mult, _, zh = ENDPOINT_PREP[endpoints[j]]
        y = log_transform_endpoint(train_pool[endpoints[j]], log_scale, mult, zh)
        targets[:, j] = y.to_numpy()

    train_data = [MoleculeDatapoint.from_smi(s, y) for s, y in zip(smis, targets)]
    # Random 90/10 split of the training pool, JUST for early stopping. This
    # internal split is contained inside the time-window train fold, so the
    # external time-window val (val_indices) is never seen by the model.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_data))
    n_internal_val = max(1, len(train_data) // 10)
    internal_val = set(perm[:n_internal_val].tolist())
    tr = [d for i, d in enumerate(train_data) if i not in internal_val]
    va = [d for i, d in enumerate(train_data) if i in internal_val]

    test_data = [MoleculeDatapoint.from_smi(s, np.zeros(len(short_cols), dtype=np.float32))
                 for s in test_smis]

    # Time-window val SMILES (held out from training): predicted at the end so
    # the ensemble layer can weight chemprop on leak-free predictions.
    holdout_smis = []
    if val_indices is not None:
        holdout_smis = train_df["SMILES"].iloc[val_indices].tolist()
    holdout_data = [MoleculeDatapoint.from_smi(s, np.zeros(len(short_cols), dtype=np.float32))
                    for s in holdout_smis]

    featurizer = _maybe_v2_featurizer(Featurizer)
    train_dset   = MoleculeDataset(tr, featurizer)
    val_dset     = MoleculeDataset(va, featurizer)
    test_dset    = MoleculeDataset(test_data, featurizer)
    holdout_dset = MoleculeDataset(holdout_data, featurizer) if holdout_smis else None

    bs = CHEMPROP_PARAMS["batch_size"]
    train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True,
                              num_workers=2, collate_fn=collate_batch)
    val_loader = DataLoader(val_dset, batch_size=bs, shuffle=False,
                            num_workers=2, collate_fn=collate_batch)
    test_loader = DataLoader(test_dset, batch_size=bs, shuffle=False,
                             num_workers=2, collate_fn=collate_batch)
    holdout_loader = (DataLoader(holdout_dset, batch_size=bs, shuffle=False,
                                  num_workers=2, collate_fn=collate_batch)
                      if holdout_dset is not None else None)

    # ---- Model -------------------------------------------------------------
    # If CHEMELEON is set, build the BondMessagePassing block from the
    # pretrained checkpoint (which dictates its own hidden_size/depth);
    # otherwise fall back to randomly-initialised one with our hyperparams.
    mp = None
    try:
        mp = get_chemeleon_message_passing(BondMessagePassing)
    except Exception as _e:  # noqa: BLE001
        print(f"  CheMeleon init failed ({type(_e).__name__}: {_e}); "
              "continuing with random weights.")
    if mp is None:
        mp = BondMessagePassing(depth=CHEMPROP_PARAMS["depth"],
                                d_h=CHEMPROP_PARAMS["hidden_size"],
                                dropout=CHEMPROP_PARAMS["dropout"])
    agg = MeanAggregation()
    # Use the message-passing block's actual output_dim for the FFN, in case
    # CheMeleon's hidden_size differs from our config default.
    ffn_kwargs = {"n_tasks": len(short_cols)}
    if hasattr(mp, "output_dim"):
        ffn_kwargs["input_dim"] = mp.output_dim
    elif hasattr(mp, "d_h"):
        ffn_kwargs["input_dim"] = mp.d_h
    try:
        ffn = RegressionFFN(**ffn_kwargs)
    except TypeError:
        # Older API: RegressionFFN doesn't take input_dim
        ffn = RegressionFFN(n_tasks=len(short_cols))
    metric_list = [MAE()]
    model = MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)

    pl.seed_everything(seed)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",            # picks GPU when available
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True,
        default_root_dir=str(work_dir),
        enable_checkpointing=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trainer.fit(model, train_loader, val_loader)

    # ---- Predict on test ---------------------------------------------------
    preds_per_batch = trainer.predict(model, test_loader)
    test_preds = torch.cat(preds_per_batch).cpu().numpy()  # (n_test, n_tasks)

    out: Dict[str, np.ndarray] = {}
    for j, sc in enumerate(short_cols):
        out[f"{sc}__test"] = test_preds[:, j]

    # ---- (Optional) Predict on the held-out time-window val ----------------
    if holdout_loader is not None:
        holdout_per_batch = trainer.predict(model, holdout_loader)
        holdout_preds = torch.cat(holdout_per_batch).cpu().numpy()  # (n_val, n_tasks)
        for j, sc in enumerate(short_cols):
            out[f"{sc}__val"] = holdout_preds[:, j]

    np.savez_compressed(work_dir / "predictions.npz", **out,
                        test_smiles=np.array(test_smis),
                        val_smiles=np.array(holdout_smis),
                        columns=np.array(short_cols))
    print(f"  Wrote {work_dir/'predictions.npz'}")
    return out


def train_all_clusters(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       seeds: List[int] | None = None,
                       train_indices=None,
                       val_indices=None,
                       ) -> Dict[str, np.ndarray]:
    """Train every TASK_GROUPS cluster, ensembled across seeds. Returns

        {"<short>__test": averaged_log_test_preds,
         "<short>__val":  averaged_log_val_preds   (only if val_indices given)}

    Pass ``train_indices`` (and ``val_indices``) to keep an external held-out
    set leak-free for ensemble weight learning. Without those, training uses
    the full train_df (legacy behaviour, kept for backward compat).
    """
    seeds = seeds or list(range(CHEMPROP_PARAMS["ensemble_size"]))
    accumulator: Dict[str, List[np.ndarray]] = {}
    for cluster_name, endpoints in TASK_GROUPS.items():
        for seed in seeds:
            print(f"== Chemprop  cluster={cluster_name}  seed={seed} ==")
            preds = train_multitask(
                train_df, test_df, cluster_name, endpoints, seed=seed,
                train_indices=train_indices, val_indices=val_indices,
            )
            for k, v in preds.items():
                accumulator.setdefault(k, []).append(v)

    averaged = {k: np.mean(np.stack(v_list), axis=0)
                for k, v_list in accumulator.items()}
    return averaged
