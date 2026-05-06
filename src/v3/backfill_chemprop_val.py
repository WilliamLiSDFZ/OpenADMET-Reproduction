"""Backfill Chemprop's time-window validation predictions.

Why this exists: ``train_multitask`` in ``models/chemprop_model.py`` only
saves test-set predictions; the per-endpoint NNLS ensemble in ``run.py``
needs val predictions to learn weights. Without them Chemprop ends up
with weight 0 in every endpoint.

This script:
  1. computes the same time-window val SMILES that the ensemble layer
     uses (70/15/15 split sorted by ``Molecule Name``);
  2. for every saved Chemprop checkpoint under
     ``output/v3/chemprop/<cluster>_seed<n>/lightning_logs/...``,
     reconstructs the model architecture and loads weights;
  3. predicts on the val SMILES;
  4. averages predictions across (cluster, seed) pairs per endpoint;
  5. writes ``chemprop__<short>__val.npz`` to the per-model cache.

After this, just re-run::

    python -m src.v3.run --resume

and the ensemble weights will pick up Chemprop properly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from . import config as cfg
from .data import load_train
from .models.chemprop_model import _check_chemprop, _resolve_chemprop_api
from .splits import time_window_split

warnings.filterwarnings("ignore", category=UserWarning)

PRED_DIR = cfg.V3_OUT / "per_model_predictions"
CHEMPROP_DIR = cfg.V3_OUT / "chemprop"


def _find_best_ckpt(work_dir: Path) -> Path | None:
    """Return the .ckpt with the largest mtime (= last epoch) under work_dir."""
    candidates = list(work_dir.rglob("*.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    _check_chemprop()
    import torch
    from lightning import pytorch as pl
    from torch.utils.data import DataLoader

    api = _resolve_chemprop_api()
    Featurizer        = api["Featurizer"]
    MoleculeDatapoint = api["MoleculeDatapoint"]
    MoleculeDataset   = api["MoleculeDataset"]
    collate_batch     = api["collate_batch"]
    BondMessagePassing = api["BondMessagePassing"]
    MeanAggregation    = api["MeanAggregation"]
    RegressionFFN      = api["RegressionFFN"]
    MPNN               = api["MPNN"]
    MAE                = api["MAE"]

    bp = cfg.CHEMPROP_PARAMS
    n_seeds = bp["ensemble_size"]

    train_df = load_train()
    tr_idx, va_idx, _ = time_window_split(
        train_df["Molecule Name"].tolist(), train_pct=0.7, val_pct=0.15)
    val_smis = train_df["SMILES"].iloc[va_idx].tolist()
    print(f"Backfilling Chemprop val predictions on "
          f"{len(val_smis)} time-window val molecules")

    # accumulator[short_name] = list of (n_val,) prediction arrays
    accumulator: dict[str, list[np.ndarray]] = {}

    n_done = 0
    n_skipped = 0
    for cluster_name, endpoints in cfg.TASK_GROUPS.items():
        short_cols = [cfg.ENDPOINT_PREP[a][2] for a in endpoints]
        for seed in range(n_seeds):
            work_dir = CHEMPROP_DIR / f"{cluster_name}_seed{seed}"
            ckpt = _find_best_ckpt(work_dir)
            if ckpt is None:
                print(f"  SKIP {cluster_name}_seed{seed}: no checkpoint at "
                      f"{work_dir}/lightning_logs/.../checkpoints/")
                n_skipped += 1
                continue

            # 1) reconstruct architecture (same as train_multitask)
            mp = BondMessagePassing(
                depth=bp["depth"], d_h=bp["hidden_size"], dropout=bp["dropout"])
            agg = MeanAggregation()
            ffn = RegressionFFN(n_tasks=len(short_cols))
            model = MPNN(mp, agg, ffn, batch_norm=True, metrics=[MAE()])

            # 2) load weights from Lightning checkpoint
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
            except TypeError:
                # older pytorch without weights_only kwarg
                state = torch.load(ckpt, map_location="cpu")
            sd = state.get("state_dict", state)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                # Chemprop sometimes wraps with extra top-level keys; warn but continue
                if missing:
                    print(f"    note: {len(missing)} missing keys "
                          f"(first: {missing[0] if missing else None})")
                if unexpected:
                    print(f"    note: {len(unexpected)} unexpected keys "
                          f"(first: {unexpected[0] if unexpected else None})")
            model.eval()

            # 3) build val dataloader (placeholder targets; not used for prediction)
            zeros = np.zeros((len(val_smis), len(short_cols)), dtype=np.float32)
            val_data = [MoleculeDatapoint.from_smi(s, t)
                        for s, t in zip(val_smis, zeros)]
            val_dset = MoleculeDataset(val_data, Featurizer())
            val_loader = DataLoader(
                val_dset, batch_size=bp["batch_size"], shuffle=False,
                num_workers=2, collate_fn=collate_batch)

            # 4) predict
            trainer = pl.Trainer(
                accelerator="auto", devices=1,
                enable_progress_bar=False, logger=False,
                enable_model_summary=False)
            preds_per_batch = trainer.predict(model, val_loader)
            preds = torch.cat(preds_per_batch).cpu().numpy()
            assert preds.shape == (len(val_smis), len(short_cols)), \
                f"unexpected pred shape {preds.shape}"

            # 5) accumulate per-endpoint predictions
            for j, short in enumerate(short_cols):
                accumulator.setdefault(short, []).append(preds[:, j])

            n_done += 1
            print(f"  ✓ {cluster_name}_seed{seed}  "
                  f"(checkpoint: {ckpt.relative_to(cfg.V3_OUT)})")

    if n_done == 0:
        print("\nNo Chemprop checkpoints found. Did training succeed?")
        sys.exit(1)

    # 6) average across (cluster × seed) and write
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting averaged val predictions for {len(accumulator)} endpoints "
          f"({n_done} ckpt loaded, {n_skipped} skipped)")
    for short, arrs in sorted(accumulator.items()):
        avg = np.mean(np.stack(arrs), axis=0)
        out_path = PRED_DIR / f"chemprop__{short}__val.npz"
        np.savez_compressed(out_path, X=avg)
        print(f"  {out_path.name}  ({len(arrs)} preds averaged)")

    print("\nDone. Re-run the ensemble step to pick up Chemprop:")
    print("  python -m src.v3.run --resume --skip-chemprop")


if __name__ == "__main__":
    main()
