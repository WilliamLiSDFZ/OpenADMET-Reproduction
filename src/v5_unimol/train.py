"""End-to-end fine-tuning of Uni-Mol2 + multi-task head (v5).

Per-group, per-seed loop is identical to v4_hybridadmet.train, but:
  * No PAMNet branch construction
  * No fingerprint branch construction
  * No fingerprint precomputation
  * No PyG Batch in collate
  * Fewer trainable params (no PAMNet + no FP MLP) → smaller GPU footprint
  * Single conformer is still embedded once via ETKDG (used as Uni-Mol2 input)
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .data import (UniMolDataset, build_targets, collate_unimol,
                   precompute_features)
from .models import UniMol2Branch, UnimolOnly, masked_mae_loss
from .splits import scaffold_kfold


# ----- Optimizer + cosine warmup --------------------------------------------
def _make_optimizer(model, lr=cfg.LR, wd=cfg.WEIGHT_DECAY):
    no_decay = {"bias", "LayerNorm.weight"}
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)]
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": wd},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )


def _lr_schedule(epoch: int, total_epochs: int = cfg.EPOCHS,
                 warmup_epochs: int = cfg.WARMUP_EPOCHS) -> float:
    """Linear warmup + cosine decay multiplier on the base LR."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ----- Build the v5 model from scratch (used by train + fast-path) ---------
def _build_model(n_tasks: int, device: torch.device) -> UnimolOnly:
    unimol = UniMol2Branch(model_variant=cfg.UNIMOL_MODEL_VARIANT,
                            freeze=not cfg.UNIMOL_FINETUNE)
    model = UnimolOnly(
        unimol, n_tasks=n_tasks,
        head_hidden=cfg.HEAD_HIDDEN, head_dropout=cfg.HEAD_DROPOUT,
    ).to(device)
    return model


# ----- One full v5 training run ---------------------------------------------
def train_one(group_name: str,
              targets_in_group: List[str],
              molecule_features,
              train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              test_features,
              seed: int,
              device: torch.device,
              ) -> dict:
    """Train one Uni-Mol2-only model on the given target subset, return test
    predictions for those targets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = cfg.CKPT_DIR / f"{group_name}_seed{seed}.pt"

    # ---- Fast-path: if a checkpoint already exists from a previous run,
    #      just load it and run inference on test.
    if ckpt_path.exists() and ckpt_path.stat().st_size > 1024:
        print(f"\n=== group={group_name} seed={seed}  "
              f"FAST-PATH from {ckpt_path.name} (skip training) ===")
        try:
            return _inference_from_checkpoint(
                ckpt_path=ckpt_path,
                targets_in_group=targets_in_group,
                test_features=test_features,
                seed=seed,
                group_name=group_name,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: fast-path failed ({type(e).__name__}: {e}); "
                  "falling back to full retrain")

    print(f"\n=== group={group_name} seed={seed}  targets={targets_in_group} ===")

    # ---- Build targets + mask --------------------------------------------
    y_train, mask_train = build_targets(train_df, targets_in_group)

    # ---- 5-fold scaffold split: use fold 0 as held-out val for early stop -
    smiles_train = train_df["SMILES"].tolist()
    folds = list(scaffold_kfold(smiles_train, n_splits=5, seed=seed))
    tr_idx, val_idx = folds[0]

    # ---- Datasets & DataLoaders ------------------------------------------
    tr_ds = UniMolDataset(
        [molecule_features[i] for i in tr_idx],
        y_train[tr_idx], mask_train[tr_idx])
    va_ds = UniMolDataset(
        [molecule_features[i] for i in val_idx],
        y_train[val_idx], mask_train[val_idx])
    te_ds = UniMolDataset(
        test_features,
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32),
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32))

    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                           collate_fn=collate_unimol, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_unimol, num_workers=2)
    te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_unimol, num_workers=2)

    # ---- Build model ------------------------------------------------------
    model = _build_model(n_tasks=len(targets_in_group), device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params/1e6:.1f} M  "
          f"(v4 was ~5-7 M with PAMNet+FP, v5 should be ~3-5 M smaller)")

    optimizer = _make_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- Training loop ----------------------------------------------------
    best_val_mae = float("inf")
    patience = 10
    bad_epochs = 0

    for epoch in range(cfg.EPOCHS):
        lr_mult = _lr_schedule(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.LR * lr_mult

        # ---- train one epoch ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in tr_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model({"unimol_batch": batch["unimol_batch"]})
                loss = masked_mae_loss(pred, batch["y"], batch["mask"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        # ---- validate ----
        model.eval()
        val_diff_sum = torch.zeros(len(targets_in_group), device=device)
        val_count    = torch.zeros(len(targets_in_group), device=device)
        with torch.no_grad():
            for batch in va_loader:
                batch = _to_device(batch, device)
                pred = model({"unimol_batch": batch["unimol_batch"]})
                val_diff_sum += ((pred - batch["y"]).abs() * batch["mask"]).sum(dim=0)
                val_count    += batch["mask"].sum(dim=0)
        per_task_mae = (val_diff_sum / val_count.clamp(min=1)).cpu().numpy()
        val_mae = float(np.nanmean(per_task_mae))

        print(f"  epoch {epoch:3d}  lr={cfg.LR*lr_mult:.1e}  "
              f"train={train_loss:.4f}  val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(),
                        "targets": targets_in_group,
                        "epoch": epoch,
                        "val_mae": val_mae}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch} (no improvement {patience})")
                break

    # ---- Load best checkpoint and predict on test ------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = _to_device(batch, device)
            pred = model({"unimol_batch": batch["unimol_batch"]})
            test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    return {
        "targets": targets_in_group,
        "test_preds": test_preds,
        "best_val_mae": ckpt["val_mae"],
        "ckpt_path": str(ckpt_path),
        "seed": seed,
        "group_name": group_name,
    }


def _inference_from_checkpoint(ckpt_path, targets_in_group, test_features,
                               seed, group_name, device):
    """Skip training when a checkpoint already exists. Rebuild the model,
    load weights, predict on test."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model(n_tasks=len(targets_in_group), device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    te_ds = UniMolDataset(
        test_features,
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32),
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32),
    )
    te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_unimol, num_workers=2)
    test_preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = _to_device(batch, device)
            pred = model({"unimol_batch": batch["unimol_batch"]})
            test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    print(f"  ✓ fast-path inference done  (val_mae from ckpt: "
          f"{ckpt.get('val_mae', 'N/A')})")
    return {
        "targets": targets_in_group,
        "test_preds": test_preds,
        "best_val_mae": ckpt.get("val_mae", float("nan")),
        "ckpt_path": str(ckpt_path),
        "seed": seed,
        "group_name": group_name,
    }


def _to_device(batch, device):
    """Move every tensor in a collated batch dict to ``device``."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv
                      for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        elif hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ----- Main: train all 5 unique groups × all seeds --------------------------
def train_all(train_df, test_df, device=None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v5] Using device: {device}")

    # 1) Precompute 3D conformers (NO fingerprints — that's the v5 point)
    mol_train = precompute_features(
        train_df["SMILES"].tolist(),
        cache_path=cfg.CACHE_DIR / "mol_features_train.pkl")
    mol_test  = precompute_features(
        test_df["SMILES"].tolist(),
        cache_path=cfg.CACHE_DIR / "mol_features_test.pkl")

    # 2) For each unique HybridADMET group, train ENSEMBLE_SEEDS models.
    #
    # RESUME: incrementally save ``results_partial.pkl`` after every
    # (group, seed) completes. Restart picks up where we left off.
    partial_path = cfg.V5_OUT / "results_partial.pkl"
    results: dict = {}
    if partial_path.exists():
        try:
            with open(partial_path, "rb") as f:
                results = pickle.load(f)
            done = sum(len(v) for v in results.values())
            print(f"  RESUME: loaded {done} previously-completed (group, seed) "
                  f"pairs from {partial_path}")
        except Exception as e:  # noqa: BLE001
            print(f"  Couldn't read {partial_path} ({e}); starting fresh")
            results = {}

    for group_name, targets in cfg.GROUP_TO_TARGETS.items():
        if group_name not in results:
            results[group_name] = []
        done_seeds = {r.get("seed") for r in results[group_name]}
        for seed in range(cfg.ENSEMBLE_SEEDS):
            if seed in done_seeds:
                print(f"  SKIP group={group_name} seed={seed} (cached)")
                continue
            r = train_one(
                group_name=group_name,
                targets_in_group=targets,
                molecule_features=mol_train,
                train_df=train_df,
                test_df=test_df,
                test_features=mol_test,
                seed=seed,
                device=device,
            )
            results[group_name].append(r)
            with open(partial_path, "wb") as f:
                pickle.dump(results, f)
    return results
