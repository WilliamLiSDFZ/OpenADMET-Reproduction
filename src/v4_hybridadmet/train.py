"""End-to-end joint training of the HybridADMET 3-branch model.

Per-endpoint multi-task grouping: for each of the 5 unique HybridADMET
target tuples, we train one model with all those tasks as outputs. After
training we extract each "primary" endpoint's prediction from its
designated group's model — same logic as our v3 ``PRIMARY_GROUP_FOR_ENDPOINT``
mechanism, but here the per-group model is a full HybridADMET stack
(Uni-Mol2 + PAMNet + fingerprints + multi-task head).

Across each group we run an ``ENSEMBLE_SEEDS``-seed ensemble (default 5).

Output: per-group, per-seed checkpoints + predictions on test.csv saved
to ``output/v4_hybridadmet/checkpoints/`` and ``.../preds/``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .data import (HybridADMETDataset, build_targets, collate_hybrid,
                   precompute_features)
from .features.fingerprints import compute_fingerprints_batch, FP_DIM
from .models import FingerprintBranch, HybridADMET, PAMNetBranch, UniMol2Branch
from .models.hybrid_model import masked_mae_loss
from .models.pamnet_branch import extend_pamnet_embedding
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


# ----- One full HybridADMET training run ------------------------------------
def train_one(group_name: str,
              targets_in_group: List[str],
              molecule_features,
              fingerprints: np.ndarray,
              train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              test_features,
              test_fingerprints: np.ndarray,
              seed: int,
              device: torch.device,
              ) -> dict:
    """Train one HybridADMET model on the given target subset, return test
    predictions for those targets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = cfg.CKPT_DIR / f"{group_name}_seed{seed}.pt"

    # ---- Fast-path: if a checkpoint already exists from a previous run,
    #      just load it and run inference on test. Skips ~40 min of retrain.
    if ckpt_path.exists() and ckpt_path.stat().st_size > 1024:
        print(f"\n=== group={group_name} seed={seed}  "
              f"FAST-PATH from {ckpt_path.name} (skip training) ===")
        try:
            return _inference_from_checkpoint(
                ckpt_path=ckpt_path,
                targets_in_group=targets_in_group,
                test_features=test_features,
                test_fingerprints=test_fingerprints,
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
    # HybridADMET note "5-fold scaffold split" but we don't run all 5 folds
    # for the final submission (they used the held-out test set via the
    # leaderboard); we use fold 0 as a val and train on the other 4 folds.
    # If you want full 5-fold CV, wrap this in a loop and average.
    smiles_train = train_df["SMILES"].tolist()
    folds = list(scaffold_kfold(smiles_train, n_splits=5, seed=seed))
    tr_idx, val_idx = folds[0]

    # ---- Datasets & DataLoaders ------------------------------------------
    tr_ds = HybridADMETDataset(
        [molecule_features[i] for i in tr_idx],
        fingerprints[tr_idx], y_train[tr_idx], mask_train[tr_idx])
    va_ds = HybridADMETDataset(
        [molecule_features[i] for i in val_idx],
        fingerprints[val_idx], y_train[val_idx], mask_train[val_idx])
    te_ds = HybridADMETDataset(
        test_features,
        test_fingerprints,
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32),
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32))

    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                           collate_fn=collate_hybrid, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_hybrid, num_workers=2)
    te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_hybrid, num_workers=2)

    # ---- Build model ------------------------------------------------------
    unimol = UniMol2Branch(model_variant=cfg.UNIMOL_MODEL_VARIANT,
                            freeze=not cfg.UNIMOL_FINETUNE)
    pamnet = PAMNetBranch(**cfg.PAMNET_CFG)
    pamnet = extend_pamnet_embedding(pamnet, max_atomic_number=53)
    fp_branch = FingerprintBranch(in_dim=FP_DIM, out_dim=cfg.DIM_FP_OUT)
    model = HybridADMET(unimol, pamnet, fp_branch,
                         n_tasks=len(targets_in_group),
                         head_hidden=cfg.HEAD_HIDDEN,
                         head_dropout=cfg.HEAD_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params/1e6:.1f} M")

    optimizer = _make_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- Training loop ----------------------------------------------------
    best_val_mae = float("inf")
    patience = 10
    bad_epochs = 0
    # ckpt_path already defined at top of train_one

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
                pred = model({
                    "unimol_batch": batch["unimol_batch"],
                    "pamnet_data": batch["pamnet_data"],
                    "fp": batch["fp"],
                })
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
                pred = model({
                    "unimol_batch": batch["unimol_batch"],
                    "pamnet_data": batch["pamnet_data"],
                    "fp": batch["fp"],
                })
                val_diff_sum += ((pred - batch["y"]).abs() * batch["mask"]).sum(dim=0)
                val_count    += batch["mask"].sum(dim=0)
        per_task_mae = (val_diff_sum / val_count.clamp(min=1)).cpu().numpy()
        val_mae = float(np.nanmean(per_task_mae))

        msg = (f"  epoch {epoch:3d}  lr={cfg.LR*lr_mult:.1e}  "
               f"train={train_loss:.4f}  val_mae={val_mae:.4f}")
        print(msg)

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
    # (ckpt_path already defined at the top of train_one for fast-path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = _to_device(batch, device)
            pred = model({
                "unimol_batch": batch["unimol_batch"],
                "pamnet_data": batch["pamnet_data"],
                "fp": batch["fp"],
            })
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
                               test_fingerprints, seed, group_name, device):
    """Skip training when a checkpoint already exists. Rebuild the model
    architecture, load weights, predict on test, return the same dict
    structure ``train_one`` would have returned.

    Used by the resume path so that a container restart only re-trains
    (group, seed) pairs that don't yet have a .pt on disk.
    """
    import numpy as np
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Rebuild the architecture identically to the training path
    unimol = UniMol2Branch(model_variant=cfg.UNIMOL_MODEL_VARIANT,
                            freeze=not cfg.UNIMOL_FINETUNE)
    pamnet = PAMNetBranch(**cfg.PAMNET_CFG)
    pamnet = extend_pamnet_embedding(pamnet, max_atomic_number=53)
    fp_branch = FingerprintBranch(in_dim=FP_DIM, out_dim=cfg.DIM_FP_OUT)
    model = HybridADMET(unimol, pamnet, fp_branch,
                         n_tasks=len(targets_in_group),
                         head_hidden=cfg.HEAD_HIDDEN,
                         head_dropout=cfg.HEAD_DROPOUT).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    te_ds = HybridADMETDataset(
        test_features, test_fingerprints,
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32),
        np.zeros((len(test_features), len(targets_in_group)), dtype=np.float32),
    )
    te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_hybrid, num_workers=2)
    test_preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = _to_device(batch, device)
            pred = model({
                "unimol_batch": batch["unimol_batch"],
                "pamnet_data": batch["pamnet_data"],
                "fp": batch["fp"],
            })
            test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    print(f"  ✓ fast-path inference done  (val_mae from ckpt: {ckpt.get('val_mae', 'N/A')})")
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
    print(f"[v4] Using device: {device}")

    # 1) Precompute molecule features (3D conformers + PyG + Uni-Mol2 input)
    mol_train = precompute_features(
        train_df["SMILES"].tolist(),
        cache_path=cfg.CACHE_DIR / "mol_features_train.pkl")
    mol_test  = precompute_features(
        test_df["SMILES"].tolist(),
        cache_path=cfg.CACHE_DIR / "mol_features_test.pkl")

    # 2) Precompute fingerprints (MACCS + ErG + PubChem)
    fp_train = compute_fingerprints_batch(
        train_df["SMILES"].tolist(),
        cache_path=cfg.CACHE_DIR / "fp_train.npy")
    fp_test  = compute_fingerprints_batch(
        test_df["SMILES"].tolist(),
        cache_path=cfg.CACHE_DIR / "fp_test.npy")

    # 3) For each unique HybridADMET group, train ENSEMBLE_SEEDS models
    #
    # RESUME: we incrementally save ``results_partial.pkl`` after every
    # (group, seed) completes. If the container restarts mid-run we can
    # pick up exactly where we left off rather than redoing everything.
    import pickle
    partial_path = cfg.V4_OUT / "results_partial.pkl"
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
                fingerprints=fp_train,
                train_df=train_df,
                test_df=test_df,
                test_features=mol_test,
                test_fingerprints=fp_test,
                seed=seed,
                device=device,
            )
            results[group_name].append(r)
            # Save after EVERY completion -- next restart picks up here
            with open(partial_path, "wb") as f:
                pickle.dump(results, f)
    return results
