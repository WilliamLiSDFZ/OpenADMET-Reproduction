"""HybridADMET top-level model: three branches concatenated, fed to a
multi-task MLP head that predicts ``n_targets`` log-space endpoint values
per molecule.

Each forward sees a dict ``inputs`` with three pre-computed parts:
    inputs["unimol_batch"]  — collated batch dict for Uni-Mol2
    inputs["pamnet_data"]   — torch_geometric Batch object for PAMNet
    inputs["fp"]            — (B, 1363) float tensor for fingerprint branch
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """A 2-layer MLP that splits into per-task heads at the last layer."""
    def __init__(self, in_dim: int, n_tasks: int,
                 hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, in_dim) -> (B, n_tasks)"""
        z = self.trunk(h)
        return torch.cat([head(z) for head in self.heads], dim=-1)


class HybridADMET(nn.Module):
    """Three-branch backbone + multi-task head, end-to-end trainable.

    Args:
        unimol_branch: pre-constructed UniMol2Branch
        pamnet_branch: pre-constructed PAMNetBranch
        fp_branch:     pre-constructed FingerprintBranch
        n_tasks:       number of endpoints this model predicts
    """
    def __init__(self, unimol_branch, pamnet_branch, fp_branch,
                 n_tasks: int, head_hidden: int = 512,
                 head_dropout: float = 0.2):
        super().__init__()
        self.unimol = unimol_branch
        self.pamnet = pamnet_branch
        self.fp     = fp_branch

        concat_dim = unimol_branch.hidden_dim + pamnet_branch.dim + fp_branch.out_dim
        self.head = MultiTaskHead(concat_dim, n_tasks,
                                   hidden_dim=head_hidden, dropout=head_dropout)
        self.n_tasks = n_tasks
        self.concat_dim = concat_dim

    def forward(self, inputs: dict) -> torch.Tensor:
        """Returns (B, n_tasks) log-space predictions."""
        h_unimol = self.unimol(inputs["unimol_batch"])    # (B, 512)
        h_pamnet = self.pamnet(inputs["pamnet_data"])     # (B, dim)
        h_fp     = self.fp(inputs["fp"])                  # (B, 256)
        h = torch.cat([h_unimol, h_pamnet, h_fp], dim=-1)
        return self.head(h)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Multi-task MAE that ignores NaN target entries.

    pred, target, mask are all (B, n_tasks). mask is 1 where target is valid.
    Computes per-task MAE only over valid entries, then averages across tasks
    (so tasks with no valid samples in this batch don't dominate).
    """
    diff = (pred - target).abs() * mask
    per_task_sum = diff.sum(dim=0)             # (n_tasks,)
    per_task_count = mask.sum(dim=0) + eps     # (n_tasks,)
    per_task_mae = per_task_sum / per_task_count
    # Tasks with zero valid samples contribute 0 to the mean
    valid_tasks = (mask.sum(dim=0) > 0).float()
    total_valid_tasks = valid_tasks.sum().clamp(min=1.0)
    return (per_task_mae * valid_tasks).sum() / total_valid_tasks
