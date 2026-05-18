"""UnimolOnly top-level model: just Uni-Mol2 + multi-task head.

This is v4's HybridADMET with PAMNet and the fingerprint branch removed.
The MultiTaskHead and masked_mae_loss are byte-identical to v4's so the
v4 vs v5 comparison is a clean ablation of just the two missing branches.

Forward signature still takes a ``dict`` for symmetry with the v4 training
loop, but only reads ``inputs["unimol_batch"]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """A 2-layer MLP that splits into per-task heads at the last layer.
    Identical to v4_hybridadmet.models.hybrid_model.MultiTaskHead.
    """
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


class UnimolOnly(nn.Module):
    """Uni-Mol2 CLS embedding (512-d) → multi-task MLP head.

    Args:
        unimol_branch: pre-constructed UniMol2Branch (v4's wrapper)
        n_tasks:       number of endpoints this model predicts
    """
    def __init__(self, unimol_branch, n_tasks: int,
                 head_hidden: int = 512, head_dropout: float = 0.2):
        super().__init__()
        self.unimol = unimol_branch
        self.head = MultiTaskHead(
            in_dim=unimol_branch.hidden_dim,
            n_tasks=n_tasks,
            hidden_dim=head_hidden,
            dropout=head_dropout,
        )
        self.n_tasks = n_tasks
        self.concat_dim = unimol_branch.hidden_dim   # kept name for symmetry

    def forward(self, inputs: dict) -> torch.Tensor:
        """Returns (B, n_tasks) log-space predictions.

        Reads only ``inputs["unimol_batch"]`` — ignores ``pamnet_data`` or
        ``fp`` keys if present so the v4 training loop can call us as-is.
        """
        h = self.unimol(inputs["unimol_batch"])   # (B, 512)
        return self.head(h)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Multi-task MAE that ignores NaN target entries.

    Byte-identical to v4_hybridadmet.models.hybrid_model.masked_mae_loss.
    pred, target, mask are all (B, n_tasks). mask is 1 where target is valid.
    """
    diff = (pred - target).abs() * mask
    per_task_sum = diff.sum(dim=0)             # (n_tasks,)
    per_task_count = mask.sum(dim=0) + eps     # (n_tasks,)
    per_task_mae = per_task_sum / per_task_count
    valid_tasks = (mask.sum(dim=0) > 0).float()
    total_valid_tasks = valid_tasks.sum().clamp(min=1.0)
    return (per_task_mae * valid_tasks).sum() / total_valid_tasks
