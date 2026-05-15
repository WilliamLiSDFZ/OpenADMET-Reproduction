"""Fingerprint branch: small MLP over MACCS(167) ⊕ ErG(315) ⊕ PubChem(881)
= 1363 input bits, outputs a 256-dim embedding to concat with the other
two branches.

HybridADMET don't publish the exact MLP hyperparameters; we use a
2-hidden-layer MLP with GELU + dropout, which is the default Pytorch
recipe for small dense feature stacks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FingerprintBranch(nn.Module):
    def __init__(self,
                 in_dim: int = 167 + 315 + 881,   # 1363
                 out_dim: int = 256,
                 hidden_dim: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        # fp: (n_mol, in_dim) float32 — 0/1 for MACCS/PubChem, float for ErG
        return self.net(fp)
