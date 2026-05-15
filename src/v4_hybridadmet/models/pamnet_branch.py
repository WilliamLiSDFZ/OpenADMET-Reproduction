"""PAMNet branch — wraps the verbatim PAMNet model from
``src/v4_hybridadmet/pamnet/`` and exposes a per-molecule hidden vector
(not the scalar regression output the original returned).

The original PAMNet.forward does:
    ...message passing...
    out = (out * att_weight).sum(dim=-1)
    out = out.sum(dim=0).unsqueeze(-1)        # (n_atoms, 1)
    out = global_add_pool(out, batch)         # (n_mol, 1)
    return out.view(-1)                        # scalar per molecule

We override that tail end: instead of pooling a per-atom scalar to a
per-molecule scalar, we pool the per-atom hidden vector ``x`` (shape
(n_atoms, dim)) to a per-molecule hidden vector (shape (n_mol, dim)),
which becomes our branch output for downstream concat with Uni-Mol2 + FP.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Defer the torch_geometric / torch_sparse imports until first call so that
# users without GPU dependencies can still import the package.
class PAMNetBranch(nn.Module):
    """Wraps PAMNet, returns ``(n_mol, dim)`` hidden vectors.

    Args:
        cfg_kwargs: kwargs passed to ``pamnet.Config`` --
            ``dim``, ``n_layer``, ``cutoff_l``, ``cutoff_g``.
            ``dataset`` is hard-coded to "QM9" since we operate on small
            molecules with atomic number labels.
    """
    def __init__(self, **cfg_kwargs):
        super().__init__()
        from torch_geometric.nn import global_mean_pool
        from ..pamnet import PAMNet, Config

        config = Config(dataset="QM9", **cfg_kwargs)
        self.model = PAMNet(config)
        self.dim = config.dim
        self._global_mean_pool = global_mean_pool
        # We monkey-patch the model's forward to expose hidden vectors
        # rather than rewriting all of PAMNet's geometry pipeline.

    def forward(self, data):
        """data is a torch_geometric.data.Batch with .x (atomic numbers),
        .pos (3D coords), .edge_index, .batch.

        Returns: (n_mol, dim) hidden tensor.
        """
        from torch_geometric.nn import radius
        from torch_geometric.utils import remove_self_loops

        m = self.model
        x_raw = data.x
        batch = data.batch
        edge_index_l = data.edge_index
        pos = data.pos

        # Atomic number → learned embedding (PAMNet QM9 path)
        # The original code uses a 5-row embedding indexed by atomic number,
        # which is enough for H/C/N/O/F. For drug-like molecules we have
        # S, Cl, P, Br, I, etc. — extend the embedding table at init time
        # before running on real data. Here we clip to the table size as a
        # robust fallback (rare atoms get treated as a generic atom).
        idx = torch.clamp(x_raw.long(), max=m.embeddings.shape[0] - 1)
        x = torch.index_select(m.embeddings, 0, idx)

        # Global edges (radius graph)
        row, col = radius(pos, pos, m.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j, i = edge_index_g
        dist_g = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Local edges (provided as data.edge_index, from MoleculeNet)
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j, i = edge_index_l
        dist_l = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Triplet indices (for angular terms in local layer)
        (idx_i, idx_j, idx_k, idx_kj, idx_ji,
         idx_i_pair, idx_j1_pair, idx_j2_pair,
         idx_jj_pair, idx_ji_pair) = m.indices(edge_index_l, num_nodes=x.size(0))

        # Two-hop angles
        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj, dim=-1).norm(dim=-1)
        angle2 = torch.atan2(b, a)

        # One-hop angles
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair = pos_j1_pair - pos_i_pair
        pos_jj_pair = pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair, dim=-1).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        # Bessel / spherical bessel basis
        rbf_l = m.rbf_l(dist_l)
        rbf_g = m.rbf_g(dist_g)
        sbf1 = m.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = m.sbf(dist_l, angle2, idx_kj)

        edge_attr_rbf_l = m.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = m.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = m.mlp_sbf1(sbf1)
        edge_attr_sbf2 = m.mlp_sbf2(sbf2)

        # Global + local message passing (we ignore the per-layer scalar
        # outputs that the original used for the fusion; we just want the
        # final per-atom hidden state in x).
        for layer in range(m.n_layer):
            x, _, _ = m.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            x, _, _ = m.local_layer[layer](
                x, edge_attr_rbf_l, edge_attr_sbf2, edge_attr_sbf1,
                idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index_l,
            )

        # Pool per-atom → per-molecule
        h_mol = self._global_mean_pool(x, batch)         # (n_mol, dim)
        return h_mol


# ----- Atomic-number embedding extension --------------------------------
def extend_pamnet_embedding(branch: PAMNetBranch, max_atomic_number: int = 53):
    """Extend PAMNet's atomic-number embedding table to fit drug-like atoms.

    The original table only has 5 rows (H/C/N/O/F for QM9).  For ADMET data
    we see at least up to Iodine (Z=53).  Call this AFTER construction
    and BEFORE training so the new rows are part of the optimizer.
    """
    import math
    dim = branch.model.dim
    new_emb = nn.Parameter(torch.ones((max_atomic_number + 1, dim)))
    stdv = math.sqrt(3)
    new_emb.data.uniform_(-stdv, stdv)
    # Preserve any already-learned rows in the original 5-row table
    n_old = branch.model.embeddings.shape[0]
    new_emb.data[:n_old] = branch.model.embeddings.data
    branch.model.embeddings = new_emb
    return branch
