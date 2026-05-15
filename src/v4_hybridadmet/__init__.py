"""v4_hybridadmet: faithful reproduction of the HybridADMET team's
submission (Shuo Zhang, Lianjin Cai, et al., reported MA-RAE ≈ 0.60).

Their architecture is three heterogeneous branches fine-tuned end-to-end
with a shared multi-task head:

   SMILES ─┬─→ Uni-Mol2 (transformer, 3D-aware, pretrained) ─┐
           ├─→ PAMNet     (physics-aware multiplex GNN)       ├─→ concat ─→ MLP head ─→ y(1..9)
           └─→ MACCS+ErG+PubChem fingerprints (small MLP)    ─┘

All three branches' parameters are updated end-to-end under a single
masked multi-task MAE loss. Per-endpoint multi-task grouping follows
HybridADMET's table (9 endpoint configs → 5 unique training-data tuples).
5-fold scaffold split + 5-seed ensemble.

Entry point::

    python -m src.v4_hybridadmet.run
"""
