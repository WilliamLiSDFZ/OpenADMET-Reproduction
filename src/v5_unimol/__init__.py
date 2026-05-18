"""v5_unimol: Uni-Mol2-only ablation of v4_hybridadmet.

Motivation
----------
v4 was a faithful 3-branch HybridADMET reproduction (Uni-Mol2 + PAMNet +
PaDEL fingerprints). It scored MA-RAE = 0.688 — worse than v3 (0.666) and
far from HybridADMET's reported 0.60.

The original HybridADMET paper writes (resource/others_focus.txt §Performance
comments):

    "Adding the fingerprint branch yields a notable improvement for Efflux
    and Papp. For other endpoints, fingerprints provide limited gains;
    however, the Uni-Mol2 + PAMNet backbone already achieves performance
    comparable to our final submissions."

So fingerprints are mostly dead weight. And one of the failure modes we
diagnosed in v4 (§13.6 hypothesis #2/#3 of REPORT.md) was the PAMNet branch
likely not converging well under single-conformer ETKDG embeddings, plus
the PaDEL Java backend silently producing zero vectors for failed SMILES.

v5 strips v4 down to **just the Uni-Mol2 transformer**: Uni-Mol2 CLS
representation (512-d) → multi-task MLP head → masked-MAE loss. Same 5
HybridADMET dedup target groups, same scaffold 5-fold split, same end-to-end
fine-tuning, same per-endpoint primary-group aggregation as v4.

If v5 reaches the v4-equivalent or beats it, that's strong evidence that
v4's PAMNet + fingerprint branches were the bottleneck. If v5 also beats
v4.1 stitch (0.651), then we have a single-model new state of the art.

Layout
------
config.py           — paths, dedup groups, Uni-Mol2 settings only
data.py             — Uni-Mol2 input packing + Dataset (no PyG, no fp)
splits.py           — scaffold 5-fold (copy of v4)
models/
  unimol_branch.py  — reused from v4 (imported directly)
  unimol_only.py    — UniMol2 + MultiTaskHead + masked_mae_loss
train.py            — train_all() with resume + fast-path inference
predict.py          — per-endpoint primary-group aggregation
run.py              — CLI entry point
"""
__all__ = ["config", "data", "splits", "models", "train", "predict", "run"]
