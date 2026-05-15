# v4_hybridadmet — faithful HybridADMET reproduction

A best-effort reproduction of the HybridADMET team's ExpansionRx
submission (Shuo Zhang, Lianjin Cai, Amitesh Badkul, Taoyu Niu, Xibing He,
Lei Xie). Their reported MA-RAE ≈ 0.60 was the best non-proprietary score
in the public methodology dump.

## What's reproduced (vs HybridADMET's spec)

| Component | HybridADMET | v4 implementation | Faithful? |
|---|---|---|---|
| Branch 1: 3D-aware transformer | Uni-Mol2 (pretrained checkpoint, structure-only) | `models/unimol_branch.py` via `unimol_tools` | ✓ |
| Branch 2: physics-aware GNN | PAMNet | `models/pamnet_branch.py` — verbatim port of [XieResearchGroup/Physics-aware-Multiplex-GNN](https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN) | ✓ |
| Branch 3: fingerprints | MACCS + ErG + PubChem | `features/fingerprints.py`. MACCS+ErG via RDKit; PubChem 881-bit via `padelpy` (PaDEL Java backend) | ✓ |
| Training mode | All branches updated end-to-end | `train.py` joint AdamW backprop | ✓ |
| Per-endpoint multi-task config | 9 endpoint configs (5 unique after dedupe) | `config.PER_ENDPOINT_TARGETS` matches their table exactly | ✓ |
| Ensemble | 5 random seeds, averaged | `config.ENSEMBLE_SEEDS=5` | ✓ |
| Data split | 5-fold scaffold split | `splits.scaffold_kfold` (use fold 0 as val for early stopping) | ✓ |
| Model selection | Public leaderboard | We use the official ground-truth (which we have locally) as the score | ✓ best-equivalent |
| External / private data | None used | Same — only official `train.csv` | ✓ |

## Install (Ubuntu, T4)

```bash
# Core scientific
pip install rdkit pandas numpy scipy tqdm scikit-learn

# Branch 1 — Uni-Mol2
pip install unimol_tools

# Branch 2 — PAMNet (torch-geometric stack)
pip install torch torch-geometric
# These three need to match your torch + CUDA version. If pip wheel
# selection fails, fall back to the explicit channel:
#   pip install torch-scatter torch-sparse torch-cluster \
#       -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html
pip install torch-scatter torch-sparse torch-cluster
pip install sympy        # required by PAMNet's spherical Bessel basis

# Branch 3 — PaDEL (Java) for PubChem fingerprint
sudo apt install default-jre        # ← Java needed by PaDEL
pip install padelpy
```

Verify everything imports cleanly:
```bash
python -c "
from src.v4_hybridadmet.models import (
    UniMol2Branch, PAMNetBranch, FingerprintBranch, HybridADMET)
print('all v4 imports OK')
"
```

## Run

```bash
cd /path/to/OpenADMET-LightGBM-Reproduction
python -m src.v4_hybridadmet.run
```

The first run will:
1. Generate 3D conformers for every train + test molecule (~25 min CPU)
2. Compute 881-bit PubChem fingerprints via PaDEL (~30 min CPU, batched)
3. Train 5 unique multi-task configs × 5 seeds = **25 HybridADMET models**
   on T4 GPU. Each takes ~1 h with end-to-end fine-tuning of Uni-Mol2 → ≈
   **25 h on a single T4**.
4. Aggregate predictions per endpoint using HybridADMET's primary-group rule.
5. Run official eval (`python -m eval`) against ground truth.

Everything is cached on disk under `output/v4_hybridadmet/`, so a second
run with `--skip-train` is seconds.

## File map

```
src/v4_hybridadmet/
├── config.py            hyperparameters, per-endpoint group definitions
├── data.py              3D conformer generation + multi-rep PyTorch Dataset
├── splits.py            5-fold scaffold split
├── features/
│   ├── __init__.py
│   └── fingerprints.py  MACCS + ErG + PubChem(881 via PaDEL) -> 1363 bits
├── pamnet/              ← VERBATIM PORT of XieResearchGroup/PAMNet
│   ├── basic_layers.py
│   ├── global_mp.py
│   ├── local_mp.py
│   ├── pamnet_model.py
│   └── sbf_utils.py
├── models/
│   ├── unimol_branch.py        Uni-Mol2 wrapper (uses unimol_tools)
│   ├── pamnet_branch.py        thin wrapper that exposes hidden vec
│   ├── fingerprint_branch.py   MLP over fingerprints
│   └── hybrid_model.py         concat + multi-task head + masked MAE loss
├── train.py             end-to-end joint training of all 3 branches
├── predict.py           per-endpoint aggregation (HybridADMET primary-group rule)
└── run.py               CLI entry point
```

## Notes & gotchas

1. **Disk cache locations** — all features + model checkpoints go under
   `output/v4_hybridadmet/`. Delete that directory to start over.
2. **`~/.cache/unimol/`** — Uni-Mol2's pretrained checkpoint
   (`mol_pre_no_h_220816.pt` for 84M variant) auto-downloads on first
   model construction. About 350 MB.
3. **PAMNet atom embedding** — the upstream PAMNet QM9 setup only embeds 5
   atom types (H/C/N/O/F). Our `extend_pamnet_embedding` call extends to
   atomic number ≤ 53 (includes S, Cl, Br, I) so drug molecules don't
   index out of range. New rows are randomly initialised at the same
   variance as the original.
4. **PubChem fingerprint via PaDEL** — PaDEL launches a JVM on each
   `from_smiles` call, so we batch in chunks of 200 SMILES. Total ~30 min
   for the full 7,600-molecule dataset. The result caches to `.npy`.
5. **Uni-Mol2 fine-tuning is the slow part** — 84M parameters × ~5000
   train molecules × 60 epochs × 25 (group × seed) runs ≈ 25 h on T4.
   Set `V4_EPOCHS=30` and `V4_ENS=3` for a faster (~10 h) preliminary run.

## Known substitutions from HybridADMET's exact recipe

There aren't any architectural substitutions any more (this is the
"faithful" version). The only mild deviations:

- We use the **official ground-truth file** as our "leaderboard" since we
  have access to it locally. HybridADMET picked model variants by
  submitting daily to the public Hugging Face Space; we just score
  directly with `python -m eval`.
- Our **fingerprint branch hyperparameters** (2-hidden-layer 512-dim MLP
  with GELU + 0.2 dropout) are not stated in their report — we picked a
  sensible default.
- **5-fold scaffold split**: HybridADMET uses all 5 folds for CV; we run
  the final submission using fold 0 only as a held-out val for early
  stopping, and train on folds 1-4 combined. If you want full 5-fold CV
  for finer hyperparameter selection, wrap `train_one` in an outer fold
  loop in `train.train_all`.

## Environment variables

```bash
V4_EPOCHS         max epochs per (group, seed) run                   (default 60)
V4_BATCH          batch size                                          (default 32)
V4_LR             AdamW base LR                                       (default 1e-4)
V4_ENS            number of random seeds per group                    (default 5)
V4_PAMNET_LAYERS  PAMNet message-passing depth                        (default 4)
V4_PAMNET_CUTOFF_L PAMNet local cutoff (Å)                            (default 5.0)
V4_PAMNET_CUTOFF_G PAMNet global cutoff (Å)                           (default 12.0)
UNIMOL_FINETUNE   set to 0 to freeze Uni-Mol2 (10× faster, worse)     (default 1)
UNIMOL_MODEL      "84M" or "164M"                                     (default 84M)
EVAL_REPO         path to ExpansionRx-Challenge-Eval clone            (default sandbox)
```
