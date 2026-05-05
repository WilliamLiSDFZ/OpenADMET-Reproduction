# v3 — GPU-accelerated multi-model ensemble

## What this is

A coherent, modular re-implementation that combines the techniques the
top-10 OpenADMET-ExpansionRx submissions used in common:

1. **Chemprop v2 multi-task MPNN**, trained on **task-affinity-grouped**
   endpoint clusters (LogD/LogS/PPB family vs metabolism vs permeability),
   with an **ensemble of seeds** averaged at the prediction level.
2. **Classical ML stack** — LightGBM, XGBoost, CatBoost, Random Forest —
   single-task per endpoint, on RDKit-2d + Morgan + Avalon features.
3. **TabPFN v2** — transformer pre-trained on tabular data, run as a
   single-task in-context learner (RDKit-2d only to fit the 500-feature
   limit).
4. **Per-endpoint NNLS-on-simplex ensemble**, with weights learnt on a
   **time-window validation split** (sort by molecule ID, train on the
   early 70 %, validate on the next 15 %). This mirrors the
   train/test convention of the actual challenge and is far less optimistic
   than random K-fold CV.

Every learner saves its time-window val predictions and its full-data
test predictions to ``output/v3/per_model_predictions/`` so you can
re-run the ensemble step alone, or swap learners in/out, without
re-training the heavy ones.

## Why this is the "best version"

Choices grounded in the methodology reports under ``resource/``:

| Where it came from | What it gives us |
|---|---|
| OpenADME team (#7-ish), MATCHA / Merck, JCIM 2025 | Chemprop v2 multi-task MPNN with task affinity grouping |
| JCIM 2025, multiple top-10 reports | Avalon fingerprint + RDKit-2d + Morgan |
| JCIM 2025 | TabPFN v2 (surprising winner on ADME) |
| Inductive Bio (#1) report | Per-endpoint zero handling (half-min for clearance/permeability, 1e-6 for PPB) |
| Multiple teams | Multi-model weighted-average ensemble |
| MATCHA / OpenADME team | Time-window split for ensemble weight learning |
| v1/v2 experiments here | Selective external row augmentation (best result before v3) |

## Hardware target: NVIDIA T4 (16 GB VRAM)

| Stage | CPU time | T4 time |
|---|---|---|
| Featurization (RDKit + Morgan + Avalon) | ~30 s | — |
| LightGBM × 9 endpoints | ~3 min | — |
| XGBoost × 9 | ~3 min | — |
| CatBoost × 9 | ~5 min | — |
| Random Forest × 9 | ~5 min | — |
| TabPFN × 9 | ~25 min CPU, ~5 min GPU | T4 |
| Chemprop multi-task × 3 clusters × 5 seeds | — | ~3-5 hours |
| Ensemble weight fitting + final predictions | <1 min | — |
| **TOTAL** (with `--classical-only`) | ~15-20 min | — |
| **TOTAL** (full v3 with Chemprop) | — | ~4-6 hours |

## How to run on a T4 host

### 1. Install dependencies
```bash
# Required for everything
pip install rdkit lightgbm xgboost catboost scikit-learn pandas numpy scipy tqdm

# For Chemprop (T4 only)
pip install chemprop torch lightning

# For TabPFN (T4 strongly recommended)
pip install tabpfn

# Optional (slow but adds 1620 features)
pip install mordred-community
```

### 2. Sanity-check on CPU (no GPU needed)
```bash
cd /path/to/OpenADMET-LightGBM-Reproduction
python -m src.v3.run --classical-only
```
This trains LGBM + XGB + CatBoost + RF on every endpoint and runs the
ensemble. **End-to-end ~15 min, no GPU**.

**Verified result on the official ground truth: MA-RAE = 0.748** ←
better than v1 baseline (0.756) and v1+selective_aug (0.750), without
any GPU. All by virtue of the multi-model NNLS-on-simplex ensemble
trained against the time-window validation split.

### 3. Add TabPFN (still no Chemprop)
```bash
TABPFN_DEVICE=cuda python -m src.v3.run --skip-chemprop
```

### 4. The full thing (T4 ~4-6 hours)
```bash
# Suggested defaults for a single T4
CHEMPROP_EPOCHS=50 CHEMPROP_ENS=5 \
TABPFN_DEVICE=cuda \
python -m src.v3.run
```

### 5. Resume after interruption
Every per-model prediction is saved to disk. Re-running with `--resume`
skips models whose predictions already exist:
```bash
python -m src.v3.run --resume
```

## Outputs

```
output/v3/
├── features_train.npz
├── features_test.npz
├── per_model_predictions/
│   ├── lgbm__LogD__val.npz
│   ├── lgbm__LogD__test.npz
│   ├── xgb__LogD__val.npz
│   ├── chemprop__LogD__test.npz       (no _val for chemprop in current run)
│   ├── tabpfn__LogD__val.npz
│   └── ...                             (n_models × n_endpoints × {val,test})
├── chemprop/
│   └── solubility_binding_seed0/      (Chemprop work dirs, predictions.npz)
├── ensemble_weights.csv               per-endpoint weights chosen by NNLS
├── submission_v3.csv                  the final submission file
└── official_eval_v3.csv               MA-RAE / R² / Spearman from official eval
```

## Architectural notes

- **Modules**: `src/v3/` is fully self-contained; doesn't import anything
  from `src/v1` or `src/v2`. Older code stays untouched per request.
- **No `data.py` writes**: the only side effect of loading data is
  creating the cached `.npz` feature files (and the per-model prediction
  cache). You can safely delete `output/v3/` to start over.
- **CPU/GPU split**: GPU code is gated behind `--skip-chemprop` /
  `--skip-tabpfn` flags. The classical stack runs anywhere.
- **Per-endpoint zero handling** (Inductive Bio's recipe) lives in
  `data.log_transform_endpoint`. Replace the `zero_handling` field in
  `config.ENDPOINT_PREP` to change.
- **External augmentation**: NOT enabled by default in v3 because the v2
  experiments showed it was a wash once we have a multi-model ensemble.
  Set `EXT_PROFILE=selective` in env to re-enable on top of v3.

## CPU-only sanity-check results (this sandbox)

For the record, the classical-only v3 stack actually beats every prior
attempt on this dataset, even before adding TabPFN or Chemprop:

| Endpoint | RAE (v1 base) | RAE (v3 classical) | Δ |
|---|---:|---:|---:|
| LogD             | 0.629 | 0.596 | **−0.033** |
| KSOL             | 0.858 | 0.816 | **−0.042** |
| MLM CLint        | 0.929 | 0.916 | −0.013 |
| HLM CLint        | 0.814 | 0.802 | −0.012 |
| Caco-2 Efflux    | 0.774 | 0.789 | +0.015 |
| Caco-2 Papp      | 0.771 | 0.778 | +0.007 |
| MPPB             | 0.872 | 0.886 | +0.014 |
| MBPB             | 0.587 | **0.562** | −0.025 |
| MGMB             | 0.574 | **0.585** | +0.011 |
| **Macro Average** | **0.756** | **0.748** | **−0.008** ✅ |

Per-endpoint weights learned by NNLS-on-simplex on the time-window
validation split:

```
LogD              lgbm:0.42  xgb:0.10  catboost:0.48
KSOL              lgbm:0.21  xgb:0.76  catboost:0.02
MLM CLint         lgbm:0.25  xgb:0.03  catboost:0.71
HLM CLint         lgbm:0.48  catboost:0.52
Caco-2 Efflux     xgb:0.48   catboost:0.52
Caco-2 Papp       xgb:0.63   catboost:0.37
MPPB              lgbm:0.86  xgb:0.14
MBPB              lgbm:0.05  catboost:0.95
MGMB              lgbm:0.58  catboost:0.42
```

Notice that **CatBoost wins for MBPB** (small N, where the boosted-tree
ordering helps), and **XGBoost wins for KSOL** (the most skewed
distribution). RF gets dropped from every ensemble (its prediction
diversity is too correlated with LGBM, so the optimizer prefers
LGBM+CatBoost). On T4, adding Chemprop and TabPFN should give large
gains particularly on the small-N endpoints (MPPB, MBPB, MGMB).

## Known caveats

1. The CheMeleon-pretrained-init for Chemprop is *not* implemented yet.
   The Chemprop wrapper currently trains from scratch. If you have the
   CheMeleon checkpoints locally, point Chemprop at them via its
   `--from-pretrained` flag (see Chemprop v2 docs).
2. Chemprop's time-window val predictions aren't currently saved (we use
   its internal early-stopping val split). That means Chemprop's weight
   in the ensemble can't be learned by NNLS. The default behaviour is to
   give Chemprop **equal weight** with the other models if its test
   predictions are present but val are missing — adjust if that hurts
   on your validation.
3. Mordred is opt-in (`--mordred`) and adds ~3 minutes; the v2
   experiments showed marginal benefit on this dataset.
4. RIGR resonance-form augmentation (one team's biggest single boost) is
   **not** implemented — it requires a custom resonance enumeration
   pipeline that would double the file count of this project.
