# CLAUDE.md — handoff notes for the next Claude session

This file is read by Claude Code / Cowork when entering this repo.
**Read it before doing anything else.**

The project is **OpenADMET-ExpansionRx Blind Challenge** —
predict 9 ADMET endpoints for 2282 lead-optimization molecules,
ranked by macro-averaged Relative Absolute Error (MA-RAE) on a
held-out test set whose ground truth we already have locally.

---

## 0. Current state in one screen

- **Best result so far: MA-RAE = 0.748** (v3 classical-only ensemble)
- **Top of leaderboard: ~0.4–0.5** (Inductive Bio #1, Chemprop ensembles)
- We have the official ground-truth file at `data/test_ground_truth.csv`,
  so every change should be re-scored with the official `python -m eval`
  in `../ExpansionRx-Challenge-Eval/eval/`.
- **v1** = single LightGBM (RDKit + Morgan), **v2** = ablations of
  Avalon / ADMET-AI distillation / SALI cliff masking,
  **v3** = the GPU-aware multi-model ensemble. Use v3 going forward;
  v1/v2 are kept only because the report `REPORT.md` references them.
- **DO NOT modify `REPORT.md`** unless the user explicitly asks. The
  user wrote a final report and froze it.

---

## 1. Repo layout

```
OpenADMET-LightGBM-Reproduction/
├── README.md           per-version usage guide (v1+v2)
├── REPORT.md           ← FROZEN, do not edit unless asked
├── V3_README.md        v3 setup + how to run on T4
├── CLAUDE.md           this file
├── requirements.txt    minimal deps for v1/v2

├── data/
│   ├── train.csv                 official 5326×11 train set
│   ├── test.csv                  official 2282×2 test set (no labels)
│   ├── test_ground_truth.csv     test set WITH labels (user-provided)
│   └── external/                 GitHub-sourced augmentation (v2)
│       ├── biogen_logS.csv
│       ├── esol_logSolubility.csv
│       └── drugbank_admet_predictions.csv

├── resource/                     reading material for v3 design
│   ├── others.txt                8 top-10 methodology reports
│   ├── 2510.12719v1.pdf          Merck/NVIDIA KERMT paper
│   └── deep-learning-vs-classical-...pdf  rced_nvx JCIM 2025

├── src/                          ALL source code
│   # ─── v1 (legacy baseline; don't extend) ─────────────────────────
│   ├── utils.py                  log transforms, endpoint metadata
│   ├── features.py               RDKit-2d + Morgan
│   ├── train.py / predict.py     v1 entry points
│   ├── holdout_eval.py           80/20 holdout sanity check
│
│   # ─── v2 (ablations; don't extend) ───────────────────────────────
│   ├── features_v2.py            adds Avalon (+optional Mordred)
│   ├── distill_features.py       ADMET-AI distillation features
│   ├── cliff_masking.py          SALI activity-cliff detection
│   ├── external_data.py          row-augmentation loader (used by v2 too)
│   ├── train_augmented.py        v1 + external row augmentation
│   ├── train_v2.py               v2 ablation runner (fixed config)
│   ├── train_v2_compose.py       v2 ablation runner (env-var toggles)
│   ├── precompute_*.py           one-shot featurization helpers
│
│   # ─── v3 (GPU-aware multi-model ensemble; this is the focus) ─────
│   └── v3/
│       ├── __init__.py           docstring
│       ├── config.py             ALL hyperparams + paths in one file
│       ├── data.py               loading + per-endpoint log transform
│       ├── features.py           RDKit + Morgan + Avalon (+ Mordred opt)
│       ├── splits.py             random / scaffold / time-window splits
│       ├── ensemble.py           NNLS-on-simplex weight learning
│       ├── run.py                main entry: `python -m src.v3.run`
│       └── models/
│           ├── lgbm_model.py     LightGBM
│           ├── xgb_model.py      XGBoost
│           ├── catboost_model.py CatBoost
│           ├── rf_model.py       sklearn RandomForest
│           ├── tabpfn_model.py   TabPFN v2  (GPU)
│           └── chemprop_model.py Chemprop v2 multi-task MPNN  (GPU)

└── output/                       all run artefacts
    ├── augmented/                v2 row-augmentation results
    ├── holdout/                  80/20 sanity-check
    ├── v2/                       v2 ablation results
    └── v3/
        ├── features_train.npz / features_test.npz
        ├── per_model_predictions/   (model__endpoint__{val,test}.npz)
        ├── chemprop/                Chemprop work dirs
        ├── ensemble_weights.csv     per-endpoint NNLS weights
        ├── submission_v3.csv        ← final submission
        └── official_eval_v3.csv     ← official-eval scoreboard
```

---

## 2. The v3 pipeline (this is what matters)

### 2.1 Recipe at a glance

```
SMILES  ─▶  RDKit-2d + Morgan + Avalon  (3289-dim cached .npz)
              │
              ├───▶  LightGBM       per endpoint   (CPU,  ~3 min total)
              ├───▶  XGBoost        per endpoint   (CPU,  ~3 min total)
              ├───▶  CatBoost       per endpoint   (CPU,  ~5 min total)
              ├───▶  RandomForest   per endpoint   (CPU,  ~5 min total)
              ├───▶  TabPFN v2 (RDKit-only)        per endpoint   (T4, ~5 min)
              └───▶  Chemprop MPNN  per task cluster × seeds      (T4, ~3-5 hr)

                              │
                              ▼
   per-endpoint NNLS-on-simplex weight learning
   on the time-window validation split (70% train / 15% val by molecule ID)

                              │
                              ▼
   submission_v3.csv  →  python -m eval  →  MA-RAE
```

### 2.2 Run it (T4 host)

```bash
# Sanity-check first (no GPU): ~15 min, expected MA-RAE ≈ 0.748
python -m src.v3.run --classical-only

# Add TabPFN (T4): ~5 min more
TABPFN_DEVICE=cuda python -m src.v3.run --skip-chemprop --resume

# Full pipeline (T4): ~4-6 hours
CHEMPROP_EPOCHS=50 CHEMPROP_ENS=5 TABPFN_DEVICE=cuda \
    python -m src.v3.run --resume
```

`--resume` is **always safe**: every per-model prediction is cached on
disk under `output/v3/per_model_predictions/`. Re-running with
`--resume` skips models whose predictions already exist. If you change
hyperparameters, delete (or zero out) the relevant cache files first.

### 2.3 Common flags

```
--classical-only      skip Chemprop AND TabPFN  (CPU-only)
--skip-chemprop       skip Chemprop (slow GPU step)
--skip-tabpfn         skip TabPFN
--skip-{lgbm,xgb,catboost,rf}    skip individual classical learners
--resume              reuse cached per-model predictions
--mordred             add Mordred descriptors (~3 min extra featurization)
```

---

## 3. Known gotchas (DO NOT REPEAT)

### 3.1 Zero-handling in `data.log_transform_endpoint`

**Default is "tutorial" — `log10((x + 1) * multiplier)`. Keep it that way.**

Earlier I tried Inductive Bio's "half_min" / "ppb_floor" recipes (replace
zero with half-min or 1e-6 before log10). On this dataset that
**catastrophically** broke training: KSOL's smallest non-zero value is
0.0029 μM, so `log10(0.0029 * 1e-6) = -8.85`, a massive outlier vs the
typical -4 of the bulk distribution. MA-RAE jumped from 0.748 → 1.78.

The default "tutorial" recipe gracefully handles zeros via the +1 offset
and produces a clean unimodal log distribution. There's a long comment
in `src/v3/data.py` explaining this, but the takeaway is: **keep
`zero_handling="tutorial"` for everything except LogD**.

### 3.2 Featurization caching keyed by name + use_mordred

`features.cached_features("train", smiles, use_mordred=True)` writes to
`features_train_mord.npz`, vs `features_train.npz` without Mordred.
**Don't mix them** — one is 3289-dim, the other is ~4900-dim, and
mismatched feature dims silently produce garbage predictions. The
classical models load whichever was cached **last**.

### 3.3 The `--resume` cache is not invalidated by config changes

Changing `LGBM_PARAMS` / `XGB_PARAMS` doesn't invalidate the cached
predictions in `output/v3/per_model_predictions/`. If you tune
hyperparameters, **manually clear that directory** (or use
`os.remove` since rm may be blocked on some mount points). I learned
this when zeroing a file produced unreadable npz; the loader now
catches that, but it still reads stale predictions if non-empty.

### 3.4 Chemprop's val predictions are NOT saved

`models/chemprop_model.py` uses Chemprop's internal early-stopping
val split for training, but it doesn't expose those predictions in the
format the ensemble layer expects. So Chemprop currently gets **equal
weight** with the others when its test predictions are present but val
are missing. If this hurts on your time-window validation, the fix is to
generate Chemprop time-window-val predictions explicitly.

### 3.5 Filesystem can't `os.remove` on the mount point

In the original sandbox `os.remove()` returned EPERM on the mount point.
Overwriting works fine (`open(p, "wb")`). On the T4 host this won't be a
problem; mentioning it only if you see weird "operation not permitted"
errors.

### 3.6 RDKit "DEPRECATION WARNING: please use MorganGenerator"

Newer RDKit (≥ 2024.03) prints this once **per molecule** when you call
``AllChem.GetMorganFingerprintAsBitVect``. With 5326 train + 2282 test
molecules that's 7600+ lines of noise that swallows the progress bar.
All `features*.py` files in this repo now (a) call
``RDLogger.DisableLog("rdApp.*")`` at module load and (b) use the new
``rdFingerprintGenerator.GetMorganGenerator`` API. If you see the
warnings come back, someone re-introduced the old API.

### 3.7 The `external_data.py` `EXT_PROFILE` env var

`external_data.AUGMENTATION_LOADERS` reads `EXT_PROFILE` at module
import time. If you `os.environ["EXT_PROFILE"] = "selective"` after
importing the module, you get the default ("all"). v3's `run.py`
re-imports it correctly, but be careful in interactive sessions.

---

## 4. What worked / what didn't (quick reference)

| Variant | MA-RAE | Notes |
|---|---:|---|
| v1 baseline (LightGBM + RDKit + Morgan)      | 0.756 | the OpenADMET tutorial recipe |
| v1 + selective external augmentation         | 0.750 | hand-picked endpoints (LogD/MLM/MPPB) |
| v2 + Avalon                                  | 0.771 | small-N endpoints overfit |
| v2 + ADMET-AI distillation                   | 0.802 | drugbank-bias transferred badly |
| v2 + SALI cliff mask                         | 0.768 | mask removed real SAR signal |
| v2 + Avalon + cliff + selective aug          | 0.757 | best v2 combination |
| **v3 classical-only (LGBM+XGB+CB+RF + NNLS)**| **0.748** | best so far, CPU-only |
| v3 + TabPFN                                  | TBD | T4 needed |
| v3 + Chemprop ensemble                       | TBD | T4 needed |

The key insight was **NNLS-on-simplex per-endpoint weights on a
time-window val split** — it's a tiny, robust meta-learner that lets
better learners win endpoint-by-endpoint, and the time-window split
(rather than random K-fold) reflects the actual train→test
distribution shift in this challenge.

---

## 5. Open tasks ordered by ROI

(roughly from `REPORT.md §7`, refined by what we now know)

1. **Run v3 full pipeline on T4** to validate the architecture.
   Expected MA-RAE: 0.55–0.65 with Chemprop ensemble. (`python -m src.v3.run`)

2. **CheMeleon-pretrained init for Chemprop**. Currently
   `models/chemprop_model.py` trains from scratch. Loading the CheMeleon
   weights would likely save 10-20 epochs of training and improve
   final accuracy (per OpenADME team's report). The chemprop v2 CLI
   supports `--from-pretrained <path>`; need to wire it up.

3. **Fix Chemprop's val predictions** so the NNLS ensemble can weight
   it properly. Right now Chemprop is treated equal-weight when its
   test preds are present but val are missing.

4. **Per-endpoint hyperparameter selection on the time-window val**.
   The current `LGBM_PARAMS` / `XGB_PARAMS` / `CATBOOST_PARAMS` are
   one-size-fits-all. Spending an hour to do small Optuna runs per
   endpoint on the time-window val (NOT on cross-val) might shave
   another 0.01-0.02.

5. **RIGR data augmentation** (resonance-form enumeration). One team
   reported this as their largest single gain. Requires a custom
   resonance enumeration pipeline; non-trivial.

6. **Add Polaris ADMET / Novartis benchmark data** to the augmentation
   pool. We already have biogen / drugbank / ESOL — adding Polaris/Novartis
   would give us coverage for the metabolism endpoints (HLM/MLM CLint).

7. **MA-RAE local validation correlation check**. Right now we trust
   `time_window_split` to mirror the real test shift. Sanity-check
   this by running v3 on, say, the latest 10 % of training data
   held out (E-0019000+ → E-0020100), then see if local MA-RAE on that
   slice tracks the official MA-RAE on the actual test (E-0020101+).
   If correlated, we can iterate on the local val without the test set.

8. **TabPFN with morgan slice instead of RDKit-only** — TabPFN v2 raised
   its feature limit to 500 in the latest release. We currently feed it
   ~210 RDKit-2d features only. Trying ~500 mixed features (e.g.
   PCA(Morgan, 290) + RDKit) might unlock a stronger TabPFN.

---

## 6. Conventions

- All time/file paths are **absolute** (or `Path(__file__)`-relative).
  No reliance on `cwd` because some shells reset it between calls.
- Python invocations use `python -m src.v3.run` (the `-m` form), so the
  imports inside v3 use relative imports cleanly.
- Output directories follow `output/<version>/` to avoid collision.
  v3 outputs go under `output/v3/`. Don't write to `output/` root.
- Submission CSVs always use the original assay column names, NOT the
  short names: `LogD, KSOL, MLM CLint, HLM CLint,
  Caco-2 Permeability Efflux, Caco-2 Permeability Papp A>B, MPPB, MBPB,
  MGMB`. The official `python -m eval` aliases short→long via
  `eval/__main__.py:COLUMN_ALIASES`, but our submission writes long
  names directly to be safe.
- `python -m eval` lives in **`/Users/william/Documents/project/python/ExpansionRx-Challenge-Eval/`**
  (separate repo). The path is hard-coded as `EVAL_REPO` in
  `src/v3/config.py`. **Update it on the server** if the evaluator
  lives elsewhere there. Without it the final eval step in `run.py`
  silently fails.

---

## 7. Verifying the eval scoring

```bash
cd /path/to/ExpansionRx-Challenge-Eval
python -m eval \
    /path/to/OpenADMET-LightGBM-Reproduction/output/v3/submission_v3.csv \
    --ground-truth \
    /path/to/OpenADMET-LightGBM-Reproduction/data/test_ground_truth.csv \
    --output /tmp/v3_eval.csv
cat /tmp/v3_eval.csv
```

The "Macro Average" row's `mean_RAE` column is **MA-RAE**. R² is in
`mean_R2`. Spearman in `mean_Spearman R`.

---

## 8. Frozen vs editable

| File | Status | Why |
|---|---|---|
| `REPORT.md` | **frozen** | The user explicitly asked not to touch it |
| `README.md` | editable | Project usage guide for v1/v2 |
| `V3_README.md` | editable | v3 setup; update with real T4 numbers |
| `CLAUDE.md` (this file) | editable | Always update when you learn something new |
| `src/v3/**` | editable | Active development |
| `src/{features,utils,external_data,distill_features,cliff_masking,...}.py` | editable but legacy | Keep working; don't break |
| `data/**` | read-only | These are inputs; never overwrite |
| `resource/**` | read-only | Reading material |

---

## 9. Cheat-sheet for the next session

```bash
# 1. Sanity-check the env
python -c "import lightgbm, xgboost, catboost, sklearn, rdkit; print('OK')"
python -c "import torch; print('cuda:', torch.cuda.is_available())"

# 2. Where are we?
cat output/v3/official_eval_v3.csv  # current MA-RAE
ls output/v3/per_model_predictions/ | wc -l  # number of cached preds

# 3. Re-run from current state
python -m src.v3.run --resume

# 4. Final score
python -m eval output/v3/submission_v3.csv \
    --ground-truth data/test_ground_truth.csv
```

Good luck. The user is technically sharp, prefers concise answers, and
appreciates honest negative results. When something doesn't work, say
so plainly and explain why — they'd rather hear that than an
overly-optimistic story.

— Claude (handoff written 2026-05-05)
