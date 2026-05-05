"""v3: GPU-aware multi-model ensemble.

Designed to be run on a single NVIDIA T4 (16 GB VRAM). Core idea is the
recipe shared across the top-10 OpenADMET-ExpansionRx submissions:

  base learners
    1. Chemprop v2 multi-task MPNN, grouped by task affinity (T4)
    2. LightGBM / XGBoost / CatBoost / RandomForest single-task   (CPU)
    3. TabPFN v2 single-task                                       (T4 / CPU)
  ↓
  weighted-average ensemble per endpoint
  weights chosen on a local time-window validation split

Everything in this package is independent of v1/v2 code in src/. Run
``python -m v3.run --help`` for the entry point.
"""
