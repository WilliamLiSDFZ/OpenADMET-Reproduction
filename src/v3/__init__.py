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
import warnings as _warnings

# Mute the harmless sklearn warning that fires every time we predict on a
# numpy array using a model that was fitted from one (LightGBM auto-names
# columns "Column_0..N" at fit-time, then sklearn complains they're missing
# at predict-time even though predictions are correct).
_warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names",
    category=UserWarning,
)
# np.float32 cast overflow on a few RDKit descriptors -- already nan_to_num'd
# downstream, no information lost.
_warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=r"overflow encountered in cast"
)
