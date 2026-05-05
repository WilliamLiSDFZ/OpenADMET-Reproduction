"""Base models with a unified single-task fit/predict interface.

Every model class implements:
    .fit(X, y, sample_weight=None)
    .predict(X) -> np.ndarray
    .name (str)

That keeps the ensemble layer dead simple.
"""
from .lgbm_model import LightGBMModel
from .xgb_model import XGBoostModel
from .rf_model import RandomForestModel
from .catboost_model import CatBoostModel

# Heavy GPU/PyTorch imports are deferred to the entry point so that
# CPU-only debugging still works.
__all__ = ["LightGBMModel", "XGBoostModel", "RandomForestModel", "CatBoostModel"]
