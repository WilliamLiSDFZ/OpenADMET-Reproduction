"""Single-task XGBoost regressor wrapper."""
from __future__ import annotations

import numpy as np

from ..config import XGB_PARAMS


class XGBoostModel:
    name = "xgb"

    def __init__(self, params: dict | None = None):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(**(params or XGB_PARAMS))

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float64)
