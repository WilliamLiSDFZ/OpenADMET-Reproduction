"""Single-task LightGBM regressor wrapper."""
from __future__ import annotations

import numpy as np

from ..config import LGBM_PARAMS


class LightGBMModel:
    name = "lgbm"

    def __init__(self, params: dict | None = None):
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(**(params or LGBM_PARAMS))

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float64)
