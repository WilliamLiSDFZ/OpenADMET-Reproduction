"""Single-task CatBoost regressor wrapper."""
from __future__ import annotations

import numpy as np

from ..config import CATBOOST_PARAMS


class CatBoostModel:
    name = "catboost"

    def __init__(self, params: dict | None = None):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(**(params or CATBOOST_PARAMS))

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float64)
