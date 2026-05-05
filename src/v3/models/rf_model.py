"""Single-task scikit-learn Random Forest wrapper."""
from __future__ import annotations

import numpy as np

from ..config import RF_PARAMS


class RandomForestModel:
    name = "rf"

    def __init__(self, params: dict | None = None):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**(params or RF_PARAMS))

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float64)
