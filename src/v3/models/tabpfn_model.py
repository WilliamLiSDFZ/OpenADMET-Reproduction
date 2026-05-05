"""TabPFN v2 wrapper.

TabPFN is a transformer pre-trained on millions of synthetic tabular
datasets to do Bayesian inference via in-context learning. The JCIM 2025
paper (Fischer/Cedeño) found it surprisingly competitive for ADME tasks.

Hard limits (TabPFN v2.0): ≤ 10000 train rows, ≤ 500 features.
We feed it ONLY the 210-dim RDKit-2d block, which keeps every challenge
endpoint comfortably below the limit.

GPU recommended (T4 fits TabPFN v2 with batch_size <= 1024). CPU works
but is ~10× slower.
"""
from __future__ import annotations

import numpy as np

from ..config import TABPFN_PARAMS


class TabPFNModel:
    name = "tabpfn"

    def __init__(self, device: str | None = None,
                 n_estimators: int | None = None):
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as e:
            raise SystemExit(
                "TabPFN not installed. On your T4 host:\n"
                "    pip install tabpfn\n"
                f"Original import error: {e}"
            )
        self._cls = TabPFNRegressor
        self.device = device or TABPFN_PARAMS["device"]
        self.n_estimators = n_estimators or TABPFN_PARAMS["n_estimators"]
        self.model = None

    def fit(self, X, y, sample_weight=None):
        """Note: TabPFN ignores sample_weight; we accept it for API uniformity."""
        # Limit feature count to 500 (TabPFN's hard cap)
        if X.shape[1] > 500:
            raise ValueError(
                f"TabPFN feature limit is 500; got {X.shape[1]}. Pass only "
                "the RDKit-2d slice (e.g. via features.rdkit_only_slice())"
            )
        self.model = self._cls(
            device=self.device, n_estimators=self.n_estimators,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        return np.asarray(self.model.predict(X), dtype=np.float64)
