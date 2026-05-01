from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.freq_maps_ = {}
        for col in list(self.cols):
            if col in X_df.columns:
                self.freq_maps_[col] = X_df[col].value_counts(normalize=True)
            else:
                self.freq_maps_[col] = pd.Series(dtype=float)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in list(self.cols):
            freq = self.freq_maps_.get(col)
            X_df[f"{col}_freq"] = (
                X_df[col].map(freq).fillna(0.0) if col in X_df.columns else 0.0
            )
        return X_df
