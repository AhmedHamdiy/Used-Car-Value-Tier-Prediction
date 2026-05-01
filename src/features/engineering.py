from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class FeatureEngineeringConfig:
    current_year: int | None = None
    vintage_age_years: int = 25
    high_mileage_quantile: float = 0.75


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        current_year: int | None = None,
        vintage_age_years: int = 25,
        high_mileage_quantile: float = 0.75,
    ) -> None:
        self.current_year = current_year
        self.vintage_age_years = vintage_age_years
        self.high_mileage_quantile = high_mileage_quantile

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        current_year = self._current_year()

        if "yearOfRegistration" in X_df.columns:
            vehicle_age = current_year - pd.to_numeric(
                X_df["yearOfRegistration"], errors="coerce"
            )
            vehicle_age = vehicle_age.clip(lower=0)
            km_per_year = pd.to_numeric(X_df.get("kilometer"), errors="coerce") / (
                vehicle_age + 1
            )
            self.high_mileage_threshold_ = float(
                km_per_year.quantile(self.high_mileage_quantile)
            )
        else:
            self.high_mileage_threshold_ = 0.0

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        current_year = self._current_year()

        year = pd.to_numeric(X_df.get("yearOfRegistration"), errors="coerce")
        power = pd.to_numeric(X_df.get("power"), errors="coerce")
        km = pd.to_numeric(X_df.get("kilometer"), errors="coerce")

        vehicle_age = (current_year - year).clip(lower=0)
        X_df["vehicleAge"] = vehicle_age

        X_df["kmPerYear"] = km / (vehicle_age + 1)
        X_df["km_power_ratio"] = km / (power + 1)

        X_df["log_power"] = np.log1p(power.clip(lower=0))
        X_df["power_per_age"] = power / (vehicle_age + 1)

        X_df["log_kilometer"] = np.log1p(km.clip(lower=0))

        X_df["is_vintage"] = (vehicle_age > self.vintage_age_years).astype(int)
        thr = getattr(self, "high_mileage_threshold_", 0.0)
        X_df["high_mileage_flag"] = (X_df["kmPerYear"] > thr).astype(int)

        X_df = X_df.drop(columns=["yearOfRegistration"], errors="ignore")
        return X_df

    def _current_year(self) -> int:
        if self.current_year is not None:
            return int(self.current_year)
        return datetime.now().year
