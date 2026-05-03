"""
Model prediction module for making inference on new car data.

This module provides functionality to:
1. Load a trained model from disk
2. Preprocess input data using the same pipeline as training
3. Make predictions on new car listings
4. Return predictions with confidence scores
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.models.train_model import (
    FeatureEngineer,
    DropColumns,
)


class CarPriceTierPredictor:
    """
    Predictor class for car price tier classification.

    Handles the full prediction pipeline including feature engineering
    and model inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        brand_freq_map: Optional[Dict[str, float]] = None,
        model_freq_map: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the pickled model file
            brand_freq_map: Mapping of brand to frequency
            model_freq_map: Mapping of model to frequency
        """
        self.model_path = model_path or "models/best_model_for_production.pkl"
        self.model = self._load_model()

        # Default frequency maps from training data
        self.brand_freq_map = brand_freq_map or {
            "volkswagen": 0.200091,
            "bmw": 0.113960,
            "mercedes-benz": 0.104804,
            "opel": 0.096641,
            "audi": 0.096032,
            "ford": 0.066154,
            "renault": 0.043726,
            "peugeot": 0.030603,
            "fiat": 0.024497,
            "seat": 0.019900,
            "skoda": 0.017669,
            "mazda": 0.016253,
            "smart": 0.015558,
            "citroen": 0.015180,
            "toyota": 0.014748,
            "nissan": 0.013821,
            "hyundai": 0.011259,
            "mini": 0.010954,
            "volvo": 0.009939,
            "mitsubishi": 0.008680,
        }

        self.model_freq_map = model_freq_map or {
            "golf": 0.082492,
            "3-series": 0.060646,
            "c-class": 0.037851,
            "a4": 0.036454,
            "corsa": 0.031765,
            "polo": 0.028268,
            "astra": 0.028048,
            "passat": 0.026612,
            "5-series": 0.024408,
            "focus": 0.022637,
            "e-class": 0.022289,
            "a3": 0.019615,
            "2-reihe": 0.018607,
            "a6": 0.018021,
            "transporter": 0.016307,
            "fiesta": 0.014207,
            "twingo": 0.013845,
            "fortwo": 0.013779,
            "punto": 0.013019,
            "a-class": 0.012424,
        }

        self.labels_map = {0: "budget", 1: "mid-range", 2: "luxury"}

    def _load_model(self) -> Any:
        """Load the trained model from disk."""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data using the same pipeline as training.

        Args:
            df: Raw input dataframe with car features

        Returns:
            Preprocessed dataframe ready for prediction
        """
        df = df.copy()

        # Feature engineering
        fe = FeatureEngineer()
        dropper = DropColumns(["brand", "model"])

        # Apply frequency encoding manually to use our maps
        df["brand_freq"] = df["brand"].map(self.brand_freq_map).fillna(0.005)
        df["model_freq"] = df["model"].map(self.model_freq_map).fillna(0.005)

        # Apply other feature engineering
        df = fe.fit_transform(df)
        df = dropper.fit_transform(df)

        # One-hot encode categorical variables
        df = pd.get_dummies(df, drop_first=True)

        return df

    def predict(
        self, df: pd.DataFrame, return_proba: bool = False
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Make predictions on input data.

        Args:
            df: Input dataframe with car features
            return_proba: Whether to return probability scores

        Returns:
            List of predicted tiers or dict with predictions and probabilities
        """
        # Preprocess
        X = self.preprocess(df)

        # Align columns with model expectations
        if hasattr(self.model, "feature_names_in_"):
            expected_cols = self.model.feature_names_in_
            X = X.reindex(columns=expected_cols, fill_value=0)

        # Predict
        predictions = self.model.predict(X)
        tier_labels = [self.labels_map.get(int(p), "unknown") for p in predictions]

        if return_proba:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X)
                return {
                    "predictions": tier_labels,
                    "probabilities": probabilities.tolist(),
                    "confidence": probabilities.max(axis=1).tolist(),
                }
            else:
                return {
                    "predictions": tier_labels,
                    "probabilities": None,
                    "confidence": None,
                }

        return tier_labels

    def predict_single(
        self,
        brand: str,
        model: str,
        vehicle_type: str,
        power: float,
        gearbox: str,
        kilometer: float,
        fuel_type: str,
        year: int,
        seller: str,
        data_source: str = "kaggle",
    ) -> Dict[str, Any]:
        """
        Make prediction for a single car.

        Args:
            brand: Car brand
            model: Car model
            vehicle_type: Type of vehicle
            power: Engine power in PS
            gearbox: Transmission type
            kilometer: Mileage in km
            fuel_type: Fuel type
            year: Year of registration
            seller: Seller type (private/dealer)
            data_source: Data source identifier

        Returns:
            Dictionary with prediction and confidence
        """
        df = pd.DataFrame({
            "brand": [brand],
            "model": [model],
            "vehicleType": [vehicle_type],
            "power": [power],
            "gearbox": [gearbox],
            "kilometer": [kilometer],
            "fuelType": [fuel_type],
            "yearOfRegistration": [year],
            "seller": [seller],
            "dataSource": [data_source],
        })

        result = self.predict(df, return_proba=True)

        return {
            "tier": result["predictions"][0],
            "confidence": result["confidence"][0] if result["confidence"] else None,
            "probabilities": {
                tier: prob
                for tier, prob in zip(
                    ["budget", "mid-range", "luxury"],
                    result["probabilities"][0] if result["probabilities"] else [0, 0, 0],
                )
            },
        }


def predict_from_csv(
    input_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch prediction from CSV file.

    Args:
        input_path: Path to input CSV
        output_path: Optional path to save predictions
        model_path: Optional path to model file

    Returns:
        DataFrame with predictions added
    """
    predictor = CarPriceTierPredictor(model_path=model_path)

    df = pd.read_csv(input_path)
    predictions = predictor.predict(df, return_proba=True)

    df["predicted_tier"] = predictions["predictions"]
    df["confidence"] = predictions["confidence"]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    return df


if __name__ == "__main__":
    # Example usage
    predictor = CarPriceTierPredictor()

    # Single prediction
    result = predictor.predict_single(
        brand="bmw",
        model="3-series",
        vehicle_type="sedan",
        power=190,
        gearbox="manual",
        kilometer=125000,
        fuel_type="diesel",
        year=2015,
        seller="private",
    )
    print(f"Prediction: {result}")
