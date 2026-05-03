"""
Feature selection module for used car price tier prediction.

This module provides multiple feature selection strategies:
1. XGBoost feature importance
2. Variance threshold
3. Correlation-based selection
4. Statistical tests (chi-square, ANOVA)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGBoost feature importance will not work.")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector supporting multiple strategies.

    Attributes:
        strategy: Selection strategy ('xgboost', 'variance', 'correlation', 'k_best')
        threshold: Threshold for selection (interpretation varies by strategy)
        selected_features: List of selected feature names
    """

    def __init__(
        self,
        strategy: str = "xgboost",
        threshold: Optional[float] = None,
        k: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize feature selector.

        Args:
            strategy: Selection strategy ('xgboost', 'variance', 'correlation', 'k_best')
            threshold: Threshold for selection (variance threshold or importance percentile)
            k: Number of top features to select (for k_best strategy)
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.threshold = threshold
        self.k = k
        self.random_state = random_state
        self.selected_features: List[str] = []
        self.feature_scores_: Optional[pd.Series] = None
        self.selector_: Optional[Any] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """
        Fit the feature selector.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            self
        """
        if self.strategy == "xgboost":
            self._fit_xgboost(X, y)
        elif self.strategy == "variance":
            self._fit_variance(X, y)
        elif self.strategy == "correlation":
            self._fit_correlation(X, y)
        elif self.strategy == "k_best":
            self._fit_k_best(X, y)
        elif self.strategy == "mutual_info":
            self._fit_mutual_info(X, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with selected features only
        """
        if not self.selected_features:
            return X

        # Ensure all selected features exist in X
        available_features = [f for f in self.selected_features if f in X.columns]
        missing_features = set(self.selected_features) - set(available_features)

        if missing_features:
            warnings.warn(f"Features not found in transform data: {missing_features}")

        return X[available_features]

    def _fit_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """Fit using XGBoost feature importance."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for this strategy")

        # Train XGBoost model to get feature importance
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            random_state=self.random_state,
            n_estimators=100,
            max_depth=6,
        )

        model.fit(X, y)

        # Get feature importance
        importance = model.feature_importances_
        self.feature_scores_ = pd.Series(importance, index=X.columns).sort_values(
            ascending=False
        )

        # Select features based on threshold
        threshold = self.threshold or 0.01  # Default: keep features with >1% importance
        self.selected_features = self.feature_scores_[
            self.feature_scores_ >= threshold
        ].index.tolist()

        if not self.selected_features:
            # If no features meet threshold, take top k or at least top 10
            k = self.k or min(10, len(X.columns))
            self.selected_features = self.feature_scores_.head(k).index.tolist()

    def _fit_variance(self, X: pd.DataFrame, y: pd.Series):
        """Fit using variance threshold."""
        threshold = self.threshold or 0.01

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        self.selector_ = selector
        self.selected_features = X.columns[selector.get_support()].tolist()

        # Calculate variance scores
        variances = X.var()
        self.feature_scores_ = variances.sort_values(ascending=False)

    def _fit_correlation(self, X: pd.DataFrame, y: pd.Series):
        """Fit using correlation with target."""
        # Encode target if needed
        if y.dtype == "object":
            y_encoded = pd.factorize(y)[0]
        else:
            y_encoded = y

        # Calculate correlation with target
        correlations = []
        for col in X.columns:
            if X[col].dtype in ["int64", "float64", "int32", "float32"]:
                corr = np.abs(np.corrcoef(X[col].fillna(0), y_encoded)[0, 1])
                correlations.append(corr)
            else:
                correlations.append(0)

        self.feature_scores_ = pd.Series(correlations, index=X.columns).sort_values(
            ascending=False
        )

        # Select features based on threshold
        threshold = self.threshold or 0.05
        self.selected_features = self.feature_scores_[
            self.feature_scores_ >= threshold
        ].index.tolist()

        if not self.selected_features:
            # If no features meet threshold, take top k or at least top 10
            k = self.k or min(10, len(X.columns))
            self.selected_features = self.feature_scores_.head(k).index.tolist()

    def _fit_k_best(self, X: pd.DataFrame, y: pd.Series):
        """Fit using SelectKBest with f_classif."""
        k = self.k or min(10, len(X.columns))

        # Handle non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(0)

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_numeric, y)

        self.selector_ = selector
        self.selected_features = X_numeric.columns[selector.get_support()].tolist()

        # Store scores
        scores = selector.scores_
        self.feature_scores_ = pd.Series(scores, index=X_numeric.columns).sort_values(
            ascending=False
        )

    def _fit_mutual_info(self, X: pd.DataFrame, y: pd.Series):
        """Fit using mutual information."""
        k = self.k or min(10, len(X.columns))

        # Handle non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(0)

        # Calculate mutual information
        mi_scores = mutual_info_classif(X_numeric, y, random_state=self.random_state)

        self.feature_scores_ = pd.Series(
            mi_scores, index=X_numeric.columns
        ).sort_values(ascending=False)

        # Select top k features
        self.selected_features = self.feature_scores_.head(k).index.tolist()
        self.selector_ = None  # No sklearn selector for this

    def get_feature_importance_report(self, top_n: int = 20) -> str:
        """
        Generate a feature importance report.

        Args:
            top_n: Number of top features to show

        Returns:
            Formatted report string
        """
        if self.feature_scores_ is None:
            return "Feature selector has not been fitted yet."

        lines = [
            "=" * 80,
            "FEATURE IMPORTANCE REPORT",
            "=" * 80,
            f"Strategy: {self.strategy}",
            f"Total features: {len(self.feature_scores_)}",
            f"Selected features: {len(self.selected_features)}",
            f"Selection rate: {len(self.selected_features) / len(self.feature_scores_) * 100:.1f}%",
            "",
            f"TOP {top_n} FEATURES:",
            "-" * 80,
        ]

        top_features = self.feature_scores_.head(top_n)
        for idx, (feature, score) in enumerate(top_features.items(), 1):
            selected = "✓" if feature in self.selected_features else " "
            lines.append(f"{idx:2d}. [{selected}] {feature:40s} {score:.6f}")

        lines.extend(
            [
                "",
                "SELECTED FEATURES:",
                "-" * 80,
            ]
        )

        for i, feature in enumerate(self.selected_features, 1):
            score = self.feature_scores_[feature]
            lines.append(f"{i:2d}. {feature:40s} {score:.6f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_selected_features(self, output_path: str):
        """Export selected feature names to a file."""
        with open(output_path, "w") as f:
            f.write("# Selected features\n")
            f.write(f"# Strategy: {self.strategy}\n")
            f.write(f"# Total: {len(self.selected_features)}\n")
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        print(f"Selected features exported to: {output_path}")


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "xgboost",
    threshold: Optional[float] = None,
    k: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureSelector]:
    """
    Convenience function to select features across train/val/test sets.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        X_test: Test features
        strategy: Selection strategy
        threshold: Selection threshold
        k: Number of features to select
        random_state: Random seed

    Returns:
        Tuple of (X_train_selected, X_val_selected, X_test_selected, selector)
    """
    selector = FeatureSelector(
        strategy=strategy,
        threshold=threshold,
        k=k,
        random_state=random_state,
    )

    selector.fit(X_train, y_train)

    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print(f"\nFeature Selection Summary:")
    print(f"  Strategy: {strategy}")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {X_train_selected.shape[1]}")
    print(
        f"  Reduction: {(1 - X_train_selected.shape[1] / X_train.shape[1]) * 100:.1f}%"
    )

    return X_train_selected, X_val_selected, X_test_selected, selector


if __name__ == "__main__":
    # Example usage
    print("Feature Selection Module")
    print("Available strategies: xgboost, variance, correlation, k_best, mutual_info")
