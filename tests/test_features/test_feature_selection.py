"""
Tests for the feature selection module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.feature_selection import (
    FeatureSelector,
    select_features,
)


class TestFeatureSelector:
    """Test the FeatureSelector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(n),
                "feature_2": np.random.randn(n),
                "feature_3": np.random.randn(n),
                "feature_4": np.random.randn(n),
                "feature_5": np.random.randn(n),
                "constant": [1.0] * n,  # Low variance
            }
        )
        # Target correlated with first 3 features
        y = (X["feature_1"] + X["feature_2"] + X["feature_3"] > 0).astype(int)
        return X, y

    def test_init(self):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector(strategy="variance", threshold=0.1, k=10)
        assert selector.strategy == "variance"
        assert selector.threshold == 0.1
        assert selector.k == 10
        assert selector.selected_features == []

    def test_variance_strategy(self, sample_data):
        """Test variance threshold strategy."""
        X, y = sample_data
        selector = FeatureSelector(strategy="variance", threshold=0.01)

        selector.fit(X, y)

        # Should remove constant feature
        assert "constant" not in selector.selected_features
        assert len(selector.selected_features) < len(X.columns)
        assert selector.feature_scores_ is not None

    def test_correlation_strategy(self, sample_data):
        """Test correlation strategy."""
        X, y = sample_data
        selector = FeatureSelector(strategy="correlation", threshold=0.01)

        selector.fit(X, y)

        # Should select at least some features
        assert len(selector.selected_features) > 0
        assert selector.feature_scores_ is not None
        # Top features should be from first 3 (correlated with target)
        top_features = selector.feature_scores_.head(3).index.tolist()
        assert any(f in ["feature_1", "feature_2", "feature_3"] for f in top_features)

    def test_k_best_strategy(self, sample_data):
        """Test k-best strategy."""
        X, y = sample_data
        selector = FeatureSelector(strategy="k_best", k=3)

        selector.fit(X, y)

        # Should select exactly k features
        assert len(selector.selected_features) == 3
        assert selector.feature_scores_ is not None

    def test_mutual_info_strategy(self, sample_data):
        """Test mutual information strategy."""
        X, y = sample_data
        selector = FeatureSelector(strategy="mutual_info", k=3)

        selector.fit(X, y)

        # Should select exactly k features
        assert len(selector.selected_features) == 3
        assert selector.feature_scores_ is not None

    def test_transform(self, sample_data):
        """Test transform method."""
        X, y = sample_data
        selector = FeatureSelector(strategy="variance", threshold=0.01)

        selector.fit(X, y)
        X_selected = selector.transform(X)

        # Should only have selected columns
        assert list(X_selected.columns) == selector.selected_features
        assert X_selected.shape[0] == X.shape[0]
        assert X_selected.shape[1] <= X.shape[1]

    def test_get_feature_importance_report(self, sample_data):
        """Test report generation."""
        X, y = sample_data
        selector = FeatureSelector(strategy="variance", threshold=0.01)

        selector.fit(X, y)
        report = selector.get_feature_importance_report(top_n=5)

        assert "FEATURE IMPORTANCE REPORT" in report
        assert "Strategy: variance" in report
        assert "SELECTED FEATURES:" in report

    def test_export_selected_features(self, sample_data, tmp_path):
        """Test exporting selected features."""
        X, y = sample_data
        selector = FeatureSelector(strategy="variance", threshold=0.01)

        selector.fit(X, y)
        output_path = tmp_path / "selected_features.txt"
        selector.export_selected_features(str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "Selected features" in content
        for feature in selector.selected_features:
            assert feature in content


class TestSelectFeatures:
    """Test the select_features convenience function."""

    @pytest.fixture
    def sample_splits(self):
        """Create sample train/val/test splits."""
        np.random.seed(42)
        n_train, n_val, n_test = 100, 30, 30

        X_train = pd.DataFrame(
            {
                "f1": np.random.randn(n_train),
                "f2": np.random.randn(n_train),
                "f3": np.random.randn(n_train),
                "constant": [1.0] * n_train,
            }
        )
        y_train = pd.Series(np.random.choice([0, 1, 2], n_train))

        X_val = pd.DataFrame(
            {
                "f1": np.random.randn(n_val),
                "f2": np.random.randn(n_val),
                "f3": np.random.randn(n_val),
                "constant": [1.0] * n_val,
            }
        )

        X_test = pd.DataFrame(
            {
                "f1": np.random.randn(n_test),
                "f2": np.random.randn(n_test),
                "f3": np.random.randn(n_test),
                "constant": [1.0] * n_test,
            }
        )

        return X_train, y_train, X_val, X_test

    def test_select_features_variance(self, sample_splits):
        """Test select_features with variance strategy."""
        X_train, y_train, X_val, X_test = sample_splits

        X_tr, X_v, X_te, selector = select_features(
            X_train,
            y_train,
            X_val,
            X_test,
            strategy="variance",
            threshold=0.01,
        )

        assert isinstance(selector, FeatureSelector)
        assert X_tr.shape[1] < X_train.shape[1]
        assert "constant" not in X_tr.columns
        assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1]

    def test_select_features_k_best(self, sample_splits):
        """Test select_features with k_best strategy."""
        X_train, y_train, X_val, X_test = sample_splits

        X_tr, X_v, X_te, selector = select_features(
            X_train,
            y_train,
            X_val,
            X_test,
            strategy="k_best",
            k=2,
        )

        assert X_tr.shape[1] == 2
        assert X_v.shape[1] == 2
        assert X_te.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
