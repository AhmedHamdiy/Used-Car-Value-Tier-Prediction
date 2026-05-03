"""
Tests for the model selection module.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.models.select_model import (
    SelectionCriteria,
    ModelSelector,
    select_best_model_from_results,
)


class TestSelectionCriteria:
    """Test the SelectionCriteria dataclass."""

    def test_default_values(self):
        """Test default criteria values."""
        criteria = SelectionCriteria()
        assert criteria.primary_metric == "luxury_recall"
        assert criteria.secondary_metric == "val_f1"
        assert criteria.weight_primary == 0.7
        assert criteria.weight_secondary == 0.3
        assert criteria.min_threshold is None
        assert criteria.max_severe_misclassification is None

    def test_custom_values(self):
        """Test custom criteria values."""
        criteria = SelectionCriteria(
            primary_metric="test_bal_acc",
            secondary_metric=None,
            weight_primary=0.8,
            weight_secondary=0.2,
            min_threshold=0.75,
            max_severe_misclassification=0.05,
        )
        assert criteria.primary_metric == "test_bal_acc"
        assert criteria.secondary_metric is None
        assert criteria.weight_primary == 0.8
        assert criteria.weight_secondary == 0.2
        assert criteria.min_threshold == 0.75
        assert criteria.max_severe_misclassification == 0.05


class TestModelSelector:
    """Test the ModelSelector class."""

    @pytest.fixture
    def sample_results_df(self):
        """Create sample results DataFrame for testing."""
        return pd.DataFrame(
            {
                "model": [
                    "xgboost",
                    "xgboost",
                    "xgboost",
                    "random_forest",
                    "random_forest",
                    "baseline",
                ],
                "dataset": ["none", "smote", "undersample", "none", "smote", "none"],
                "test_f1": [0.90, 0.88, 0.85, 0.82, 0.80, 0.33],
                "val_f1": [0.89, 0.87, 0.84, 0.81, 0.79, 0.33],
                "test_bal_acc": [0.89, 0.87, 0.85, 0.81, 0.79, 0.33],
                "luxury_recall": [0.92, 0.90, 0.88, 0.85, 0.83, 0.00],
                "severe_misclassification_rate": [0.01, 0.02, 0.03, 0.05, 0.06, 0.20],
            }
        )

    @pytest.fixture
    def temp_results_file(self, tmp_path, sample_results_df):
        """Create temporary results file."""
        file_path = tmp_path / "model_comparison.csv"
        sample_results_df.to_csv(file_path, index=False)
        return str(file_path)

    def test_load_results(self, temp_results_file):
        """Test loading results from file."""
        selector = ModelSelector(temp_results_file)
        assert len(selector.results_df) == 6
        assert "model" in selector.results_df.columns
        assert "test_f1" in selector.results_df.columns

    def test_load_results_file_not_found(self):
        """Test error when results file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ModelSelector("nonexistent_file.csv")
        assert "Results file not found" in str(exc_info.value)

    def test_calculate_composite_score(self, temp_results_file):
        """Test composite score calculation."""
        selector = ModelSelector(temp_results_file)
        df = selector.calculate_composite_score(selector.results_df)

        assert "composite_score" in df.columns
        # Best model should have highest composite score
        best_idx = df["test_f1"].idxmax()
        assert df.loc[best_idx, "composite_score"] == pytest.approx(1.0, abs=0.01)

    def test_filter_models_by_min_threshold(self, temp_results_file):
        """Test filtering models by minimum threshold."""
        criteria = SelectionCriteria(min_threshold=0.80)
        selector = ModelSelector(temp_results_file, criteria)

        df = selector.filter_models(selector.results_df, criteria)
        assert all(df["test_f1"] >= 0.80)
        assert len(df) == 5  # baseline removed

    def test_filter_models_by_severe_misclass(self, temp_results_file):
        """Test filtering models by severe misclassification rate."""
        criteria = SelectionCriteria(max_severe_misclassification=0.04)
        selector = ModelSelector(temp_results_file, criteria)

        df = selector.filter_models(selector.results_df, criteria)
        assert all(df["severe_misclassification_rate"] <= 0.04)
        assert len(df) == 3

    def test_select_best_model(self, temp_results_file):
        """Test selecting the best model."""
        selector = ModelSelector(temp_results_file)
        best = selector.select_best_model()

        assert "error" not in best
        assert best["model_name"] == "xgboost"
        assert best["dataset_type"] == "none"
        assert best["rank"] == 1
        assert best["metrics"]["test_f1"] == pytest.approx(0.90, abs=0.01)

    def test_select_best_model_with_filter(self, temp_results_file):
        """Test selecting best model with model type filter."""
        selector = ModelSelector(temp_results_file)
        best = selector.select_best_model(model_type="random_forest")

        assert best["model_name"] == "random_forest"
        assert best["dataset_type"] == "none"

    def test_select_best_model_no_match(self, temp_results_file):
        """Test selecting best model when no models meet criteria."""
        criteria = SelectionCriteria(min_threshold=0.95)
        selector = ModelSelector(temp_results_file, criteria)
        best = selector.select_best_model()

        assert "error" in best
        assert "No models meet the selection criteria" in best["error"]

    def test_get_top_k_models(self, temp_results_file):
        """Test getting top k models."""
        selector = ModelSelector(temp_results_file)
        top3 = selector.get_top_k_models(3)

        assert len(top3) == 3
        # Should be sorted by composite score
        assert top3.iloc[0]["model"] == "xgboost"
        assert top3.iloc[0]["dataset"] == "none"

    def test_compare_models_by_type(self, temp_results_file):
        """Test comparing best config per model type."""
        selector = ModelSelector(temp_results_file)
        comparison = selector.compare_models_by_type()

        assert len(comparison) == 3  # xgboost, random_forest, baseline
        # Should have one row per unique model
        assert set(comparison["model"].unique()) == {
            "xgboost",
            "random_forest",
            "baseline",
        }

    def test_analyze_dataset_impact(self, temp_results_file):
        """Test analyzing dataset impact."""
        selector = ModelSelector(temp_results_file)
        impact = selector.analyze_dataset_impact()

        assert len(impact) == 3  # none, smote, undersample
        # Check that aggregation was performed
        assert ("mean", "test_f1") in impact.columns or "test_f1" in str(impact.columns)

    def test_generate_report(self, temp_results_file, tmp_path):
        """Test generating selection report."""
        selector = ModelSelector(temp_results_file)
        output_path = tmp_path / "selection_report.txt"

        report = selector.generate_report(str(output_path))

        assert "MODEL SELECTION REPORT" in report
        assert "BEST MODEL:" in report
        assert output_path.exists()
        assert "MODEL SELECTION REPORT" in output_path.read_text()

    def test_export_best_model_info(self, temp_results_file, tmp_path):
        """Test exporting best model info to JSON."""
        selector = ModelSelector(temp_results_file)
        output_path = tmp_path / "best_model.json"

        selector.export_best_model_info(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "best_model" in data
        assert "selection_criteria" in data
        assert "top_5_models" in data


class TestSelectBestModelFromResults:
    """Test the convenience function."""

    @pytest.fixture
    def temp_results_file(self, tmp_path):
        """Create temporary results file."""
        df = pd.DataFrame(
            {
                "model": ["xgboost", "random_forest"],
                "dataset": ["none", "none"],
                "test_f1": [0.90, 0.82],
                "val_f1": [0.89, 0.81],
                "test_bal_acc": [0.89, 0.81],
                "luxury_recall": [0.92, 0.85],
                "severe_misclassification_rate": [0.01, 0.05],
            }
        )
        file_path = tmp_path / "model_comparison.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)

    def test_select_best_model_from_results(self, temp_results_file):
        """Test convenience function."""
        best = select_best_model_from_results(
            results_path=temp_results_file,
            primary_metric="test_f1",
            secondary_metric="luxury_recall",
        )

        assert best["model_name"] == "xgboost"
        assert best["dataset_type"] == "none"

    def test_select_with_min_threshold(self, temp_results_file):
        """Test with minimum threshold."""
        best = select_best_model_from_results(
            results_path=temp_results_file,
            min_test_f1=0.85,
        )

        assert best["model_name"] == "xgboost"
        # random_forest should be filtered out due to threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
