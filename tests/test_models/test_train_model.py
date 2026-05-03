import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.train_model import (
    FeatureEngineer,
    DropColumns,
    FrequencyEncoder,
    calculate_business_metrics,
    get_model_configs,
    prepare_data,
    create_resampled_datasets,
    train_model,
)


class TestFeatureEngineer:
    """Test the FeatureEngineer transformer."""

    def test_basic_transformation(self):
        """Test that features are correctly engineered."""
        df = pd.DataFrame(
            {
                "yearOfRegistration": [2015, 2018, 2020],
                "kilometer": [50000, 30000, 10000],
                "power": [100, 150, 200],
            }
        )

        fe = FeatureEngineer(current_year=2024)
        result = fe.fit_transform(df)

        assert "vehicleAge" in result.columns
        assert "kmPerYear" in result.columns
        assert "km_power_ratio" in result.columns
        assert "log_km" in result.columns
        assert "yearOfRegistration" not in result.columns

        assert result["vehicleAge"].iloc[0] == 9

    def test_km_per_year_calculation(self):
        """Test kmPerYear is calculated correctly."""
        df = pd.DataFrame(
            {
                "yearOfRegistration": [2020],
                "kilometer": [50000.0],
                "power": [100.0],
            }
        )

        fe = FeatureEngineer(current_year=2024)
        result = fe.fit_transform(df)

        expected_km_per_year = 50000 / (4 + 1)
        assert abs(result["kmPerYear"].iloc[0] - expected_km_per_year) < 0.01


class TestDropColumns:
    """Test the DropColumns transformer."""

    def test_drop_single_column(self):
        """Test dropping a single column."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        dropper = DropColumns(["b"])
        result = dropper.fit_transform(df)

        assert "b" not in result.columns
        assert "a" in result.columns
        assert "c" in result.columns

    def test_drop_multiple_columns(self):
        """Test dropping multiple columns."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        dropper = DropColumns(["a", "c"])
        result = dropper.fit_transform(df)

        assert list(result.columns) == ["b"]


class TestFrequencyEncoder:
    """Test the FrequencyEncoder transformer."""

    def test_frequency_encoding(self):
        """Test that frequencies are correctly encoded."""
        df = pd.DataFrame(
            {
                "brand": ["bmw", "bmw", "audi", "audi", "audi"],
                "model": ["x5", "x3", "a4", "a6", "a4"],
            }
        )

        encoder = FrequencyEncoder(cols=["brand", "model"])
        result = encoder.fit_transform(df)

        assert "brand_freq" in result.columns
        assert "model_freq" in result.columns

        # BMW appears 2/5 times = 0.4
        assert abs(result["brand_freq"].iloc[0] - 0.4) < 0.01
        # Audi appears 3/5 times = 0.6
        assert abs(result["brand_freq"].iloc[2] - 0.6) < 0.01

    def test_missing_column_handling(self):
        """Test handling of missing columns."""
        df = pd.DataFrame(
            {
                "brand": ["bmw", "audi"],
            }
        )

        encoder = FrequencyEncoder(cols=["brand", "model"])
        result = encoder.fit_transform(df)

        assert "brand_freq" in result.columns
        assert "model_freq" in result.columns
        assert all(result["model_freq"] == 0.0)


class TestCalculateBusinessMetrics:
    """Test the calculate_business_metrics function."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = calculate_business_metrics(y_true, y_pred)

        assert metrics["luxury_recall"] == 1.0
        assert metrics["severe_misclassification_rate"] == 0.0

    def test_severe_misclassifications(self):
        """Test severe misclassifications (budget <-> luxury)."""
        y_true = np.array([0, 0, 2, 2])  # budget, budget, luxury, luxury
        y_pred = np.array([2, 0, 0, 2])  # luxury, budget, budget, luxury

        metrics = calculate_business_metrics(y_true, y_pred)

        # 2 severe errors out of 4 = 0.5
        assert metrics["severe_misclassification_rate"] == 0.5

    def test_no_severe_misclassifications(self):
        """Test with only mid-range errors."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 0, 1])  # Only mid-range involved

        metrics = calculate_business_metrics(y_true, y_pred)

        assert metrics["severe_misclassification_rate"] == 0.0


class TestGetModelConfigs:
    """Test the get_model_configs function."""

    def test_returns_dict(self):
        """Test that a dictionary is returned."""
        configs = get_model_configs()
        assert isinstance(configs, dict)

    def test_has_expected_models(self):
        """Test that expected models are present."""
        configs = get_model_configs()
        expected_models = ["baseline", "random_forest", "log_reg", "decision_tree"]

        for model in expected_models:
            assert model in configs, f"Model {model} not found in configs"

    def test_config_structure(self):
        """Test that each config has required keys."""
        configs = get_model_configs()

        for model_name, config in configs.items():
            assert "estimator" in config
            assert "params" in config
            assert "n_iter" in config


class TestPrepareData:
    """Test the prepare_data function."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame(
            {
                "brand": ["bmw"] * n,
                "model": ["x5"] * n,
                "vehicleType": ["suv"] * n,
                "power": np.random.uniform(100, 300, n),
                "gearbox": ["automatic"] * n,
                "kilometer": np.random.uniform(10000, 100000, n),
                "fuelType": ["gasoline"] * n,
                "yearOfRegistration": np.random.randint(2010, 2023, n),
                "seller": ["private"] * n,
                "dataSource": ["test"] * n,
                "price": np.random.uniform(5000, 50000, n),
                "price_tier": ["budget", "mid-range", "luxury"] * 33
                + ["budget"],  # 100 rows
            }
        )

    def test_returns_correct_splits(self, sample_df):
        """Test that correct number of splits is returned."""
        result = prepare_data(sample_df, current_year=2024)

        assert len(result) == 6
        X_train, X_val, X_test, y_train, y_val, y_test = result

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)

    def test_splits_are_properly_sized(self, sample_df):
        """Test that splits have expected sizes."""
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
            sample_df, test_size=0.2, val_size=0.25, current_year=2024
        )

        total = len(sample_df)

        # Test set should be ~20%
        assert abs(len(X_test) - int(total * 0.2)) <= 1

        # Val set should be ~25% of remaining 80% = ~20% of total
        assert abs(len(X_val) - int(total * 0.8 * 0.25)) <= 1

    def test_price_tier_encoded(self, sample_df):
        """Test that price_tier is properly encoded."""
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
            sample_df, current_year=2024
        )

        # Check that target values are numeric
        assert y_train.dtype in [np.int64, np.int32, int]

        # Check values are 0, 1, or 2
        assert set(y_train.unique()).issubset({0, 1, 2})


class TestCreateResampledDatasets:
    """Test the create_resampled_datasets function."""

    def test_returns_dict(self):
        """Test that a dictionary is returned."""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1, 2], 100))

        datasets = create_resampled_datasets(X, y)

        assert isinstance(datasets, dict)
        assert "none" in datasets

    def test_none_dataset_is_original(self):
        """Test that 'none' dataset is the original data."""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1, 2], 100))

        datasets = create_resampled_datasets(X, y)
        X_none, y_none = datasets["none"]

        assert len(X_none) == len(X)
        assert len(y_none) == len(y)


class TestTrainModel:
    """Test the train_model function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample train/val/test data."""
        np.random.seed(42)
        n_train = 100
        n_val = 30
        n_test = 30

        X_train = pd.DataFrame(
            np.random.randn(n_train, 5), columns=[f"f{i}" for i in range(5)]
        )
        y_train = pd.Series(np.random.choice([0, 1, 2], n_train))

        X_val = pd.DataFrame(
            np.random.randn(n_val, 5), columns=[f"f{i}" for i in range(5)]
        )
        y_val = pd.Series(np.random.choice([0, 1, 2], n_val))

        X_test = pd.DataFrame(
            np.random.randn(n_test, 5), columns=[f"f{i}" for i in range(5)]
        )
        y_test = pd.Series(np.random.choice([0, 1, 2], n_test))

        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_train_baseline_returns_results(self, sample_data):
        """Test training baseline model returns expected results."""
        X_train, y_train, X_val, y_val, X_test, y_test = sample_data

        result = train_model(
            model_name="baseline",
            dataset_type="none",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            mlflow_tracking_uri=None,  # Disable MLflow for testing
        )

        assert isinstance(result, dict)
        assert "model" in result
        assert "dataset" in result
        assert "val_f1" in result
        assert "test_f1" in result
        assert "test_bal_acc" in result
        assert "luxury_recall" in result
        assert "severe_misclassification_rate" in result

        assert result["model"] == "baseline"
        assert result["dataset"] == "none"

    def test_train_decision_tree(self, sample_data):
        """Test training decision tree model."""
        X_train, y_train, X_val, y_val, X_test, y_test = sample_data

        result = train_model(
            model_name="decision_tree",
            dataset_type="none",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            mlflow_tracking_uri=None,
        )

        assert result["model"] == "decision_tree"
        assert result["dataset"] == "none"
        assert result["test_f1"] >= 0  # F1 should be non-negative
        assert result["test_f1"] <= 1  # F1 should be at most 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
