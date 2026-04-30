from __future__ import annotations
from src.data.validate_data import (
    CATEGORICAL_RULES,
    DataValidator,
    EXPECTED_COLUMNS,
    NUMERIC_COLS,
    RANGE_RULES,
    REQUIRED_COLUMNS,
)
import numpy as np
import pandas as pd
import pytest

# ===========================================================================
# Shared fixtures
# ===========================================================================


def _make_valid_df(n: int = 50) -> pd.DataFrame:
    """Return a small but fully valid merged DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "brand": ["volkswagen"] * n,
            "model": ["golf"] * n,
            "vehicleType": ["sedan"] * n,
            "power": np.random.uniform(60, 200, n),
            "gearbox": ["manual"] * n,
            "kilometer": np.random.uniform(20_000, 120_000, n),
            "fuelType": ["gasoline"] * n,
            "yearOfRegistration": np.random.randint(
                2000, 2018, n
            ).astype("int64"),
            "seller": ["private"] * n,
            "price": np.random.randint(3_000, 20_000, n).astype("int64"),
        }
    )


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    return _make_valid_df()


@pytest.fixture()
def validator() -> DataValidator:
    return DataValidator()

class TestInternalHelpers:
    def test_make_report_keys(self, validator: DataValidator) -> None:
        report = validator._make_report("Schema")
        assert report["check_type"] == "Schema"
        assert report["passed"] is True
        assert report["issues"] == []
        assert "timestamp" in report
        assert "stats" in report

    def test_fail_sets_passed_false(self, validator: DataValidator) -> None:
        report = validator._make_report("Test")
        validator._fail(report, "something went wrong")
        assert report["passed"] is False
        assert "something went wrong" in report["issues"]

    def test_store_appends_result(self, validator: DataValidator) -> None:
        report = validator._make_report("Test")
        validator._store(report)
        assert len(validator.validation_results) == 1

    def test_fail_accumulates_multiple_issues(
        self, validator: DataValidator
    ) -> None:
        report = validator._make_report("Multi")
        validator._fail(report, "issue 1")
        validator._fail(report, "issue 2")
        assert len(report["issues"]) == 2

class TestValidateSchema:
    def test_passes_for_valid_df(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_schema(valid_df)
        assert report["passed"] is True
        assert report["issues"] == []

    def test_fails_on_missing_column(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.drop(columns=["price"])
        report = validator.validate_schema(df)
        assert report["passed"] is False
        assert any("price" in issue for issue in report["issues"])

    def test_reports_extra_columns(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["extra_col"] = 0
        report = validator.validate_schema(df)
        # Extra columns are noted but do not fail the check
        assert any("Extra" in issue for issue in report["issues"])

    def test_fails_on_dtype_mismatch(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["price"] = df["price"].astype(str)
        report = validator.validate_schema(df)
        assert report["passed"] is False
        assert any("price" in issue for issue in report["issues"])

    def test_stats_contain_col_counts(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_schema(valid_df)
        assert "col_cnt" in report["stats"]
        assert "expected_col_cnt" in report["stats"]
        assert report["stats"]["expected_col_cnt"] == len(EXPECTED_COLUMNS)


# ===========================================================================
# validate_completeness
# ===========================================================================


class TestValidateCompleteness:
    def test_passes_for_complete_df(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_completeness(valid_df)
        assert report["passed"] is True

    def test_fails_when_missing_exceeds_threshold(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        # Null-out 20 % of 'price' (well above the 5 % threshold)
        null_idx = df.sample(frac=0.2, random_state=0).index
        df.loc[null_idx, "price"] = np.nan
        report = validator.validate_completeness(df)
        assert report["passed"] is False
        assert any("price" in issue for issue in report["issues"])

    def test_completeness_score_present(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_completeness(valid_df)
        assert "completeness_score_pct" in report["stats"]
        score = report["stats"]["completeness_score_pct"]
        assert 0 <= score <= 100

    def test_handles_missing_required_column(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.drop(columns=["price"])
        report = validator.validate_completeness(df)
        assert report["passed"] is False

    def test_column_detail_populated(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_completeness(valid_df)
        detail = report["stats"]["column_detail"]
        for col in REQUIRED_COLUMNS:
            if col in valid_df.columns:
                assert col in detail
                assert "missing_count" in detail[col]
                assert "missing_pct" in detail[col]


# ===========================================================================
# validate_uniqueness
# ===========================================================================


class TestValidateUniqueness:
    def test_passes_for_unique_df(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_uniqueness(valid_df)
        assert report["passed"] is True

    def test_fails_on_duplicate_rows(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = pd.concat([valid_df, valid_df.iloc[:5]], ignore_index=True)
        report = validator.validate_uniqueness(df)
        assert report["passed"] is False
        assert report["stats"]["duplicate_rows"] == 5

    def test_uniqueness_score_100_for_clean(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_uniqueness(valid_df)
        assert report["stats"]["uniqueness_score_pct"] == 100.0

    def test_uniqueness_score_below_100_for_dups(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = pd.concat([valid_df, valid_df.iloc[:10]], ignore_index=True)
        report = validator.validate_uniqueness(df)
        assert report["stats"]["uniqueness_score_pct"] < 100.0


# ===========================================================================
# validate_validity
# ===========================================================================


class TestValidateValidity:
    def test_passes_for_valid_df(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_validity(valid_df)
        assert report["passed"] is True

    def test_fails_on_price_below_min(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df.loc[0, "price"] = -1
        report = validator.validate_validity(df)
        assert report["passed"] is False
        assert any("price" in i for i in report["issues"])

    def test_fails_on_price_above_max(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df.loc[0, "price"] = 999_999_999
        report = validator.validate_validity(df)
        assert report["passed"] is False

    def test_fails_on_power_above_max(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df.loc[0, "power"] = 9_999.0
        report = validator.validate_validity(df)
        assert report["passed"] is False
        assert any("power" in i for i in report["issues"])

    def test_fails_on_invalid_categorical_seller(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df.loc[0, "seller"] = "unknown_seller"
        report = validator.validate_validity(df)
        assert report["passed"] is False
        assert any("seller" in i for i in report["issues"])

    def test_fails_on_invalid_fueltype(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["fuelType"] = "warp_plasma"
        report = validator.validate_validity(df)
        assert report["passed"] is False

    def test_range_stats_structure(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_validity(valid_df)
        range_checks = report["stats"]["range_checks"]
        for col in RANGE_RULES:
            if col in valid_df.columns:
                assert col in range_checks
                assert "accuracy_score_pct" in range_checks[col]

    def test_categorical_stats_structure(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_validity(valid_df)
        cat_checks = report["stats"]["categorical_checks"]
        for col in CATEGORICAL_RULES:
            if col in valid_df.columns:
                assert col in cat_checks


# ===========================================================================
# validate_outliers_zscore
# ===========================================================================


class TestValidateOutliersZScore:
    def test_passes_when_no_outliers(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_outliers_zscore(valid_df)
        # A clean dataset may still have natural Z-score outliers;
        # just assert the report is structured correctly.
        assert "column_detail" in report["stats"]

    def test_detects_obvious_outlier(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        # Inject an extreme outlier (|z| >> 3)
        df.loc[0, "price"] = 999_999_999
        report = validator.validate_outliers_zscore(df)
        assert report["stats"]["column_detail"]["price"][
            "outlier_count"
        ] >= 1

    def test_quality_score_structure(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_outliers_zscore(valid_df)
        for col in NUMERIC_COLS:
            if col in valid_df.columns:
                detail = report["stats"]["column_detail"].get(col, {})
                if detail:
                    assert "quality_score_pct" in detail
                    assert 0 <= detail["quality_score_pct"] <= 100


# ===========================================================================
# validate_outliers_iqr
# ===========================================================================


class TestValidateOutliersIQR:
    def test_passes_for_valid_df(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_outliers_iqr(valid_df)
        assert "column_detail" in report["stats"]

    def test_detects_iqr_outlier(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        # Force a value far outside the fences
        df.loc[0, "kilometer"] = 1_000_000.0
        report = validator.validate_outliers_iqr(df)
        detail = report["stats"]["column_detail"]["kilometer"]
        assert detail["outlier_count"] >= 1

    def test_iqr_stats_keys(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_outliers_iqr(valid_df)
        for col in NUMERIC_COLS:
            detail = report["stats"]["column_detail"].get(col)
            if detail is not None:
                for key in (
                    "Q1", "Q3", "IQR",
                    "lower_fence", "upper_fence",
                    "outlier_count", "outlier_pct",
                    "quality_score_pct",
                ):
                    assert key in detail, (
                        f"Key '{key}' missing for column '{col}'"
                    )


# ===========================================================================
# validate_outliers_isolation_forest
# ===========================================================================


class TestValidateOutliersIsolationForest:
    def test_runs_on_sufficient_data(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_outliers_isolation_forest(valid_df)
        assert "outlier_count" in report["stats"]
        assert "quality_score_pct" in report["stats"]

    def test_fails_on_insufficient_rows(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.iloc[:5].copy()
        report = validator.validate_outliers_isolation_forest(df)
        assert report["passed"] is False
        assert any("10" in i for i in report["issues"])

    def test_contamination_stored_in_stats(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_outliers_isolation_forest(
            valid_df, contamination=0.05
        )
        assert report["stats"]["contamination_param"] == 0.05


# ===========================================================================
# validate_distribution
# ===========================================================================


class TestValidateDistribution:
    def test_stats_keys_present(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_distribution(valid_df)
        assert "column_detail" in report["stats"]
        for col in NUMERIC_COLS:
            if col in valid_df.columns:
                detail = report["stats"]["column_detail"].get(col, {})
                if detail:
                    for key in (
                        "mean", "median", "std",
                        "skewness", "kurtosis_excess",
                        "ks_statistic", "ks_pvalue",
                    ):
                        assert key in detail, (
                            f"Key '{key}' missing for '{col}'"
                        )

    def test_fails_on_mean_out_of_bounds(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        # Push price mean far above upper bound (30_000)
        df["price"] = 5_000_000
        df["price"] = df["price"].astype("int64")
        report = validator.validate_distribution(df)
        assert report["passed"] is False
        assert any("price" in i for i in report["issues"])

    def test_skips_column_with_single_value(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.iloc[:1].copy()
        # Should not raise even with only one row
        report = validator.validate_distribution(df)
        assert isinstance(report, dict)


# ===========================================================================
# validate_relationships
# ===========================================================================


class TestValidateRelationships:
    def test_pearson_and_spearman_matrices_present(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_relationships(valid_df)
        assert "pearson_matrix" in report["stats"]
        assert "spearman_matrix" in report["stats"]

    def test_expected_correlation_checks_populated(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_relationships(valid_df)
        checks = report["stats"].get("expected_correlation_checks", {})
        assert len(checks) > 0

    def test_correlation_status_field_present(
        self, validator: DataValidator, valid_df: pd.DataFrame
    ) -> None:
        report = validator.validate_relationships(valid_df)
        checks = report["stats"]["expected_correlation_checks"]
        for key, val in checks.items():
            assert "status" in val, f"'status' missing for {key}"
            assert val["status"] in ("within bounds", "OUT OF BOUNDS")


# ===========================================================================
# run_all — integration smoke test
# ===========================================================================


class TestRunAll:
    def test_run_all_returns_summary_dict(
        self,
        validator: DataValidator,
        valid_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        output_prefix = str(tmp_path / "report")
        result = validator.run_all(valid_df, output_file=output_prefix)
        for key in ("total", "passed", "failed", "success_rate", "details"):
            assert key in result, f"Key '{key}' missing from run_all result"

    def test_run_all_writes_txt_and_json(
        self,
        validator: DataValidator,
        valid_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        output_prefix = str(tmp_path / "report")
        validator.run_all(valid_df, output_file=output_prefix)
        assert (tmp_path / "report.txt").exists()
        assert (tmp_path / "report.json").exists()

    def test_run_all_total_equals_nine_checks(
        self,
        validator: DataValidator,
        valid_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        output_prefix = str(tmp_path / "report")
        result = validator.run_all(valid_df, output_file=output_prefix)
        # Schema, Completeness, Uniqueness, Validity,
        # ZScore, IQR, IsolationForest, Distribution, Relationships = 9
        assert result["total"] == 9

    def test_success_rate_between_0_and_100(
        self,
        validator: DataValidator,
        valid_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        output_prefix = str(tmp_path / "report")
        result = validator.run_all(valid_df, output_file=output_prefix)
        assert 0.0 <= result["success_rate"] <= 100.0