from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── modules under test ──────────────────────────────────────────
from merge_data import (
    CPI_USED_CARS,
    REFERENCE_YEAR,
    TARGET_COLS,
    _transform_crawled,
    _transform_kaggle,
    normalize_price,
    price_tier,
    transform_data,
)
from validate_data import (
    CATEGORICAL_RULES,
    EXPECTED_COLUMNS,
    EXPECTED_CORRELATIONS,
    EXPECTED_DTYPES,
    MEAN_BOUNDS,
    NUMERIC_COLS,
    RANGE_RULES,
    REQUIRED_COLUMNS,
    DataValidator,
)


# ═══════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════

def _kaggle_df(n: int = 5, year: int = 2021) -> pd.DataFrame:
    """Minimal kaggle-style raw DataFrame."""
    date_str = f"{year}-03-15 10:00:00"
    return pd.DataFrame({
        "dateCrawled": [date_str] * n,
        "price": [10_000.0] * n,
        "powerPS": [120.0] * n,
        "brand": ["volkswagen"] * n,
        "model": ["golf"] * n,
        "vehicleType": ["sedan"] * n,
        "gearbox": ["manual"] * n,
        "kilometer": [50_000.0] * n,
        "fuelType": ["gasoline"] * n,
        "yearOfRegistration": [2018] * n,
        "seller": ["private"] * n,
        "dataSource": ["kaggle"] * n,
        "price_tier": ["mid-range"] * n,
    })


def _crawled_df(n: int = 5) -> pd.DataFrame:
    """Minimal crawled-style raw DataFrame."""
    return pd.DataFrame({
        "mileage": [50_000.0] * n,
        "price": [10_000.0] * n,
        "power": ["120 kW (162 hp)"] * n,
        "year": ["2018"] * n,
        "brand": ["bmw"] * n,
        "model": ["3-series"] * n,
        "vehicleType": ["sedan"] * n,
        "gearbox": ["manual"] * n,
        "fuelType": ["gasoline"] * n,
        "yearOfRegistration": [2018] * n,
        "seller": ["private"] * n,
        "dataSource": ["crawled"] * n,
        "price_tier": ["mid-range"] * n,
    })


def _clean_df(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    Return a DataFrame that satisfies the validator's numeric dtype
    expectations as closely as possible.  Note: 'price' is float64
    in practice (a known EXPECTED_DTYPES bug is documented below).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "brand": ["volkswagen"] * n,
        "model": ["golf"] * n,
        "vehicleType": ["sedan"] * n,
        "power": rng.uniform(60, 200, n).astype("float64"),
        "gearbox": ["manual"] * n,
        "kilometer": rng.uniform(10_000, 100_000, n).astype("float64"),
        "fuelType": ["gasoline"] * n,
        "yearOfRegistration": rng.integers(
            2000, 2020, n
        ).astype("int64"),
        "seller": ["private"] * n,
        "dataSource": ["kaggle"] * n,
        "price": rng.uniform(5_000, 25_000, n).astype("float64"),
        "price_tier": ["mid-range"] * n,
    })


# ═══════════════════════════════════════════════════════════════
# 1. merge_data — normalize_price
# ═══════════════════════════════════════════════════════════════

class TestNormalizePrice:

    def test_same_year_returns_same_price(self):
        result = normalize_price(10_000.0, REFERENCE_YEAR)
        assert result == pytest.approx(10_000.0)

    def test_earlier_year_inflates_price(self):
        # Prices from earlier years should be higher when normalised
        result = normalize_price(10_000.0, 2015)
        assert result > 10_000.0

    def test_formula_correctness(self):
        year = 2020
        expected = 10_000.0 * (
            CPI_USED_CARS[REFERENCE_YEAR] / CPI_USED_CARS[year]
        )
        assert normalize_price(10_000.0, year) == pytest.approx(expected)

    def test_invalid_year_raises_value_error(self):
        with pytest.raises(ValueError, match="No CPI entry"):
            normalize_price(10_000.0, 1800)

    def test_all_cpi_years_work(self):
        for year in CPI_USED_CARS:
            result = normalize_price(1_000.0, year)
            assert result > 0

    def test_zero_price_stays_zero(self):
        assert normalize_price(0.0, 2020) == pytest.approx(0.0)

    def test_negative_price_preserved(self):
        # Function does not guard against negative prices; test for
        # consistent behaviour (not a crash).
        result = normalize_price(-500.0, 2020)
        assert result < 0


# ═══════════════════════════════════════════════════════════════
# 2. merge_data — price_tier
# ═══════════════════════════════════════════════════════════════

class TestPriceTier:

    @pytest.mark.parametrize("price,expected", [
        (0, "budget"),
        (1_000, "budget"),
        (4_999.99, "budget"),
        (5_000, "mid-range"),
        (10_000, "mid-range"),
        (14_999.99, "mid-range"),
        (15_000, "luxury"),
        (100_000, "luxury"),
    ])
    def test_tier_boundaries(self, price, expected):
        assert price_tier(price) == expected

    def test_returns_string(self):
        assert isinstance(price_tier(8_000), str)

    def test_exact_boundary_5000_is_mid_range(self):
        assert price_tier(5_000) == "mid-range"

    def test_exact_boundary_15000_is_luxury(self):
        assert price_tier(15_000) == "luxury"


# ═══════════════════════════════════════════════════════════════
# 3. merge_data — _transform_kaggle
# ═══════════════════════════════════════════════════════════════

class TestTransformKaggle:

    def test_datasource_set_to_kaggle(self):
        df = _kaggle_df()
        result = _transform_kaggle(df)
        assert (result["dataSource"] == "kaggle").all()

    def test_powerps_renamed_to_power(self):
        df = _kaggle_df()
        result = _transform_kaggle(df)
        assert "power" in result.columns
        assert "powerPS" not in result.columns

    def test_year_crawled_extracted_from_date(self):
        df = _kaggle_df(year=2022)
        result = _transform_kaggle(df)
        assert (result["yearCrawled"] == 2022).all()

    def test_price_normalised(self):
        df = _kaggle_df(year=2020)
        result = _transform_kaggle(df)
        expected = normalize_price(10_000.0, 2020)
        assert result["price"].iloc[0] == pytest.approx(expected)

    def test_price_tier_column_added(self):
        df = _kaggle_df()
        result = _transform_kaggle(df)
        assert "price_tier" in result.columns

    def test_price_tier_values_valid(self):
        df = _kaggle_df()
        result = _transform_kaggle(df)
        valid = {"budget", "mid-range", "luxury"}
        assert set(result["price_tier"].unique()).issubset(valid)

    def test_original_df_not_mutated(self):
        df = _kaggle_df()
        original_cols = set(df.columns)
        _transform_kaggle(df)
        assert set(df.columns) == original_cols


# ═══════════════════════════════════════════════════════════════
# 4. merge_data — _transform_crawled
# ═══════════════════════════════════════════════════════════════

class TestTransformCrawled:

    def test_datasource_set_to_crawled(self):
        df = _crawled_df()
        result = _transform_crawled(df)
        assert (result["dataSource"] == "crawled").all()

    def test_mileage_renamed_to_kilometer(self):
        df = _crawled_df()
        result = _transform_crawled(df)
        assert "kilometer" in result.columns
        assert "mileage" not in result.columns

    def test_power_extracted_from_string(self):
        df = _crawled_df()
        result = _transform_crawled(df)
        assert result["power"].iloc[0] == pytest.approx(162.0)

    def test_year_extracted_from_string(self):
        df = _crawled_df()
        result = _transform_crawled(df)
        assert result["yearOfRegistration"].iloc[0] == pytest.approx(2018)

    def test_price_tier_assigned(self):
        df = _crawled_df()
        result = _transform_crawled(df)
        assert "price_tier" in result.columns

    def test_original_df_not_mutated(self):
        df = _crawled_df()
        original_cols = set(df.columns)
        _transform_crawled(df)
        assert set(df.columns) == original_cols

    def test_power_nan_when_no_hp_pattern(self):
        df = _crawled_df()
        df["power"] = "120 kW"  # no parenthesised hp value
        result = _transform_crawled(df)
        assert result["power"].isna().all()


# ═══════════════════════════════════════════════════════════════
# 5. merge_data — transform_data (integration: merge pipeline)
# ═══════════════════════════════════════════════════════════════

class TestTransformData:

    def test_output_is_dataframe(self):
        result = transform_data(_kaggle_df(), _crawled_df())
        assert isinstance(result, pd.DataFrame)

    def test_output_has_target_cols(self):
        result = transform_data(_kaggle_df(), _crawled_df())
        assert set(TARGET_COLS).issubset(set(result.columns))

    def test_row_count_is_sum_of_inputs(self):
        k, c = 7, 4
        result = transform_data(_kaggle_df(k), _crawled_df(c))
        assert len(result) == k + c

    def test_datasource_values_both_present(self):
        result = transform_data(_kaggle_df(3), _crawled_df(3))
        sources = set(result["dataSource"].unique())
        assert "kaggle" in sources
        assert "crawled" in sources

    def test_index_is_reset(self):
        result = transform_data(_kaggle_df(5), _crawled_df(5))
        assert list(result.index) == list(range(len(result)))

    def test_price_tier_all_valid(self):
        result = transform_data(_kaggle_df(), _crawled_df())
        valid = {"budget", "mid-range", "luxury"}
        assert set(result["price_tier"].dropna().unique()).issubset(valid)

    def test_only_target_cols_returned(self):
        result = transform_data(_kaggle_df(), _crawled_df())
        assert list(result.columns) == TARGET_COLS

    def test_kaggle_prices_are_normalised(self):
        """Kaggle prices must differ from raw after CPI adjustment."""
        k_df = _kaggle_df(year=2015)
        c_df = _crawled_df(1)
        result = transform_data(k_df, c_df)
        kaggle_rows = result[result["dataSource"] == "kaggle"]
        raw_price = 10_000.0
        expected = normalize_price(raw_price, 2015)
        assert kaggle_rows["price"].iloc[0] == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════
# 6. validate_data — DataValidator._make_report / _fail / _store
# ═══════════════════════════════════════════════════════════════

class TestDataValidatorInternals:

    def setup_method(self):
        self.dv = DataValidator()

    def test_make_report_default_passed_true(self):
        r = self.dv._make_report("Test")
        assert r["passed"] is True
        assert r["issues"] == []
        assert r["check_type"] == "Test"

    def test_fail_sets_passed_false(self):
        r = self.dv._make_report("Test")
        self.dv._fail(r, "something broke")
        assert r["passed"] is False
        assert "something broke" in r["issues"]

    def test_fail_accumulates_multiple_issues(self):
        r = self.dv._make_report("Test")
        self.dv._fail(r, "issue 1")
        self.dv._fail(r, "issue 2")
        assert len(r["issues"]) == 2

    def test_store_appends_to_results(self):
        r = self.dv._make_report("Test")
        self.dv._store(r)
        assert r in self.dv.validation_results

    def test_store_returns_report(self):
        r = self.dv._make_report("Test")
        returned = self.dv._store(r)
        assert returned is r


# ═══════════════════════════════════════════════════════════════
# 7. validate_data — validate_schema
# ═══════════════════════════════════════════════════════════════

class TestValidateSchema:

    def setup_method(self):
        self.dv = DataValidator()

    def test_extra_column_flagged_as_issue_not_failure(self):
        df = _clean_df()
        df["extra_col"] = 0
        r = self.dv.validate_schema(df)
        # Extra columns are issues but do NOT flip passed=False
        assert any("extra_col" in i for i in r["issues"])

    def test_missing_column_fails(self):
        df = _clean_df().drop(columns=["brand"])
        r = self.dv.validate_schema(df)
        assert r["passed"] is False
        assert any("brand" in i for i in r["issues"])

    def test_col_count_stored_in_stats(self):
        df = _clean_df()
        r = self.dv.validate_schema(df)
        assert "col_cnt" in r["stats"]
        assert r["stats"]["expected_col_cnt"] == len(EXPECTED_COLUMNS)

    def test_dtype_checks_count_stored(self):
        df = _clean_df()
        r = self.dv.validate_schema(df)
        assert r["stats"]["dtype_checks"] == len(EXPECTED_DTYPES)

    # ── Bug documentation ──────────────────────────────────────
    def test_bug_expected_dtypes_str_version_dependent(self):
        """
        BUG NOTE (version-dependent): EXPECTED_DTYPES maps 'seller',
        'dataSource', and 'price_tier' to the Python string "str".

        - pandas < 2.x: string columns have dtype 'object', so "str"
          never matches → schema validation always flags them as
          dtype mismatches on valid DataFrames.
        - pandas >= 3.x: the default string dtype IS 'str', so "str"
          matches and no issue is raised.

        This test documents the dependency and asserts that str-typed
        column entries exist in EXPECTED_DTYPES (the underlying source
        of the bug regardless of pandas version).
        """
        str_cols = [c for c, t in EXPECTED_DTYPES.items() if t == "str"]
        assert len(str_cols) > 0, (
            "EXPECTED_DTYPES should still declare some cols as 'str'"
        )
        # Verify known columns are among them
        for col in ["seller", "dataSource", "price_tier"]:
            assert col in EXPECTED_DTYPES
            assert EXPECTED_DTYPES[col] == "str"

    def test_bug_price_dtype_int64_mismatches_float64(self):
        """
        BUG: EXPECTED_DTYPES declares 'price' as 'int64', but the
        pipeline always produces float64 prices (after CPI
        normalisation).  Schema will always fail on price dtype.
        """
        df = _clean_df()
        r = self.dv.validate_schema(df)
        assert any(
            "price" in i and "int64" in i for i in r["issues"]
        ), "Expected price int64/float64 dtype mismatch bug"


# ═══════════════════════════════════════════════════════════════
# 8. validate_data — validate_completeness
# ═══════════════════════════════════════════════════════════════

class TestValidateCompleteness:

    def setup_method(self):
        self.dv = DataValidator()

    def test_full_df_has_100_pct_score(self):
        df = _clean_df()
        r = self.dv.validate_completeness(df)
        assert r["stats"]["completeness_score_pct"] == pytest.approx(
            100.0
        )

    def test_missing_values_reduce_score(self):
        df = _clean_df()
        df.loc[0, "power"] = np.nan
        r = self.dv.validate_completeness(df)
        assert r["stats"]["completeness_score_pct"] < 100.0

    def test_column_detail_present_for_all_cols(self):
        df = _clean_df()
        r = self.dv.validate_completeness(df)
        for col in df.columns:
            assert col in r["stats"]["column_detail"]

    def test_exceeding_5pct_threshold_fails(self):
        df = _clean_df(n=100)
        # Inject >5% NaN into 'power' (which is in REQUIRED_COLUMNS
        # after the REQUIRED_COLUMNS bug is applied; see below)
        df.loc[:6, "power"] = np.nan  # 7 / 100 = 7%
        r = self.dv.validate_completeness(df)
        issues_str = " ".join(r["issues"])
        assert "power" in issues_str or not r["passed"]

    def test_empty_df_completeness_score_zero(self):
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        r = self.dv.validate_completeness(df)
        assert r["stats"]["completeness_score_pct"] == pytest.approx(
            0.0
        )

    # ── Bug documentation ──────────────────────────────────────
    def test_bug_required_columns_concatenation(self):
        """
        BUG (validate_data.py line 82-83): The list literal
        'price_tier' 'seller' (adjacent string literals) is silently
        concatenated by Python into the single token 'price_tierseller'.
        This means:
          • 'price_tier' and 'seller' are never individually required.
          • A column named 'price_tierseller' is always flagged missing.
        """
        assert "price_tierseller" in REQUIRED_COLUMNS, (
            "Bug present: adjacent string literal concatenation"
        )
        assert "seller" not in REQUIRED_COLUMNS, (
            "Bug present: 'seller' was dropped from REQUIRED_COLUMNS"
        )
        assert "price_tier" not in REQUIRED_COLUMNS, (
            "Bug present: 'price_tier' was dropped from REQUIRED_COLUMNS"
        )


# ═══════════════════════════════════════════════════════════════
# 9. validate_data — validate_uniqueness
# ═══════════════════════════════════════════════════════════════

class TestValidateUniqueness:

    def setup_method(self):
        self.dv = DataValidator()

    def test_unique_df_passes(self):
        df = _clean_df()
        r = self.dv.validate_uniqueness(df)
        assert r["passed"] is True

    def test_duplicate_rows_fail(self):
        df = _clean_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        r = self.dv.validate_uniqueness(df)
        assert r["passed"] is False
        assert r["stats"]["duplicate_rows"] >= 1

    def test_uniqueness_score_100_when_no_dups(self):
        df = _clean_df()
        r = self.dv.validate_uniqueness(df)
        assert r["stats"]["uniqueness_score_pct"] == pytest.approx(
            100.0
        )

    def test_uniqueness_score_less_than_100_with_dups(self):
        df = _clean_df()
        df = pd.concat([df, df], ignore_index=True)
        r = self.dv.validate_uniqueness(df)
        assert r["stats"]["uniqueness_score_pct"] < 100.0

    def test_empty_df_uniqueness_score_100(self):
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        r = self.dv.validate_uniqueness(df)
        assert r["stats"]["uniqueness_score_pct"] == pytest.approx(
            100.0
        )


# ═══════════════════════════════════════════════════════════════
# 10. validate_data — validate_validity
# ═══════════════════════════════════════════════════════════════

class TestValidateValidity:

    def setup_method(self):
        self.dv = DataValidator()

    def test_valid_df_range_no_violations(self):
        df = _clean_df()
        r = self.dv.validate_validity(df)
        range_checks = r["stats"]["range_checks"]
        for col, stats in range_checks.items():
            assert stats["violations"] == {}, (
                f"Unexpected violations in '{col}'"
            )

    def test_price_below_min_flagged(self):
        df = _clean_df()
        df.loc[0, "price"] = RANGE_RULES["price"]["min"] - 1
        r = self.dv.validate_validity(df)
        assert r["passed"] is False
        assert any("price" in i and "below min" in i
                   for i in r["issues"])

    def test_price_above_max_flagged(self):
        df = _clean_df()
        df.loc[0, "price"] = RANGE_RULES["price"]["max"] + 1
        r = self.dv.validate_validity(df)
        assert r["passed"] is False

    def test_invalid_categorical_flagged(self):
        df = _clean_df()
        df.loc[0, "fuelType"] = "steam"
        r = self.dv.validate_validity(df)
        assert r["passed"] is False
        assert any("fuelType" in i for i in r["issues"])

    def test_valid_categoricals_pass(self):
        df = _clean_df()
        r = self.dv.validate_validity(df)
        cat_checks = r["stats"]["categorical_checks"]
        for col, detail in cat_checks.items():
            assert detail["invalid_values"] == [], (
                f"'{col}' has unexpected invalid values"
            )

    def test_accuracy_score_100_when_valid(self):
        df = _clean_df()
        r = self.dv.validate_validity(df)
        for col, detail in r["stats"]["range_checks"].items():
            assert detail["accuracy_score_pct"] == pytest.approx(100.0)

    def test_column_missing_from_range_check_skipped(self):
        df = _clean_df().drop(columns=["power"])
        r = self.dv.validate_validity(df)
        assert "power" not in r["stats"]["range_checks"]


# ═══════════════════════════════════════════════════════════════
# 11. validate_data — validate_outliers_zscore
# ═══════════════════════════════════════════════════════════════

class TestValidateOutliersZScore:

    def setup_method(self):
        self.dv = DataValidator()

    def test_no_outliers_in_uniform_data(self):
        df = _clean_df()
        r = self.dv.validate_outliers_zscore(df)
        for col, detail in r["stats"]["column_detail"].items():
            assert detail["outlier_count"] == 0

    def test_extreme_outlier_detected(self):
        df = _clean_df(n=50)
        df.loc[0, "price"] = 999_999_999.0  # extreme
        r = self.dv.validate_outliers_zscore(df)
        price_stat = r["stats"]["column_detail"]["price"]
        assert price_stat["outlier_count"] >= 1

    def test_all_numeric_cols_checked(self):
        df = _clean_df()
        r = self.dv.validate_outliers_zscore(df)
        for col in NUMERIC_COLS:
            assert col in r["stats"]["column_detail"]

    def test_quality_score_100_when_no_outliers(self):
        df = _clean_df()
        r = self.dv.validate_outliers_zscore(df)
        for col, detail in r["stats"]["column_detail"].items():
            assert detail["quality_score_pct"] == pytest.approx(100.0)

    def test_less_than_2_rows_skipped(self):
        df = _clean_df(n=1)
        r = self.dv.validate_outliers_zscore(df)
        # With only 1 row per numeric col, nothing should be in detail
        assert r["stats"]["column_detail"] == {}


# ═══════════════════════════════════════════════════════════════
# 12. validate_data — validate_outliers_iqr
# ═══════════════════════════════════════════════════════════════

class TestValidateOutliersIqr:

    def setup_method(self):
        self.dv = DataValidator()

    def test_iqr_stats_computed(self):
        df = _clean_df()
        r = self.dv.validate_outliers_iqr(df)
        for col in NUMERIC_COLS:
            detail = r["stats"]["column_detail"][col]
            assert "Q1" in detail and "Q3" in detail and "IQR" in detail
            assert "lower_fence" in detail and "upper_fence" in detail

    def test_outlier_beyond_fence_detected(self):
        df = _clean_df(n=50)
        df.loc[0, "kilometer"] = 1_000_000.0  # way above fence
        r = self.dv.validate_outliers_iqr(df)
        km_stat = r["stats"]["column_detail"]["kilometer"]
        assert km_stat["outlier_count"] >= 1

    def test_quality_score_between_0_and_100(self):
        df = _clean_df()
        r = self.dv.validate_outliers_iqr(df)
        for col, detail in r["stats"]["column_detail"].items():
            assert 0.0 <= detail["quality_score_pct"] <= 100.0


# ═══════════════════════════════════════════════════════════════
# 13. validate_data — validate_outliers_isolation_forest
# ═══════════════════════════════════════════════════════════════

class TestValidateOutliersIsolationForest:

    def setup_method(self):
        self.dv = DataValidator()

    def test_insufficient_rows_fails(self):
        df = _clean_df(n=5)
        r = self.dv.validate_outliers_isolation_forest(df)
        assert r["passed"] is False
        assert any("10" in i for i in r["issues"])

    def test_sufficient_rows_runs(self):
        df = _clean_df(n=30)
        r = self.dv.validate_outliers_isolation_forest(df)
        assert "total_rows_evaluated" in r["stats"]
        assert "outlier_count" in r["stats"]

    def test_stats_keys_present(self):
        df = _clean_df(n=30)
        r = self.dv.validate_outliers_isolation_forest(df)
        for key in ["features_used", "outlier_count", "outlier_pct",
                    "quality_score_pct", "contamination_param"]:
            assert key in r["stats"]

    def test_contamination_param_stored(self):
        df = _clean_df(n=30)
        r = self.dv.validate_outliers_isolation_forest(
            df, contamination=0.05
        )
        assert r["stats"]["contamination_param"] == pytest.approx(0.05)

    def test_outlier_pct_matches_count(self):
        df = _clean_df(n=30)
        r = self.dv.validate_outliers_isolation_forest(df)
        n_rows = r["stats"]["total_rows_evaluated"]
        expected_pct = round(
            r["stats"]["outlier_count"] / n_rows * 100, 4
        )
        assert r["stats"]["outlier_pct"] == pytest.approx(expected_pct)


# ═══════════════════════════════════════════════════════════════
# 14. validate_data — validate_distribution
# ═══════════════════════════════════════════════════════════════

class TestValidateDistribution:

    def setup_method(self):
        self.dv = DataValidator()

    def test_stats_keys_computed(self):
        df = _clean_df()
        r = self.dv.validate_distribution(df)
        for col in NUMERIC_COLS:
            stat = r["stats"]["column_detail"][col]
            for key in ["mean", "median", "std", "Q1", "Q3",
                        "skewness", "kurtosis_excess",
                        "ks_statistic", "ks_pvalue"]:
                assert key in stat, (
                    f"'{key}' missing from distribution stats for {col}"
                )

    def test_mean_bounds_checked(self):
        df = _clean_df()
        r = self.dv.validate_distribution(df)
        for col in NUMERIC_COLS:
            stat = r["stats"]["column_detail"][col]
            assert "mean_bounds" in stat

    def test_skewness_label_assigned(self):
        df = _clean_df()
        r = self.dv.validate_distribution(df)
        valid_labels = {
            "Symmetric",
            "Positive Skew (right tail)",
            "Negative Skew (left tail)",
        }
        for col in NUMERIC_COLS:
            label = r["stats"]["column_detail"][col]["skewness_label"]
            assert label in valid_labels

    def test_kurtosis_label_assigned(self):
        df = _clean_df()
        r = self.dv.validate_distribution(df)
        valid_labels = {
            "Mesokurtic (normal)",
            "Leptokurtic (heavy tails)",
            "Platykurtic (light tails)",
        }
        for col in NUMERIC_COLS:
            label = r["stats"]["column_detail"][col]["kurtosis_label"]
            assert label in valid_labels

    def test_mean_out_of_bounds_fails(self):
        """A column with a mean far outside MEAN_BOUNDS triggers fail."""
        df = _clean_df(n=40)
        # Overwrite price to have a very low mean
        df["price"] = 1.0
        r = self.dv.validate_distribution(df)
        assert r["passed"] is False
        assert any("price" in i for i in r["issues"])

    def test_single_row_col_skipped(self):
        df = _clean_df(n=1)
        r = self.dv.validate_distribution(df)
        # With only 1 row, col_stats should be empty (len < 2 guard)
        assert r["stats"]["column_detail"] == {}


# ═══════════════════════════════════════════════════════════════
# 15. validate_data — validate_relationships
# ═══════════════════════════════════════════════════════════════

class TestValidateRelationships:

    def setup_method(self):
        self.dv = DataValidator()

    def test_matrices_computed(self):
        df = _clean_df()
        r = self.dv.validate_relationships(df)
        assert "pearson_matrix" in r["stats"]
        assert "spearman_matrix" in r["stats"]

    def test_correlation_check_keys_present(self):
        df = _clean_df()
        r = self.dv.validate_relationships(df)
        checks = r["stats"].get("expected_correlation_checks", {})
        for col_a, col_b, _, _ in EXPECTED_CORRELATIONS:
            key = f"{col_a}_vs_{col_b}"
            assert key in checks

    def test_each_check_has_required_fields(self):
        df = _clean_df()
        r = self.dv.validate_relationships(df)
        checks = r["stats"]["expected_correlation_checks"]
        for key, detail in checks.items():
            for field in ["correlation", "expected_range",
                          "status", "method"]:
                assert field in detail, (
                    f"'{field}' missing in check '{key}'"
                )

    def test_status_is_within_bounds_or_out_of_bounds(self):
        df = _clean_df()
        r = self.dv.validate_relationships(df)
        checks = r["stats"]["expected_correlation_checks"]
        valid_statuses = {"within bounds", "OUT OF BOUNDS"}
        for key, detail in checks.items():
            assert detail["status"] in valid_statuses

    def test_method_is_spearman(self):
        df = _clean_df()
        r = self.dv.validate_relationships(df)
        checks = r["stats"]["expected_correlation_checks"]
        for detail in checks.values():
            assert detail["method"] == "spearman"

    def test_columns_used_stored(self):
        df = _clean_df()
        r = self.dv.validate_relationships(df)
        assert r["stats"]["columns_used"] == NUMERIC_COLS


# ═══════════════════════════════════════════════════════════════
# 16. validate_data — generate_report
# ═══════════════════════════════════════════════════════════════

class TestGenerateReport:

    def setup_method(self):
        self.dv = DataValidator()
        df = _clean_df()
        self.dv.validate_schema(df)
        self.dv.validate_completeness(df)

    def test_returns_dict_with_required_keys(self, tmp_path):
        prefix = str(tmp_path / "rpt")
        result = self.dv.generate_report(output_prefix=prefix)
        for key in ["total", "passed", "failed",
                    "success_rate", "details"]:
            assert key in result

    def test_txt_file_created(self, tmp_path):
        prefix = str(tmp_path / "rpt")
        self.dv.generate_report(output_prefix=prefix)
        assert Path(prefix + ".txt").exists()

    def test_json_file_created(self, tmp_path):
        prefix = str(tmp_path / "rpt")
        self.dv.generate_report(output_prefix=prefix)
        assert Path(prefix + ".json").exists()

    def test_json_is_valid(self, tmp_path):
        prefix = str(tmp_path / "rpt")
        self.dv.generate_report(output_prefix=prefix)
        with open(prefix + ".json") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_counts_match(self, tmp_path):
        prefix = str(tmp_path / "rpt")
        result = self.dv.generate_report(output_prefix=prefix)
        total = result["total"]
        passed = result["passed"]
        failed = result["failed"]
        assert passed + failed == total

    def test_success_rate_formula(self, tmp_path):
        prefix = str(tmp_path / "rpt")
        result = self.dv.generate_report(output_prefix=prefix)
        if result["total"] > 0:
            expected_rate = result["passed"] / result["total"] * 100
            assert result["success_rate"] == pytest.approx(
                expected_rate
            )

    def test_empty_results_success_rate_zero(self, tmp_path):
        dv = DataValidator()
        prefix = str(tmp_path / "empty_rpt")
        result = dv.generate_report(output_prefix=prefix)
        assert result["success_rate"] == pytest.approx(0.0)
        assert result["total"] == 0


# ═══════════════════════════════════════════════════════════════
# 17. validate_data — run_all
# ═══════════════════════════════════════════════════════════════

class TestRunAll:

    def test_run_all_returns_dict(self, tmp_path):
        df = _clean_df(n=40)
        dv = DataValidator()
        result = dv.run_all(df, str(tmp_path / "run_all"))
        assert isinstance(result, dict)

    def test_run_all_executes_all_checks(self, tmp_path):
        df = _clean_df(n=40)
        dv = DataValidator()
        dv.run_all(df, str(tmp_path / "run_all"))
        check_types = [r["check_type"]
                       for r in dv.validation_results]
        expected_checks = [
            "Schema", "Completeness", "Uniqueness", "Validity",
            "Outliers_ZScore", "Outliers_IQR",
            "Outliers_IsolationForest", "Distribution",
            "Relationships",
        ]
        for ct in expected_checks:
            assert ct in check_types, (
                f"Check '{ct}' not executed by run_all"
            )

    def test_run_all_produces_report_files(self, tmp_path):
        df = _clean_df(n=40)
        dv = DataValidator()
        prefix = str(tmp_path / "full_report")
        dv.run_all(df, prefix)
        assert Path(prefix + ".txt").exists()
        assert Path(prefix + ".json").exists()

    def test_run_all_total_equals_check_count(self, tmp_path):
        df = _clean_df(n=40)
        dv = DataValidator()
        result = dv.run_all(df, str(tmp_path / "r"))
        assert result["total"] == len(dv.validation_results)


# ═══════════════════════════════════════════════════════════════
# 18. Integration — merge pipeline → validator
# ═══════════════════════════════════════════════════════════════

class TestMergeToValidateIntegration:
    """
    These tests drive the complete data flow:
      raw kaggle CSV + raw crawled CSV
      → transform_data()  (merge_data module)
      → DataValidator checks  (validate_data module)

    They verify that the merged output respects the shape, column,
    and range contracts expected by the validator.
    """

    def _merged_df(self, n_k: int = 20,
                   n_c: int = 20) -> pd.DataFrame:
        return transform_data(_kaggle_df(n_k), _crawled_df(n_c))

    # ── Shape / structure ───────────────────────────────────────

    def test_merged_has_all_target_cols(self):
        df = self._merged_df()
        assert set(TARGET_COLS).issubset(set(df.columns))

    def test_merged_row_count(self):
        df = self._merged_df(10, 15)
        assert len(df) == 25

    # ── Schema validation of merged output ──────────────────────

    def test_validator_receives_expected_columns(self):
        df = self._merged_df()
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        assert missing == set(), (
            f"Merged DF missing validator columns: {missing}"
        )

    def test_no_extra_columns_in_merged_output(self):
        df = self._merged_df()
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        assert extra == set(), (
            f"Unexpected extra columns in merged output: {extra}"
        )

    # ── Completeness validation ──────────────────────────────────

    def test_completeness_score_high_after_merge(self):
        df = self._merged_df(30, 30)
        dv = DataValidator()
        r = dv.validate_completeness(df)
        # Merged data may have some NaN (e.g. crawled power when
        # pattern fails), but the score should still be above 80%
        assert r["stats"]["completeness_score_pct"] >= 80.0

    # ── Uniqueness check ────────────────────────────────────────

    def test_no_cross_source_duplicates(self):
        """
        Rows from kaggle and crawled sources differ in brand, model,
        price, and dataSource.  Within each source the fixture rows are
        identical (same price, power, km), so intra-source duplicates
        exist by construction.

        This test verifies that the two *groups* are disjoint — i.e.
        no kaggle row is a full duplicate of a crawled row — by
        asserting that the duplicate count equals (n-1) per source
        (all identical within each block of n rows).
        """
        n = 5
        df = self._merged_df(n, n)
        dv = DataValidator()
        r = dv.validate_uniqueness(df)
        # Each source contributes n rows all identical → n-1 dups
        expected_dups = (n - 1) * 2
        assert r["stats"]["duplicate_rows"] == expected_dups

    def test_unique_rows_preserved_when_fixtures_differ(self):
        """Merged df with distinct rows per source has 0 duplicates."""
        import pandas as pd
        k_df = pd.DataFrame({
            "dateCrawled": ["2021-01-01 00:00:00"],
            "price": [8_000.0], "powerPS": [100.0],
            "brand": ["volkswagen"], "model": ["golf"],
            "vehicleType": ["sedan"], "gearbox": ["manual"],
            "kilometer": [50_000.0], "fuelType": ["gasoline"],
            "yearOfRegistration": [2018], "seller": ["private"],
            "dataSource": ["kaggle"], "price_tier": ["mid-range"],
        })
        c_df = pd.DataFrame({
            "mileage": [30_000.0], "price": [15_000.0],
            "power": ["200 kW (268 hp)"],
            "year": ["2020"], "brand": ["bmw"],
            "model": ["5-series"], "vehicleType": ["sedan"],
            "gearbox": ["automatic"], "fuelType": ["diesel"],
            "yearOfRegistration": [2020], "seller": ["dealer"],
            "dataSource": ["crawled"], "price_tier": ["luxury"],
        })
        from merge_data import transform_data
        merged = transform_data(k_df, c_df)
        dv = DataValidator()
        result = dv.validate_uniqueness(merged)
        assert result["stats"]["duplicate_rows"] == 0

    # ── Validity / range checks ─────────────────────────────────

    def test_prices_in_valid_range_after_merge(self):
        df = self._merged_df()
        lo = RANGE_RULES["price"]["min"]
        hi = RANGE_RULES["price"]["max"]
        assert (df["price"] >= lo).all()
        assert (df["price"] <= hi).all()

    def test_price_tier_values_recognised_by_validator(self):
        df = self._merged_df()
        dv = DataValidator()
        r = dv.validate_validity(df)
        cat_checks = r["stats"]["categorical_checks"]
        if "price_tier" in cat_checks:
            invalid = cat_checks["price_tier"]["invalid_values"]
            assert invalid == [], (
                f"price_tier has invalid values: {invalid}"
            )

    def test_datasource_values_recognised_by_validator(self):
        df = self._merged_df()
        dv = DataValidator()
        r = dv.validate_validity(df)
        cat_checks = r["stats"]["categorical_checks"]
        if "dataSource" in cat_checks:
            invalid = cat_checks["dataSource"]["invalid_values"]
            assert invalid == [], (
                f"dataSource has invalid values: {invalid}"
            )

    # ── Outlier pipeline ────────────────────────────────────────

    def test_zscore_outlier_check_runs_on_merged(self):
        df = self._merged_df(25, 25)
        dv = DataValidator()
        r = dv.validate_outliers_zscore(df)
        assert "column_detail" in r["stats"]

    def test_iqr_outlier_check_runs_on_merged(self):
        df = self._merged_df(25, 25)
        dv = DataValidator()
        r = dv.validate_outliers_iqr(df)
        assert "column_detail" in r["stats"]

    def test_isolation_forest_on_merged_sufficient_rows(self):
        df = self._merged_df(15, 15)
        dv = DataValidator()
        r = dv.validate_outliers_isolation_forest(df)
        assert "outlier_count" in r["stats"]

    # ── Full run_all on merged data ──────────────────────────────

    def test_run_all_on_merged_produces_report(self, tmp_path):
        df = self._merged_df(30, 30)
        dv = DataValidator()
        result = dv.run_all(df, str(tmp_path / "merged_report"))
        assert result["total"] == 9
        assert Path(str(tmp_path / "merged_report") + ".json").exists()

    def test_run_all_report_json_parseable(self, tmp_path):
        df = self._merged_df(30, 30)
        dv = DataValidator()
        prefix = str(tmp_path / "parseable_report")
        dv.run_all(df, prefix)
        with open(prefix + ".json") as fh:
            data = json.load(fh)
        assert "details" in data
        assert len(data["details"]) == 9

    # ── Data-flow integrity ──────────────────────────────────────

    def test_kaggle_normalised_prices_flow_to_validator(self):
        """
        Prices normalised in _transform_kaggle must survive unchanged
        through to the validator's range check.
        """
        k_df = _kaggle_df(n=5, year=2015)
        c_df = _crawled_df(n=5)
        merged = transform_data(k_df, c_df)
        kaggle_prices = merged[merged["dataSource"] == "kaggle"]["price"]
        expected = normalize_price(10_000.0, 2015)
        for p in kaggle_prices:
            assert p == pytest.approx(expected)

    def test_crawled_power_extracted_and_validated(self):
        """
        Power values extracted from the '(N hp)' string in
        _transform_crawled must be numeric float64 so the
        validator's Z-score / IQR checks can process them.
        """
        c_df = _crawled_df()
        merged = transform_data(_kaggle_df(3), c_df)
        crawled_power = merged[
            merged["dataSource"] == "crawled"
        ]["power"]
        assert crawled_power.dtype == "float64"
        assert crawled_power.notna().all()

    def test_both_sources_present_after_merge(self):
        merged = self._merged_df()
        sources = set(merged["dataSource"].unique())
        assert sources == {"kaggle", "crawled"}

    # ── Bug documentation ────────────────────────────────────────

    def test_bug_extract_year_returns_series_not_int(self):
        """
        BUG (merge_data.py extract_year): The function calls
        pd.Series(...).str.extract(r'(\\d{4})') which returns a
        DataFrame, not a scalar.  Then `int(match[0])` tries to
        convert a Series to int, which raises TypeError.
        The function is defined but never called by the pipeline
        directly; however, any downstream use will crash.
        """
        from merge_data import extract_year
        with pytest.raises(TypeError):
            extract_year("2018-05")

    def test_bug_extract_year_none_input_returns_none(self):
        """
        extract_year(None) correctly short-circuits to return None
        before reaching the broken code path.
        """
        from merge_data import extract_year
        assert extract_year(None) is None


# ═══════════════════════════════════════════════════════════════
# 19. Configuration / constant sanity checks
# ═══════════════════════════════════════════════════════════════

class TestConfigurationSanity:

    def test_cpi_used_cars_has_reference_year(self):
        assert REFERENCE_YEAR in CPI_USED_CARS

    def test_cpi_all_values_positive(self):
        for year, val in CPI_USED_CARS.items():
            assert val > 0, f"CPI for {year} is not positive"

    def test_target_cols_match_expected_cols(self):
        assert set(TARGET_COLS) == set(EXPECTED_COLUMNS)

    def test_numeric_cols_subset_of_expected(self):
        for col in NUMERIC_COLS:
            assert col in EXPECTED_COLUMNS

    def test_range_rules_bounds_sensible(self):
        for col, rules in RANGE_RULES.items():
            assert rules["min"] < rules["max"], (
                f"Range min >= max for '{col}'"
            )

    def test_mean_bounds_sensible(self):
        for col, (lo, hi) in MEAN_BOUNDS.items():
            assert lo < hi, f"Mean bounds lo >= hi for '{col}'"

    def test_expected_correlations_min_less_than_max(self):
        for col_a, col_b, min_r, max_r in EXPECTED_CORRELATIONS:
            assert min_r < max_r, (
                f"Correlation bounds inverted for {col_a} vs {col_b}"
            )

    def test_categorical_rules_non_empty(self):
        for col, allowed in CATEGORICAL_RULES.items():
            assert len(allowed) > 0, (
                f"CATEGORICAL_RULES for '{col}' is empty"
            )
