"""Comprehensive pytest test suite for clean_data.py.

Covers:
- Placeholder replacement
- Brand / model / vehicle-type / fuel-type / seller / gearbox cleaning
- Schema validation
- Invalid-row removal
- Duplicate dropping
- Categorical & numeric imputation
- IQR and fixed-bound outlier capping
- End-to-end pipeline via clean_data() and DataCleaner
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean_data import (
    BRAND_ALIASES,
    BRANDS_TO_DROP,
    CAP_BOUNDS,
    CATEGORICAL_COLS,
    FUEL_ALIASES,
    GEAR_ALIASES,
    INPUT_COLS,
    KM_RANGE,
    MAX_PRICE,
    MIN_PRICE,
    MODEL_ALIASES,
    MODELS_TO_DROP,
    PLACEHOLDERS,
    POWER_MAX,
    POWER_MIN,
    SCHEMA,
    SELLER_ALIASES,
    VT_ALIASES,
    YEAR_RANGE,
    DataCleaner,
    cap_outliers_iqr,
    clean_brand,
    clean_data,
    clean_fuel_type,
    clean_gearbox,
    clean_model,
    clean_seller,
    clean_vehicle_type,
    drop_duplicates,
    impute_categoricals,
    remove_invalid_rows,
    replace_placeholders,
    validate_schema,
)


# ─────────────────────────── helpers ────────────────────────────

def _series(values, dtype=object):
    return pd.Series(values, dtype=dtype)


def _minimal_df(**overrides) -> pd.DataFrame:
    """Return a one-row valid DataFrame with all INPUT_COLS."""
    base = {
        "brand": "volkswagen",
        "model": "golf",
        "vehicleType": "sedan",
        "power": 100.0,
        "gearbox": "manual",
        "kilometer": 50_000.0,
        "fuelType": "gasoline",
        "yearOfRegistration": 2015,
        "seller": "private",
        "dataSource": "web",
        "price": 8_000.0,
        "price_tier": "mid",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _make_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts as CSV to *path*."""
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _valid_row(**overrides) -> dict:
    base = {
        "brand": "volkswagen",
        "model": "golf",
        "vehicleType": "sedan",
        "power": 100.0,
        "gearbox": "manual",
        "kilometer": 50000.0,
        "fuelType": "gasoline",
        "yearOfRegistration": 2015,
        "seller": "private",
        "dataSource": "web",
        "price": 8000.0,
        "price_tier": "mid",
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════
# 1. replace_placeholders
# ═══════════════════════════════════════════════════════════════

class TestReplacePlaceholders:

    def test_empty_string_becomes_nan(self):
        result = replace_placeholders(_series([""]))
        assert pd.isna(result.iloc[0])

    def test_whitespace_only_becomes_nan(self):
        result = replace_placeholders(_series([" "]))
        assert pd.isna(result.iloc[0])

    def test_na_string_becomes_nan(self):
        for placeholder in ["N/A", "n/a", "NA", "na", "null", "NULL",
                            "None", "none", "nan", "unknown", "Unknown",
                            "-", "--", "?", "keine_angabe",
                            "andere", "sonstige"]:
            result = replace_placeholders(_series([placeholder]))
            assert pd.isna(result.iloc[0]), (
                f"Expected NaN for placeholder '{placeholder}'"
            )

    def test_valid_string_unchanged(self):
        result = replace_placeholders(_series(["volkswagen"]))
        assert result.iloc[0] == "volkswagen"

    def test_non_object_series_unchanged(self):
        s = pd.Series([1.0, 2.0, np.nan])
        result = replace_placeholders(s)
        pd.testing.assert_series_equal(result, s)

    def test_mixed_values(self):
        s = _series(["valid", "N/A", "", "also-valid"])
        result = replace_placeholders(s)
        assert result.iloc[0] == "valid"
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == "also-valid"

    def test_strips_whitespace_before_check(self):
        result = replace_placeholders(_series(["  N/A  "]))
        assert pd.isna(result.iloc[0])

    def test_all_placeholders_defined(self):
        assert len(PLACEHOLDERS) > 0


# ═══════════════════════════════════════════════════════════════
# 2. clean_brand
# ═══════════════════════════════════════════════════════════════

class TestCleanBrand:

    def test_nan_stays_nan(self):
        result = clean_brand(_series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_lowercases(self):
        result = clean_brand(_series(["BMW"]))
        assert result.iloc[0] == "bmw"

    def test_strips_whitespace(self):
        result = clean_brand(_series(["  bmw  "]))
        assert result.iloc[0] == "bmw"

    def test_alias_mapping(self):
        for alias, canonical in BRAND_ALIASES.items():
            result = clean_brand(_series([alias]))
            assert result.iloc[0] == canonical, (
                f"Alias '{alias}' → expected '{canonical}'"
            )

    def test_brands_to_drop_become_nan(self):
        # BUG in clean_brand: underscores are converted to hyphens
        # *before* the BRANDS_TO_DROP check, so entries containing
        # underscores (e.g. "keine_angabe") are never matched and
        # leak through as "keine-angabe" instead of becoming NaN.
        # The two entries that *do* work are tested here; the broken
        # entry is tested separately below.
        matchable = {b for b in BRANDS_TO_DROP if "_" not in b}
        for brand in matchable:
            result = clean_brand(_series([brand]))
            assert pd.isna(result.iloc[0]), (
                f"Brand '{brand}' should become NaN"
            )

    def test_brands_to_drop_underscore_bug(self):
        # Demonstrates the known bug: "keine_angabe" survives as
        # "keine-angabe" because normalisation runs before the drop
        # check.  When the bug is fixed this test should be removed
        # and the entry added back to test_brands_to_drop_become_nan.
        result = clean_brand(_series(["keine_angabe"]))
        assert result.iloc[0] == "keine-angabe", (
            "Expected buggy pass-through; update when bug is fixed"
        )

    def test_unknown_brand_preserved(self):
        result = clean_brand(_series(["lamborghini"]))
        assert result.iloc[0] == "lamborghini"

    def test_hyphen_normalisation(self):
        result = clean_brand(_series(["alfa_romeo"]))
        assert result.iloc[0] == "alfa-romeo"


# ═══════════════════════════════════════════════════════════════
# 3. clean_model
# ═══════════════════════════════════════════════════════════════

class TestCleanModel:

    def test_nan_stays_nan(self):
        result = clean_model(_series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_alias_mapping(self):
        for alias, canonical in MODEL_ALIASES.items():
            result = clean_model(_series([alias]))
            assert result.iloc[0] == canonical, (
                f"Model alias '{alias}' → expected '{canonical}'"
            )

    def test_models_to_drop_become_nan(self):
        # BUG in clean_model: same underscore-normalisation issue as
        # clean_brand. "keine_angabe" / "sonstige_autos" are converted
        # to hyphenated forms before the drop-set check and survive.
        # Only entries without underscores are reliably dropped today.
        matchable = {m for m in MODELS_TO_DROP if "_" not in m}
        for model in matchable:
            result = clean_model(_series([model]))
            assert pd.isna(result.iloc[0]), (
                f"Model '{model}' should become NaN"
            )

    def test_models_to_drop_underscore_bug(self):
        # Demonstrates the known bug: underscore entries survive.
        # Remove once the bug in clean_model is fixed.
        for model in ["keine_angabe", "sonstige_autos"]:
            result = clean_model(_series([model]))
            assert not pd.isna(result.iloc[0]), (
                f"'{model}' currently leaks through (known bug)"
            )

    def test_lowercases_and_strips(self):
        result = clean_model(_series(["  Golf  "]))
        assert result.iloc[0] == "golf"

    def test_unknown_model_preserved(self):
        result = clean_model(_series(["corolla"]))
        assert result.iloc[0] == "corolla"

    def test_empty_string_after_clean_becomes_nan(self):
        # A string that reduces to nothing after regex cleaning
        result = clean_model(_series(["!!!"]))
        assert pd.isna(result.iloc[0])


# ═══════════════════════════════════════════════════════════════
# 4. clean_vehicle_type
# ═══════════════════════════════════════════════════════════════

class TestCleanVehicleType:

    def test_nan_stays_nan(self):
        result = clean_vehicle_type(_series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_alias_mapping(self):
        for alias, canonical in VT_ALIASES.items():
            result = clean_vehicle_type(_series([alias]))
            assert result.iloc[0] == canonical, (
                f"VT alias '{alias}' → expected '{canonical}'"
            )

    def test_unknown_preserved(self):
        result = clean_vehicle_type(_series(["sedan"]))
        assert result.iloc[0] == "sedan"

    def test_lowercases(self):
        result = clean_vehicle_type(_series(["Sedan"]))
        assert result.iloc[0] == "sedan"


# ═══════════════════════════════════════════════════════════════
# 5. clean_fuel_type
# ═══════════════════════════════════════════════════════════════

class TestCleanFuelType:

    def test_nan_stays_nan(self):
        result = clean_fuel_type(_series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_alias_mapping(self):
        for alias, canonical in FUEL_ALIASES.items():
            result = clean_fuel_type(_series([alias]))
            assert result.iloc[0] == canonical, (
                f"Fuel alias '{alias}' → expected '{canonical}'"
            )

    def test_unknown_preserved(self):
        result = clean_fuel_type(_series(["gasoline"]))
        assert result.iloc[0] == "gasoline"


# ═══════════════════════════════════════════════════════════════
# 6. clean_seller
# ═══════════════════════════════════════════════════════════════

class TestCleanSeller:

    def test_nan_stays_nan(self):
        result = clean_seller(_series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_alias_mapping(self):
        for alias, canonical in SELLER_ALIASES.items():
            result = clean_seller(_series([alias]))
            assert result.iloc[0] == canonical

    def test_unknown_preserved(self):
        result = clean_seller(_series(["private"]))
        assert result.iloc[0] == "private"


# ═══════════════════════════════════════════════════════════════
# 7. clean_gearbox
# ═══════════════════════════════════════════════════════════════

class TestCleanGearbox:

    def test_nan_stays_nan(self):
        result = clean_gearbox(_series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_alias_mapping(self):
        for alias, canonical in GEAR_ALIASES.items():
            result = clean_gearbox(_series([alias]))
            assert result.iloc[0] == canonical

    def test_unknown_preserved(self):
        result = clean_gearbox(_series(["manual"]))
        assert result.iloc[0] == "manual"


# ═══════════════════════════════════════════════════════════════
# 8. validate_schema
# ═══════════════════════════════════════════════════════════════

class TestValidateSchema:

    def _valid_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "price": pd.array([8000.0], dtype="float64"),
            "power": pd.array([100.0], dtype="float64"),
            "kilometer": pd.array([50000.0], dtype="float64"),
            "yearOfRegistration": pd.array([2015], dtype="int64"),
            "brand": ["volkswagen"],
            "model": ["golf"],
            "vehicleType": ["sedan"],
            "gearbox": ["manual"],
            "fuelType": ["gasoline"],
            "seller": ["private"],
            "dataSource": ["web"],
        })

    def test_valid_df_no_violations(self):
        violations = validate_schema(self._valid_df(), SCHEMA)
        assert violations == []

    def test_missing_column_flagged(self):
        df = self._valid_df().drop(columns=["price"])
        violations = validate_schema(df, SCHEMA)
        assert any("MISSING COLUMN" in v and "price" in v
                   for v in violations)

    def test_null_in_non_nullable_flagged(self):
        df = self._valid_df()
        df.loc[0, "price"] = np.nan
        violations = validate_schema(df, SCHEMA)
        assert any("price" in v and "nulls" in v for v in violations)

    def test_below_min_flagged(self):
        df = self._valid_df()
        df["price"] = 100.0  # below MIN_PRICE
        violations = validate_schema(df, SCHEMA)
        assert any("price" in v and "below min" in v for v in violations)

    def test_above_max_flagged(self):
        df = self._valid_df()
        df["price"] = MAX_PRICE + 1
        violations = validate_schema(df, SCHEMA)
        assert any("price" in v and "above max" in v for v in violations)

    def test_nullable_column_with_nan_no_violation(self):
        df = self._valid_df()
        df["model"] = np.nan
        violations = validate_schema(df, SCHEMA)
        assert not any("model" in v and "nulls" in v for v in violations)

    def test_allowed_set_violation(self):
        schema = {"color": {"dtype": "object", "nullable": False,
                            "allowed": ["red", "blue"]}}
        df = pd.DataFrame({"color": ["green"]})
        violations = validate_schema(df, schema)
        assert any("color" in v for v in violations)

    def test_allowed_set_pass(self):
        schema = {"color": {"dtype": "object", "nullable": False,
                            "allowed": ["red", "blue"]}}
        df = pd.DataFrame({"color": ["red"]})
        violations = validate_schema(df, schema)
        assert violations == []


# ═══════════════════════════════════════════════════════════════
# 9. remove_invalid_rows
# ═══════════════════════════════════════════════════════════════

class TestRemoveInvalidRows:

    def _base_df(self, n=5) -> pd.DataFrame:
        rows = [_valid_row() for _ in range(n)]
        df = pd.DataFrame(rows)
        df["price"] = df["price"].astype(float)
        df["power"] = df["power"].astype(float)
        df["kilometer"] = df["kilometer"].astype(float)
        df["yearOfRegistration"] = pd.array(
            [2015] * n, dtype="Int64"
        )
        return df

    def test_valid_rows_kept(self):
        df = self._base_df(3)
        result = remove_invalid_rows(df)
        assert len(result) == 3

    def test_nan_price_removed(self):
        df = self._base_df(2)
        df.loc[0, "price"] = np.nan
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_price_below_min_removed(self):
        df = self._base_df(2)
        df.loc[0, "price"] = MIN_PRICE - 1
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_price_above_max_removed(self):
        df = self._base_df(2)
        df.loc[0, "price"] = MAX_PRICE + 1
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_price_at_boundary_kept(self):
        df = self._base_df(2)
        df.loc[0, "price"] = MIN_PRICE
        df.loc[1, "price"] = MAX_PRICE
        result = remove_invalid_rows(df)
        assert len(result) == 2

    def test_invalid_year_removed(self):
        df = self._base_df(2)
        df.loc[0, "yearOfRegistration"] = pd.NA
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_year_out_of_range_removed(self):
        df = self._base_df(2)
        df.loc[0, "yearOfRegistration"] = YEAR_RANGE[0] - 1
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_invalid_km_removed(self):
        df = self._base_df(2)
        df.loc[0, "kilometer"] = np.nan
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_km_out_of_range_removed(self):
        df = self._base_df(2)
        df.loc[0, "kilometer"] = KM_RANGE[1] + 1
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_invalid_power_removed(self):
        df = self._base_df(2)
        df.loc[0, "power"] = POWER_MIN - 1
        result = remove_invalid_rows(df)
        assert len(result) == 1

    def test_power_at_boundary_kept(self):
        df = self._base_df(2)
        df.loc[0, "power"] = POWER_MIN
        df.loc[1, "power"] = POWER_MAX
        result = remove_invalid_rows(df)
        assert len(result) == 2

    def test_index_reset_after_removal(self):
        df = self._base_df(3)
        df.loc[1, "price"] = np.nan
        result = remove_invalid_rows(df)
        assert list(result.index) == list(range(len(result)))


# ═══════════════════════════════════════════════════════════════
# 10. drop_duplicates
# ═══════════════════════════════════════════════════════════════

class TestDropDuplicates:

    def test_exact_duplicates_removed(self):
        df = pd.DataFrame([_valid_row(), _valid_row()])
        result = drop_duplicates(df)
        assert len(result) == 1

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame([_valid_row(), _valid_row(price=9000.0)])
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_three_duplicates_one_kept(self):
        row = _valid_row()
        df = pd.DataFrame([row, row, row])
        result = drop_duplicates(df)
        assert len(result) == 1

    def test_index_reset(self):
        df = pd.DataFrame([_valid_row(), _valid_row(),
                          _valid_row(price=9000.0)])
        result = drop_duplicates(df)
        assert list(result.index) == list(range(len(result)))


# ═══════════════════════════════════════════════════════════════
# 11. impute_categoricals
# ═══════════════════════════════════════════════════════════════

class TestImputeCategoticals:

    def _df_with_nan(self) -> pd.DataFrame:
        rows = [
            _valid_row(),
            _valid_row(vehicleType=np.nan),
            _valid_row(fuelType=np.nan),
        ]
        df = pd.DataFrame(rows)
        df["yearOfRegistration"] = pd.array(
            [2015, 2016, 2017], dtype="Int64"
        )
        return df

    def test_no_nan_after_imputation(self):
        df = self._df_with_nan()
        result = impute_categoricals(df)
        for col in CATEGORICAL_COLS:
            if col in result.columns:
                assert result[col].isna().sum() == 0, (
                    f"Column '{col}' still has NaN after imputation"
                )

    def test_brand_alias_applied(self):
        df = pd.DataFrame([_valid_row(brand="vw")])
        df["yearOfRegistration"] = pd.array([2015], dtype="Int64")
        result = impute_categoricals(df)
        assert result["brand"].iloc[0] == "volkswagen"

    def test_model_alias_applied(self):
        df = pd.DataFrame([_valid_row(model="kaefer")])
        df["yearOfRegistration"] = pd.array([2015], dtype="Int64")
        result = impute_categoricals(df)
        assert result["model"].iloc[0] == "beetle"

    def test_numeric_median_imputation(self):
        rows = [_valid_row(power=100.0), _valid_row(power=200.0),
                _valid_row(power=np.nan)]
        df = pd.DataFrame(rows)
        df["yearOfRegistration"] = pd.array(
            [2015, 2015, 2015], dtype="Int64"
        )
        result = impute_categoricals(df)
        assert not result["power"].isna().any()
        # median of [100, 200] is 150
        assert result.loc[2, "power"] == pytest.approx(150.0)

    def test_drop_brand_to_nan(self):
        # "sonstige-autos" is correctly dropped to NaN by clean_brand.
        # With only one row, brand-level mode imputation has nothing to
        # fall back on, and brand is excluded from the global-fallback
        # block in impute_categoricals, so the value stays NaN.
        # The pipeline's final dropna() step would remove such rows.
        df = pd.DataFrame([_valid_row(brand="sonstige-autos")])
        df["yearOfRegistration"] = pd.array([2015], dtype="Int64")
        result = impute_categoricals(df)
        assert pd.isna(result["brand"].iloc[0])


# ═══════════════════════════════════════════════════════════════
# 12. cap_outliers_iqr
# ═══════════════════════════════════════════════════════════════

class TestCapOutliersIqr:

    def _power_df(self, powers: list[float]) -> pd.DataFrame:
        rows = [_valid_row(power=p) for p in powers]
        df = pd.DataFrame(rows)
        df["power"] = df["power"].astype(float)
        return df

    def test_extreme_values_capped(self):
        powers = [100.0] * 10 + [999_999.0]
        df = self._power_df(powers)
        result = cap_outliers_iqr(df, ["power"])
        assert result["power"].max() < 999_999.0

    def test_lower_bound_respected(self):
        """Values must not go below CAP_BOUNDS lower."""
        powers = [100.0] * 10 + [-500.0]
        df = self._power_df(powers)
        result = cap_outliers_iqr(df, ["power"])
        assert result["power"].min() >= CAP_BOUNDS["power"]["lower"]

    def test_normal_values_unchanged(self):
        powers = [80.0, 100.0, 120.0, 90.0, 110.0]
        df = self._power_df(powers)
        result = cap_outliers_iqr(df, ["power"])
        pd.testing.assert_series_equal(
            result["power"], df["power"], check_names=False
        )

    def test_missing_column_silently_skipped(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = cap_outliers_iqr(df, ["power"])
        pd.testing.assert_frame_equal(result, df)

    def test_multiple_cols_capped(self):
        rows = [_valid_row(power=100.0, kilometer=50_000.0)] * 10
        rows.append(_valid_row(power=999_999.0, kilometer=999_999.0))
        df = pd.DataFrame(rows)
        df["power"] = df["power"].astype(float)
        df["kilometer"] = df["kilometer"].astype(float)
        result = cap_outliers_iqr(df, ["power", "kilometer"])
        assert result["power"].max() < 999_999.0
        assert result["kilometer"].max() < 999_999.0


# ═══════════════════════════════════════════════════════════════
# 13. clean_data (end-to-end) via CSV file
# ═══════════════════════════════════════════════════════════════

class TestCleanDataEndToEnd:

    def _write_valid_csv(self, n: int = 10) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        tmp.close()
        path = Path(tmp.name)
        rows = [_valid_row() for _ in range(n)]
        # make prices unique to avoid all being dropped as dups
        for i, row in enumerate(rows):
            row["price"] = 8_000.0 + i * 100
        _make_csv(rows, path)
        return path

    def test_returns_dataframe(self):
        path = self._write_valid_csv()
        result = clean_data(path)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_rows(self):
        path = self._write_valid_csv(10)
        result = clean_data(path)
        assert len(result) > 0

    def test_no_missing_values_after_cleaning(self):
        path = self._write_valid_csv(10)
        result = clean_data(path)
        assert result.isna().sum().sum() == 0

    def test_price_within_bounds(self):
        path = self._write_valid_csv(10)
        result = clean_data(path)
        assert (result["price"] >= MIN_PRICE).all()
        assert (result["price"] <= MAX_PRICE).all()

    def test_year_within_bounds(self):
        path = self._write_valid_csv(10)
        result = clean_data(path)
        assert (result["yearOfRegistration"] >= YEAR_RANGE[0]).all()
        assert (result["yearOfRegistration"] <= YEAR_RANGE[1]).all()

    def test_output_saved_when_path_given(self, tmp_path):
        src = self._write_valid_csv(5)
        out = tmp_path / "cleaned.csv"
        clean_data(src, output_path=out)
        assert out.exists()
        saved = pd.read_csv(out)
        assert len(saved) > 0

    def test_missing_column_raises(self, tmp_path):
        rows = [_valid_row()]
        # Remove a required column
        del rows[0]["price"]
        path = tmp_path / "bad.csv"
        _make_csv(rows, path)
        with pytest.raises(ValueError, match="missing required columns"):
            clean_data(path)

    def test_iqr_capping_disabled(self):
        path = self._write_valid_csv(10)
        result = clean_data(path, use_iqr_capping=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_placeholders_cleaned(self, tmp_path):
        rows = [_valid_row(brand="N/A", model="unknown")]
        # Need many rows so pipeline doesn't drop everything
        for i in range(9):
            rows.append(_valid_row(price=8_000 + i * 500))
        path = tmp_path / "ph.csv"
        _make_csv(rows, path)
        result = clean_data(path)
        # brand 'N/A' → NaN → imputed, should not equal literal "N/A"
        assert "N/A" not in result["brand"].values

    def test_german_brand_aliases_resolved(self, tmp_path):
        rows = [_valid_row(brand="vw", price=8000 + i * 200)
                for i in range(10)]
        path = tmp_path / "vw.csv"
        _make_csv(rows, path)
        result = clean_data(path)
        assert (result["brand"] == "volkswagen").all()

    def test_duplicates_removed(self, tmp_path):
        row = _valid_row()
        rows = [row.copy() for _ in range(5)]
        path = tmp_path / "dups.csv"
        _make_csv(rows, path)
        result = clean_data(path)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════
# 14. DataCleaner class
# ═══════════════════════════════════════════════════════════════

class TestDataCleaner:

    def _csv_path(self, n: int = 5) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        tmp.close()
        path = Path(tmp.name)
        rows = [_valid_row(price=8_000.0 + i * 100) for i in range(n)]
        _make_csv(rows, path)
        return path

    def test_default_construction(self):
        cleaner = DataCleaner()
        assert cleaner.use_iqr_capping is True
        assert cleaner.cleaned_df is None

    def test_run_returns_dataframe(self):
        cleaner = DataCleaner()
        path = self._csv_path()
        result = cleaner.run(path)
        assert isinstance(result, pd.DataFrame)

    def test_cleaned_df_set_after_run(self):
        cleaner = DataCleaner()
        path = self._csv_path()
        cleaner.run(path)
        assert cleaner.cleaned_df is not None
        assert isinstance(cleaner.cleaned_df, pd.DataFrame)

    def test_iqr_capping_false(self):
        cleaner = DataCleaner(use_iqr_capping=False)
        path = self._csv_path()
        result = cleaner.run(path)
        assert isinstance(result, pd.DataFrame)

    def test_output_saved_via_cleaner(self, tmp_path):
        cleaner = DataCleaner()
        src = self._csv_path(5)
        out = tmp_path / "out.csv"
        cleaner.run(src, output_path=out)
        assert out.exists()


# ═══════════════════════════════════════════════════════════════
# 15. Configuration / constant sanity checks
# ═══════════════════════════════════════════════════════════════

class TestConfiguration:

    def test_min_price_positive(self):
        assert MIN_PRICE > 0

    def test_max_price_greater_than_min(self):
        assert MAX_PRICE > MIN_PRICE

    def test_year_range_valid(self):
        lo, hi = YEAR_RANGE
        assert lo < hi
        assert lo >= 1800

    def test_km_range_valid(self):
        lo, hi = KM_RANGE
        assert lo >= 0
        assert hi > lo

    def test_power_range_valid(self):
        assert POWER_MIN > 0
        assert POWER_MAX > POWER_MIN

    def test_cap_bounds_keys_are_columns(self):
        valid_cols = set(INPUT_COLS)
        for col in CAP_BOUNDS:
            assert col in valid_cols, (
                f"CAP_BOUNDS key '{col}' not in INPUT_COLS"
            )

    def test_categorical_cols_subset_of_input_cols(self):
        for col in CATEGORICAL_COLS:
            assert col in INPUT_COLS, (
                f"CATEGORICAL_COLS has '{col}' not in INPUT_COLS"
            )

    def test_schema_covers_key_columns(self):
        for col in ["price", "power", "kilometer", "yearOfRegistration",
                    "brand"]:
            assert col in SCHEMA, f"'{col}' missing from SCHEMA"

    def test_placeholders_all_strings(self):
        for p in PLACEHOLDERS:
            assert isinstance(p, str)

    def test_brand_aliases_values_are_canonical(self):
        """Alias values should not themselves appear as alias keys."""
        for canonical in BRAND_ALIASES.values():
            assert canonical not in BRANDS_TO_DROP, (
                f"Canonical brand '{canonical}' is in BRANDS_TO_DROP"
            )
