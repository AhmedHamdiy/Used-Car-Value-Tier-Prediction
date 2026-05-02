# from __future__ import annotations
# from src.data.validate_data import (
#     CATEGORICAL_RULES,
#     DataValidator,
#     EXPECTED_COLUMNS,
#     NUMERIC_COLS,
#     RANGE_RULES,
#     REQUIRED_COLUMNS,
# )
# import numpy as np
# import pandas as pd
# import pytest

# # ===========================================================================
# # Shared fixtures
# # ===========================================================================


# def _make_valid_df(n: int = 50) -> pd.DataFrame:
#     """Return a small but fully valid merged DataFrame."""
#     np.random.seed(42)
#     return pd.DataFrame(
#         {
#             "brand": ["volkswagen"] * n,
#             "model": ["golf"] * n,
#             "vehicleType": ["sedan"] * n,
#             "power": np.random.uniform(60, 200, n),
#             "gearbox": ["manual"] * n,
#             "kilometer": np.random.uniform(20_000, 120_000, n),
#             "fuelType": ["gasoline"] * n,
#             "yearOfRegistration": np.random.randint(
#                 2000, 2018, n
#             ).astype("int64"),
#             "seller": ["private"] * n,
#             "price": np.random.randint(3_000, 20_000, n).astype("int64"),
#         }
#     )


# @pytest.fixture()
# def valid_df() -> pd.DataFrame:
#     return _make_valid_df()


# @pytest.fixture()
# def validator() -> DataValidator:
#     return DataValidator()

# class TestInternalHelpers:
#     def test_make_report_keys(self, validator: DataValidator) -> None:
#         report = validator._make_report("Schema")
#         assert report["check_type"] == "Schema"
#         assert report["passed"] is True
#         assert report["issues"] == []
#         assert "timestamp" in report
#         assert "stats" in report

#     def test_fail_sets_passed_false(self, validator: DataValidator) -> None:
#         report = validator._make_report("Test")
#         validator._fail(report, "something went wrong")
#         assert report["passed"] is False
#         assert "something went wrong" in report["issues"]

#     def test_store_appends_result(self, validator: DataValidator) -> None:
#         report = validator._make_report("Test")
#         validator._store(report)
#         assert len(validator.validation_results) == 1

#     def test_fail_accumulates_multiple_issues(
#         self, validator: DataValidator
#     ) -> None:
#         report = validator._make_report("Multi")
#         validator._fail(report, "issue 1")
#         validator._fail(report, "issue 2")
#         assert len(report["issues"]) == 2

# class TestValidateSchema:
#     def test_passes_for_valid_df(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_schema(valid_df)
#         assert report["passed"] is True
#         assert report["issues"] == []

#     def test_fails_on_missing_column(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.drop(columns=["price"])
#         report = validator.validate_schema(df)
#         assert report["passed"] is False
#         assert any("price" in issue for issue in report["issues"])

#     def test_reports_extra_columns(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df["extra_col"] = 0
#         report = validator.validate_schema(df)
#         # Extra columns are noted but do not fail the check
#         assert any("Extra" in issue for issue in report["issues"])

#     def test_fails_on_dtype_mismatch(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df["price"] = df["price"].astype(str)
#         report = validator.validate_schema(df)
#         assert report["passed"] is False
#         assert any("price" in issue for issue in report["issues"])

#     def test_stats_contain_col_counts(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_schema(valid_df)
#         assert "col_cnt" in report["stats"]
#         assert "expected_col_cnt" in report["stats"]
#         assert report["stats"]["expected_col_cnt"] == len(EXPECTED_COLUMNS)


# # ===========================================================================
# # validate_completeness
# # ===========================================================================


# class TestValidateCompleteness:
#     def test_passes_for_complete_df(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_completeness(valid_df)
#         assert report["passed"] is True

#     def test_fails_when_missing_exceeds_threshold(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         # Null-out 20 % of 'price' (well above the 5 % threshold)
#         null_idx = df.sample(frac=0.2, random_state=0).index
#         df.loc[null_idx, "price"] = np.nan
#         report = validator.validate_completeness(df)
#         assert report["passed"] is False
#         assert any("price" in issue for issue in report["issues"])

#     def test_completeness_score_present(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_completeness(valid_df)
#         assert "completeness_score_pct" in report["stats"]
#         score = report["stats"]["completeness_score_pct"]
#         assert 0 <= score <= 100

#     def test_handles_missing_required_column(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.drop(columns=["price"])
#         report = validator.validate_completeness(df)
#         assert report["passed"] is False

#     def test_column_detail_populated(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_completeness(valid_df)
#         detail = report["stats"]["column_detail"]
#         for col in REQUIRED_COLUMNS:
#             if col in valid_df.columns:
#                 assert col in detail
#                 assert "missing_count" in detail[col]
#                 assert "missing_pct" in detail[col]


# # ===========================================================================
# # validate_uniqueness
# # ===========================================================================


# class TestValidateUniqueness:
#     def test_passes_for_unique_df(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_uniqueness(valid_df)
#         assert report["passed"] is True

#     def test_fails_on_duplicate_rows(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = pd.concat([valid_df, valid_df.iloc[:5]], ignore_index=True)
#         report = validator.validate_uniqueness(df)
#         assert report["passed"] is False
#         assert report["stats"]["duplicate_rows"] == 5

#     def test_uniqueness_score_100_for_clean(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_uniqueness(valid_df)
#         assert report["stats"]["uniqueness_score_pct"] == 100.0

#     def test_uniqueness_score_below_100_for_dups(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = pd.concfrom __future__ import annotations

# import re
# import warnings
# from pathlib import Path
# from typing import Any

# import numpy as np
# import pandas as pd

# warnings.filterwarnings("ignore")

# # ========================= CONFIGURATION =========================

# # Missing value placeholders (implicit)
# PLACEHOLDERS: set[str] = {
#     "", " ", "?", "N/A", "n/a", "NA", "na", "null", "NULL", "None",
#     "none", "unknown", "Unknown", "-", "--", "keine_angabe",
#     "andere", "sonstige", "nan"
# }

# # Mapping dictionaries
# FUEL_MAP: dict[str, str] = {
#     "Gasoline": "gasoline",
#     "benzin": "gasoline",
#     "diesel": "diesel",
#     "Electric": "electric",
#     "elektro": "electric",
#     "Electric/Gasoline": "hybrid",
#     "Electric/Diesel": "hybrid",
#     "hybrid": "hybrid",
#     "lpg": "lpg",
#     "cng": "cng",
#     "andere": "other",
# }

# VEHICLE_MAP: dict[str, str] = {
#     "Compact": "compact",
#     "kleinwagen": "compact",
#     "Station Wagon": "station_wagon",
#     "kombi": "station_wagon",
#     "SUV/Off-Road/Pick-Up": "suv",
#     "suv": "suv",
#     "Sedan": "sedan",
#     "limousine": "sedan",
#     "Van": "van",
#     "bus": "van",
#     "Convertible": "convertible",
#     "cabrio": "convertible",
#     "coupe": "coupe",
#     "Other": "other",
#     "andere": "other",
# }

# # brand mapping
# BRAND_ALIASES: dict[str, str] = {
#     'alfa':       'alfa-romeo',
#     'land':       'land-rover',
#     'aston':      'aston-martin',
#     'lynk':       'lynk-co',
#     'alpine':     'alpina',
#     'merc':       'mercedes-benz',
#     'vw':         'volkswagen',
#     'landrover':  'land-rover'
# }

# BRANDS_TO_DROP: set[str] = {
#     'sonstige-autos', 'unbekannt', 'keine_angabe'
# }

# # Simplified model mapping
# MODEL_ALIASES: dict[str, str] = {
#     'kaefer': 'beetle',
#     'käfer': 'beetle',
#     'new-beetle': 'beetle',
#     '1er': '1-series',
#     '2er': '2-series',
#     '3er': '3-series',
#     '4er': '4-series',
#     '5er': '5-series',
#     '6er': '6-series',
#     '7er': '7-series',
#     '8er': '8-series',
#     'a-klasse': 'a-class',
#     'b-klasse': 'b-class',
#     'c-klasse': 'c-class',
#     'e-klasse': 'e-class',
#     'g-klasse': 'g-class',
#     'm-klasse': 'm-class',
#     's-klasse': 's-class',
#     'v-klasse': 'v-class',
#     'x-klasse': 'x-class',
#     'up!': 'up',
#     'e-up!': 'e-up',
#     'ceed': 'ceed',
#     'andere' : 'other',
#     'unknown' : 'other',
# }

# MODELS_TO_DROP: set[str] = {
#     'sonstige', 'sonstige_autos', 'keine_angabe',
#     'nan', 'none'
# }

# # Categorical columns (after cleaning)
# CATEGORICAL_COLS: list[str] = [
#     "seller",
#     "vehicleType",
#     "gearbox",
#     "model",
#     "brand",
#     "fuelType",
#     "dataSource",
# ]

# # Expected columns for raw data (input)
# INPUT_COLS: list[str] = [
#     "brand",
#     "model",
#     "vehicleType",
#     "power",
#     "gearbox",
#     "kilometer",
#     "fuelType",
#     "yearOfRegistration",
#     "seller",
#     "dataSource",
#     "price_reference_year",
#     "price",
#     "price_normalized",
# ]

# # Domain constraints (based on validation report)
# YEAR_RANGE: tuple[int, int] = (1900, 2026)
# KM_RANGE: tuple[float, float] = (0.0, 300_000.0)
# MIN_PRICE: float = 500.0
# MAX_PRICE: float = 3_000_000.0
# POWER_MIN: float = 5.0
# POWER_MAX: float = 3000.0

# # Fixed capping bounds (domain knowledge)
# CAP_BOUNDS: dict[str, dict[str, float]] = {
#     "power": {"lower": POWER_MIN, "upper": POWER_MAX},
#     "price": {"lower": MIN_PRICE, "upper": MAX_PRICE},
#     "kilometer": {"lower": KM_RANGE[0], "upper": KM_RANGE[1]},
# }

# # Schema definition for validation (tutorial golden rule)
# SCHEMA: dict[str, dict[str, Any]] = {
#     "price": {"dtype": "float64", "nullable": False, "min": MIN_PRICE,
#               "max": MAX_PRICE},
#     "power": {"dtype": "float64", "nullable": False, "min": POWER_MIN,
#               "max": POWER_MAX},
#     "kilometer": {"dtype": "float64", "nullable": False, "min": KM_RANGE[0],
#                   "max": KM_RANGE[1]},
#     "yearOfRegistration": {"dtype": "int64", "nullable": False,
#                            "min": YEAR_RANGE[0], "max": YEAR_RANGE[1]},
#     "brand": {"dtype": "object", "nullable": False},
#     "model": {"dtype": "object", "nullable": True},
#     "vehicleType": {"dtype": "object", "nullable": True},
#     "gearbox": {"dtype": "object", "nullable": True},
#     "fuelType": {"dtype": "object", "nullable": True},
#     "seller": {"dtype": "object", "nullable": True},
#     "dataSource": {"dtype": "object", "nullable": True},
# }


# # ========================= HELPER FUNCTIONS =========================

# def replace_placeholders(series: pd.Series) -> pd.Series:
#     """Replace common placeholder strings with NaN."""
#     if series.dtype == "object":
#         stripped = series.astype(str).str.strip()
#         return stripped.where(~stripped.isin(PLACEHOLDERS), np.nan)
#     return series


# def clean_brand(series: pd.Series) -> pd.Series:
#     """Normalise brand names using aliases and drop unwanted brands."""
#     def _clean_one(val):
#         if pd.isna(val):
#             return np.nan
#         s = str(val).strip().lower().replace(r'[-_]', '-', regex=True)
#         # drop explicitly unwanted brands
#         if s in BRANDS_TO_DROP:
#             return np.nan
#         # apply alias mapping
#         return BRAND_ALIASES.get(s, s)

#     return series.apply(_clean_one)


# def clean_model(series: pd.Series) -> pd.Series:
#     """Normalise model names using aliases and pattern removal."""
#     def _clean_one(val):
#         if pd.isna(val):
#             return np.nan
#         s = str(val).strip().lower().replace(r'[-_]', '-', regex=True)
#         if s in MODELS_TO_DROP:
#             return np.nan
#         # apply alias mapping
#         s = MODEL_ALIASES.get(s, s)
#         # remove any remaining non‑alphanumeric characters except hyphen
#         s = re.sub(r"[^\w\-]", "", s)
#         return s if s else np.nan

#     return series.apply(_clean_one)


# def validate_schema(df: pd.DataFrame, schema: dict) -> list[str]:
#     """Validate a DataFrame against a schema (tutorial pattern)."""
#     violations = []
#     for col, rules in schema.items():
#         if col not in df.columns:
#             violations.append(f"MISSING COLUMN: {col}")
#             continue
#         # Null check
#         null_count = df[col].isnull().sum()
#         if not rules.get("nullable", True) and null_count > 0:
#             violations.append(
#                 f"{col}: {null_count} nulls in non-nullable column"
#             )
#         # Min/Max
#         if "min" in rules:
#             bad = (df[col].dropna() < rules["min"]).sum()
#             if bad:
#                 violations.append(
#                     f"{col}: {bad} values below min={rules['min']}"
#                 )
#         if "max" in rules:
#             bad = (df[col].dropna() > rules["max"]).sum()
#             if bad:
#                 violations.append(
#                     f"{col}: {bad} values above max={rules['max']}"
#                 )
#         # Allowed set (if present)
#         if "allowed" in rules:
#             allowed_set = set(rules["allowed"])
#             na_allowed = pd.isna(allowed_set) if allowed_set else False
#             data = df[col].dropna() if not na_allowed else df[col]
#             bad = (~data.isin(allowed_set)).sum()
#             if bad:
#                 violations.append(
#                     f"{col}: {bad} values not in {rules['allowed']}"
#                 )
#     return violations


# def _log_step(msg: str) -> None:
#     print(f"  ⚙  {msg}")


# def _log_ok(msg: str) -> None:
#     print(f"  ✅ {msg}")


# def _log_warn(msg: str) -> None:
#     print(f"  ⚠️  {msg}")


# # ========================= CLEANING STEPS =========================

# def load_and_coerce(path: str | Path) -> pd.DataFrame:
#     """Load CSV, keep only expected columns, coerce types, and clean
#     strings."""
#     _log_step(f"Loading data from {path} …")
#     df = pd.read_csv(path, low_memory=False)
#     _log_ok(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

#     # Keep only expected columns
#     missing_cols = set(INPUT_COLS) - set(df.columns)
#     if missing_cols:
#         raise ValueError(
#             f"Raw file is missing required columns: {missing_cols}"
#         )
#     df = df[INPUT_COLS].copy()

#     # Coerce numeric columns (errors → NaN)
#     for col in ["power", "kilometer", "yearOfRegistration", "price"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     # yearOfRegistration -> nullable integer
#     df["yearOfRegistration"] = (
#         df["yearOfRegistration"].fillna(-1).astype("int64")
#         .replace(-1, np.nan)
#     )

#     # String columns – strip whitespace, lower-case, replace placeholders
#     for col in CATEGORICAL_COLS:
#         if col in df.columns:
#             df[col] = (
#                 df[col].astype(str).str.strip().str.lower()
#                 .replace("nan", np.nan)
#             )
#             df[col] = replace_placeholders(df[col])

#     _log_ok("Schema coercion and placeholder replacement complete.")
#     return df


# def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
#     """Remove rows with invalid values based on domain constraints."""
#     _log_step("Removing invalid rows …")
#     before = len(df)

#     # price
#     df = df.dropna(subset=["price"])
#     df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)]

#     # year
#     year_valid = (
#         df["yearOfRegistration"].notna()
#         & (df["yearOfRegistration"] >= YEAR_RANGE[0])
#         & (df["yearOfRegistration"] <= YEAR_RANGE[1])
#     )
#     df = df[year_valid]

#     # kilometer
#     km_valid = (
#         df["kilometer"].notna()
#         & (df["kilometer"] >= KM_RANGE[0])
#         & (df["kilometer"] <= KM_RANGE[1])
#     )
#     df = df[km_valid]

#     # power
#     power_valid = (
#         df["power"].notna()
#         & (df["power"] >= POWER_MIN)
#         & (df["power"] <= POWER_MAX)
#     )
#     df = df[power_valid]

#     after = len(df)
#     removed = before - after
#     _log_ok(f"Removed {removed:,} invalid rows. Remaining: {after:,}.")
#     return df.reset_index(drop=True)


# def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
#     """Remove exact duplicate rows."""
#     _log_step("Dropping full‑row duplicates …")
#     before = len(df)
#     df = df.drop_duplicates(keep="first")
#     after = len(df)
#     removed = before - after
#     _log_ok(f"Dropped {removed:,} duplicate rows. Remaining: {after:,}.")
#     return df.reset_index(drop=True)


# def _mode_or_fallback(series: pd.Series, fallback: str) -> str:
#     mode_vals = series.dropna().mode()
#     return str(mode_vals.iloc[0]) if len(mode_vals) > 0 else fallback


# def impute_categoricals(df: pd.DataFrame) -> pd.DataFrame:
#     """Impute missing categorical values using brand-group mode,
#     then global mode."""
#     _log_step("Imputing missing categorical values …")

#     # First clean brand and model using alias functions
#     df["brand"] = clean_brand(df["brand"])
#     df["model"] = clean_model(df["model"])

#     impute_cols = [c for c in CATEGORICAL_COLS if c not in ("brand",)
#                    and c in df.columns]

#     for col in impute_cols:
#         # Group-wise mode imputation per brand (if brand exists)
#         if "brand" in df.columns and df["brand"].notna().any():
#             brand_modes = df.groupby("brand")[col].transform(
#                 _mode_or_fallback, fallback="unknown"
#             )
#             df[col] = df[col].fillna(brand_modes)

#         # Remaining NaN -> global mode
#         if df[col].isna().any():
#             global_mode = _mode_or_fallback(df[col], "unknown")
#             df[col] = df[col].fillna(global_mode)
#             _log_warn(
#                 f"'{col}': residual NaN filled with "
#                 f"global mode '{global_mode}'."
#             )

#     # Numeric imputation: median for power and year
#     for num_col in ["power", "yearOfRegistration"]:
#         if df[num_col].isna().any():
#             med = df[num_col].median()
#             df[num_col] = df[num_col].fillna(med)
#             _log_warn(f"'{num_col}': NaN filled with median {med:.1f}.")

#     _log_ok("Categorical imputation complete.")
#     return df


# def cap_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
#     """Cap outliers using the IQR rule (Tukey's fences)."""
#     _log_step("Capping outliers using IQR …")
#     for col in cols:
#         if col not in df.columns:
#             continue
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         before_lower = (df[col] < lower).sum()
#         before_upper = (df[col] > upper).sum()
#         df[col] = df[col].clip(lower=lower, upper=upper)
#         total = before_lower + before_upper
#         _log_ok(f"'{col}': capped {total:,} IQR outliers "
#                 f"(lower={lower:.1f}, upper={upper:.1f}).")
#     return df


# def cap_outliers_fixed(df: pd.DataFrame) -> pd.DataFrame:
#     """Cap outliers using fixed domain bounds."""
#     _log_step("Capping outliers with fixed bounds …")
#     for col, bounds in CAP_BOUNDS.items():
#         if col not in df.columns:
#             continue
#         upper = bounds.get("upper")
#         lower = bounds.get("lower")
#         capped = 0
#         if upper is not None:
#             before = (df[col] > upper).sum()
#             df[col] = df[col].clip(upper=upper)
#             capped += before
#         if lower is not None:
#             before = (df[col] < lower).sum()
#             df[col] = df[col].clip(lower=lower)
#             capped += before
#         _log_ok(f"'{col}': capped {capped:,} values "
#                 f"(upper={upper}, lower={lower}).")
#     return df


# # ========================= MAIN CLEANING FUNCTION =========================

# def clean_data(
#     raw_path: str | Path,
#     use_iqr_capping: bool = True,
#     output_path: str | Path | None = None,
# ) -> pd.DataFrame:
#     print("\n" + "=" * 65)
#     print("  DATA CLEANING PIPELINE")
#     print("=" * 65)

#     # Step 1 – Load and coerce
#     df = load_and_coerce(raw_path)
#     raw_shape = df.shape

#     # Optional: schema validation before cleaning
#     violations = validate_schema(df, SCHEMA)
#     if violations:
#         _log_warn(f"Schema violations before cleaning: {violations}")

#     # Step 2 – Remove invalid rows
#     df = remove_invalid_rows(df)

#     # Step 3 – Drop exact duplicates
#     df = drop_duplicates(df)

#     # Step 4 – Impute missing categoricals (also cleans brand/model)
#     df = impute_categoricals(df)

#     # Step 5 – Outlier handling
#     if use_iqr_capping:
#         df = cap_outliers_iqr(
#             df, ["price", "power", "kilometer", "yearOfRegistration"]
#         )
#     df = cap_outliers_fixed(df)

#     # Final validation (tutorial golden rule #3)
#     final_violations = validate_schema(df, SCHEMA)
#     if final_violations:
#         _log_warn(f"Post‑cleaning schema violations: {final_violations}")
#     else:
#         _log_ok("All schema checks passed after cleaning.")

#     # Report
#     print("\n" + "=" * 65)
#     print("  CLEANING REPORT")
#     print("=" * 65)
#     print(f"  Raw dataset        : {raw_shape[0]:,} rows × {raw_shape[1]} cols")
#     print(f"  Cleaned dataset    : {len(df):,} rows × {len(df.columns)} cols")
#     print(f"  Missing values     : {df.isna().sum().sum():,}")
#     print("=" * 65 + "\n")

#     if output_path:
#         df.to_csv(output_path, index=False)
#         _log_ok(f"Cleaned data saved to {output_path}")

#     return df


# class DataCleaner:

#     def __init__(self, use_iqr_capping: bool = True):
#         self.use_iqr_capping = use_iqr_capping
#         self.cleaned_df: pd.DataFrame | None = None

#     def run(self, raw_path: str | Path, output_path: str | Path | None = None
#             ) -> pd.DataFrame:
#         self.cleaned_df = clean_data(
#             raw_path,
#             use_iqr_capping=self.use_iqr_capping,
#             output_path=output_path
#         )
#         return self.cleaned_df
# at([valid_df, valid_df.iloc[:10]], ignore_index=True)
#         report = validator.validate_uniqueness(df)
#         assert report["stats"]["uniqueness_score_pct"] < 100.0


# # ===========================================================================
# # validate_validity
# # ===========================================================================


# class TestValidateValidity:
#     def test_passes_for_valid_df(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_validity(valid_df)
#         assert report["passed"] is True

#     def test_fails_on_price_below_min(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = -1
#         report = validator.validate_validity(df)
#         assert report["passed"] is False
#         assert any("price" in i for i in report["issues"])

#     def test_fails_on_price_above_max(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = 999_999_999
#         report = validator.validate_validity(df)
#         assert report["passed"] is False

#     def test_fails_on_power_above_max(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "power"] = 9_999.0
#         report = validator.validate_validity(df)
#         assert report["passed"] is False
#         assert any("power" in i for i in report["issues"])

#     def test_fails_on_invalid_categorical_seller(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "seller"] = "unknown_seller"
#         report = validator.validate_validity(df)
#         assert report["passed"] is False
#         assert any("seller" in i for i in report["issues"])

#     def test_fails_on_invalid_fueltype(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df["fuelType"] = "warp_plasma"
#         report = validator.validate_validity(df)
#         assert report["passed"] is False

#     def test_range_stats_structure(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_validity(valid_df)
#         range_checks = report["stats"]["range_checks"]
#         for col in RANGE_RULES:
#             if col in valid_df.columns:
#                 assert col in range_checks
#                 assert "accuracy_score_pct" in range_checks[col]

#     def test_categorical_stats_structure(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_validity(valid_df)
#         cat_checks = report["stats"]["categorical_checks"]
#         for col in CATEGORICAL_RULES:
#             if col in valid_df.columns:
#                 assert col in cat_checks


# # ===========================================================================
# # validate_outliers_zscore
# # ===========================================================================


# class TestValidateOutliersZScore:
#     def test_passes_when_no_outliers(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_outliers_zscore(valid_df)
#         # A clean dataset may still have natural Z-score outliers;
#         # just assert the report is structured correctly.
#         assert "column_detail" in report["stats"]

#     def test_detects_obvious_outlier(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         # Inject an extreme outlier (|z| >> 3)
#         df.loc[0, "price"] = 999_999_999
#         report = validator.validate_outliers_zscore(df)
#         assert report["stats"]["column_detail"]["price"][
#             "outlier_count"
#         ] >= 1

#     def test_quality_score_structure(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_outliers_zscore(valid_df)
#         for col in NUMERIC_COLS:
#             if col in valid_df.columns:
#                 detail = report["stats"]["column_detail"].get(col, {})
#                 if detail:
#                     assert "quality_score_pct" in detail
#                     assert 0 <= detail["quality_score_pct"] <= 100


# # ===========================================================================
# # validate_outliers_iqr
# # ===========================================================================


# class TestValidateOutliersIQR:
#     def test_passes_for_valid_df(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_outliers_iqr(valid_df)
#         assert "column_detail" in report["stats"]

#     def test_detects_iqr_outlier(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         # Force a value far outside the fences
#         df.loc[0, "kilometer"] = 1_000_000.0
#         report = validator.validate_outliers_iqr(df)
#         detail = report["stats"]["column_detail"]["kilometer"]
#         assert detail["outlier_count"] >= 1

#     def test_iqr_stats_keys(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_outliers_iqr(valid_df)
#         for col in NUMERIC_COLS:
#             detail = report["stats"]["column_detail"].get(col)
#             if detail is not None:
#                 for key in (
#                     "Q1", "Q3", "IQR",
#                     "lower_fence", "upper_fence",
#                     "outlier_count", "outlier_pct",
#                     "quality_score_pct",
#                 ):
#                     assert key in detail, (
#                         f"Key '{key}' missing for column '{col}'"
#                     )


# # ===========================================================================
# # validate_outliers_isolation_forest
# # ===========================================================================


# class TestValidateOutliersIsolationForest:
#     def test_runs_on_sufficient_data(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_outliers_isolation_forest(valid_df)
#         assert "outlier_count" in report["stats"]
#         assert "quality_score_pct" in report["stats"]

#     def test_fails_on_insufficient_rows(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.iloc[:5].copy()
#         report = validator.validate_outliers_isolation_forest(df)
#         assert report["passed"] is False
#         assert any("10" in i for i in report["issues"])

#     def test_contamination_stored_in_stats(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_outliers_isolation_forest(
#             valid_df, contamination=0.05
#         )
#         assert report["stats"]["contamination_param"] == 0.05


# # ===========================================================================
# # validate_distribution
# # ===========================================================================


# class TestValidateDistribution:
#     def test_stats_keys_present(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_distribution(valid_df)
#         assert "column_detail" in report["stats"]
#         for col in NUMERIC_COLS:
#             if col in valid_df.columns:
#                 detail = report["stats"]["column_detail"].get(col, {})
#                 if detail:
#                     for key in (
#                         "mean", "median", "std",
#                         "skewness", "kurtosis_excess",
#                         "ks_statistic", "ks_pvalue",
#                     ):
#                         assert key in detail, (
#                             f"Key '{key}' missing for '{col}'"
#                         )

#     def test_fails_on_mean_out_of_bounds(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         # Push price mean far above upper bound (30_000)
#         df["price"] = 5_000_000
#         df["price"] = df["price"].astype("int64")
#         report = validator.validate_distribution(df)
#         assert report["passed"] is False
#         assert any("price" in i for i in report["issues"])

#     def test_skips_column_with_single_value(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.iloc[:1].copy()
#         # Should not raise even with only one row
#         report = validator.validate_distribution(df)
#         assert isinstance(report, dict)


# # ===========================================================================
# # validate_relationships
# # ===========================================================================


# class TestValidateRelationships:
#     def test_pearson_and_spearman_matrices_present(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_relationships(valid_df)
#         assert "pearson_matrix" in report["stats"]
#         assert "spearman_matrix" in report["stats"]

#     def test_expected_correlation_checks_populated(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_relationships(valid_df)
#         checks = report["stats"].get("expected_correlation_checks", {})
#         assert len(checks) > 0

#     def test_correlation_status_field_present(
#         self, validator: DataValidator, valid_df: pd.DataFrame
#     ) -> None:
#         report = validator.validate_relationships(valid_df)
#         checks = report["stats"]["expected_correlation_checks"]
#         for key, val in checks.items():
#             assert "status" in val, f"'status' missing for {key}"
#             assert val["status"] in ("within bounds", "OUT OF BOUNDS")


# # ===========================================================================
# # run_all — integration smoke test
# # ===========================================================================


# class TestRunAll:
#     def test_run_all_returns_summary_dict(
#         self,
#         validator: DataValidator,
#         valid_df: pd.DataFrame,
#         tmp_path,
#     ) -> None:
#         output_prefix = str(tmp_path / "report")
#         result = validator.run_all(valid_df, output_file=output_prefix)
#         for key in ("total", "passed", "failed", "success_rate", "details"):
#             assert key in result, f"Key '{key}' missing from run_all result"

#     def test_run_all_writes_txt_and_json(
#         self,
#         validator: DataValidator,
#         valid_df: pd.DataFrame,
#         tmp_path,
#     ) -> None:
#         output_prefix = str(tmp_path / "report")
#         validator.run_all(valid_df, output_file=output_prefix)
#         assert (tmp_path / "report.txt").exists()
#         assert (tmp_path / "report.json").exists()

#     def test_run_all_total_equals_nine_checks(
#         self,
#         validator: DataValidator,
#         valid_df: pd.DataFrame,
#         tmp_path,
#     ) -> None:
#         output_prefix = str(tmp_path / "report")
#         result = validator.run_all(valid_df, output_file=output_prefix)
#         # Schema, Completeness, Uniqueness, Validity,
#         # ZScore, IQR, IsolationForest, Distribution, Relationships = 9
#         assert result["total"] == 9

#     def test_success_rate_between_0_and_100(
#         self,
#         validator: DataValidator,
#         valid_df: pd.DataFrame,
#         tmp_path,
#     ) -> None:
#         output_prefix = str(tmp_path / "report")
#         result = validator.run_all(valid_df, output_file=output_prefix)
#         assert 0.0 <= result["success_rate"] <= 100.0