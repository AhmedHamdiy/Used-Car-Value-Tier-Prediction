"""
data_validation.py
==================
Data validation pipeline for the car listings dataset.
Architecture follows the "Data Validation: Pandas vs. Great Expectations" slide deck.

Structure
---------
Part 1 — Pure Pandas  : DataValidator class covering all 6 quality dimensions
                        (Schema, Completeness, Ranges, Uniqueness, Categorical, Distribution)
Part 2 — Great Expectations v1.x : same checks via GX modern API with HTML Data Docs
Part 3 — Cleaning     : filter invalid rows, impute, dedup, price-tier, export
Part 4 — Visualisations
"""

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# (mirrors the "Defining Validation Rules" slide)
# ════════════════════════════════════════════════════════════════════════════
INPUT_PATH  = "../data/clean/merged_data.csv"
OUTPUT_CSV  = "../data/clean/validated_data.csv"
REPORTS_DIR = "../reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

CURRENT_YEAR = datetime.datetime.now().year   # dynamic — never hardcode the year

# --- Schema rules ---
EXPECTED_COLUMNS = ["price", "yearOfRegistration", "power", "kilometer",
                    "fuelType", "brand", "model", "vehicleType", "seller"]

EXPECTED_TYPES = {
    "price":               "float64",
    "yearOfRegistration":  "int64",
    "power":               "int64",
    "kilometer":           "int64",
    "fuelType":            "object",
    "brand":               "object",
    "model":               "object",
    "vehicleType":         "object",
    "seller":              "object",
}

# --- Completeness rules ---
REQUIRED_COLUMNS = ["price", "yearOfRegistration", "power", "kilometer", "brand"]
MAX_MISSING_PCT  = 0.05   # 5% tolerance

# --- Range rules (Accuracy dimension) ---
RANGE_RULES = {
    "power":              {"min": 1,          "max": 4999},
    "kilometer":          {"min": 0,          "max": 300_000},
    "yearOfRegistration": {"min": 1950,       "max": CURRENT_YEAR},
    "price":              {"min": 100,        "max": 150_000},
}

# --- Uniqueness rules ---
UNIQUE_COLUMNS = []   # e.g. ["listing_id"] — add if dataset has a PK column

# --- Categorical (Validity) rules ---
CATEGORICAL_RULES = {
    "fuelType":    ["petrol", "diesel", "lpg", "cng", "hybrid", "electro", "other"],
    "vehicleType": ["limousine", "kleinwagen", "kombi", "bus", "cabrio",
                    "coupe", "suv", "other"],
    "seller":      ["private", "dealer"],
}

# --- Distribution rules ---
DISTRIBUTION_RULES = {
    "price": {"mean_min": 5_000,  "mean_max": 30_000,
              "row_min":  10_000, "row_max":  500_000},
    "power": {"mean_min": 50,     "mean_max": 250},
}

# --- Skewness ---
SKEWNESS_THRESHOLD = 1.0

# --- Price tiering ---
PRICE_TIERS = {"bins":   [0, 3_500, 12_000, float("inf")],
               "labels": ["budget", "mid-range", "luxury"]}


def sep(title: str) -> None:
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


# ════════════════════════════════════════════════════════════════════════════
# PART 1 — PURE PANDAS  ▸  DataValidator class
# (follows the class architecture shown in the slides)
# ════════════════════════════════════════════════════════════════════════════

class DataValidator:
    """
    Pandas-based data validator covering all 6 quality dimensions.
    Each validate_*() method is independent and stores its report in
    self.validation_results so generate_report() can aggregate them.
    """

    def __init__(self):
        self.validation_results: list = []

    # ── internal helper ──────────────────────────────────────────────────
    def _make_report(self, check_type: str) -> dict:
        return {
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "check_type": check_type,
            "passed":     True,
            "issues":     [],
        }

    def _store(self, report: dict) -> dict:
        self.validation_results.append(report)
        return report

    # ── 1. SCHEMA — columns present & dtypes correct ─────────────────────
    def validate_schema(self,
                        df: pd.DataFrame,
                        expected_columns: list,
                        expected_types: dict) -> dict:
        """
        Check that all expected columns exist and their dtypes match.
        str(df[col].dtype) converts numpy dtype to string for comparison.
        Also checks for whitespace / mixed-case issues in string columns.
        """
        report = self._make_report("Schema")

        # 1a. Missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            report["passed"] = False
            report["issues"].append(f"Missing columns: {missing_cols}")

        # 1b. Extra columns not in spec (informational)
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            report["issues"].append(f"Extra columns (not in spec): {extra_cols}")

        # 1c. Dtype mismatches
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                # allow int64/float64 interchangeably for numeric cols
                both_numeric = {actual_type, expected_type} <= {"int64", "float64"}
                if actual_type != expected_type and not both_numeric:
                    report["passed"] = False
                    report["issues"].append(
                        f"Column '{col}': expected dtype '{expected_type}', "
                        f"got '{actual_type}'"
                    )

        # 1d. String-quality: whitespace / mixed case
        for col in df.select_dtypes(include="object").columns:
            series = df[col].dropna().astype(str)
            n_ws = (series != series.str.strip()).sum()
            if n_ws:
                report["issues"].append(
                    f"Column '{col}': {n_ws} rows have leading/trailing whitespace"
                )
            if series.str.lower().nunique() < series.nunique():
                report["issues"].append(
                    f"Column '{col}': mixed casing detected (e.g. 'BMW' vs 'bmw')"
                )

        return self._store(report)

    # ── 2. COMPLETENESS — no NULLs in required columns ───────────────────
    def validate_completeness(self,
                               df: pd.DataFrame,
                               required_columns: list,
                               max_missing_pct: float = 0.05) -> dict:
        """
        For each required column: count NULLs and compare against threshold.
        df[col].isnull().sum() -> count of NaN/None values.
        """
        report = self._make_report("Completeness")

        for col in required_columns:
            if col not in df.columns:
                continue
            missing_count = df[col].isnull().sum()
            missing_pct   = (missing_count / len(df)) * 100
            if missing_pct > max_missing_pct * 100:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': {missing_pct:.2f}% missing "
                    f"(allowed: {max_missing_pct * 100:.0f}%)"
                )

        return self._store(report)

    # ── 3. RANGES — numeric min/max bounds (Accuracy) ────────────────────
    def validate_ranges(self,
                        df: pd.DataFrame,
                        range_rules: dict) -> dict:
        """
        range_rules = { 'col': {'min': 0, 'max': 120} }
        df[df[col] < rules['min']] — boolean indexing to find violations.
        """
        report = self._make_report("Ranges")

        for col, rules in range_rules.items():
            if col not in df.columns:
                continue
            if "min" in rules:
                below = df[df[col] < rules["min"]]
                if len(below):
                    report["passed"] = False
                    report["issues"].append(
                        f"Column '{col}': {len(below)} values below minimum ({rules['min']})"
                    )
            if "max" in rules:
                above = df[df[col] > rules["max"]]
                if len(above):
                    report["passed"] = False
                    report["issues"].append(
                        f"Column '{col}': {len(above)} values above maximum ({rules['max']})"
                    )

        return self._store(report)

    # ── 4. UNIQUENESS — no duplicates in key columns ──────────────────────
    def validate_uniqueness(self,
                             df: pd.DataFrame,
                             unique_columns: list) -> dict:
        """
        .duplicated() -> True for all duplicates except first occurrence.
        Also reports full-row duplicates as informational.
        """
        report = self._make_report("Uniqueness")

        # 4a. Column-level uniqueness
        for col in unique_columns:
            if col not in df.columns:
                continue
            dup_count = df[col].duplicated().sum()
            if dup_count:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': {dup_count} duplicate values found"
                )

        # 4b. Full-row duplicates (informational)
        full_dup = df.duplicated().sum()
        if full_dup:
            report["issues"].append(
                f"Full-row duplicates: {full_dup} rows "
                f"({round(full_dup * 100 / len(df), 2)}% of dataset)"
            )

        return self._store(report)

    # ── 5. CATEGORICAL (Validity) — values in allowed set ─────────────────
    def validate_categorical(self,
                              df: pd.DataFrame,
                              categorical_rules: dict) -> dict:
        """
        actual_values = set(df[col].dropna().unique())
        invalid_values = actual_values - set(allowed_values)
        Set arithmetic is faster and cleaner than loops.
        .dropna() prevents NaN from triggering false invalid-category hits.
        """
        report = self._make_report("Categorical")

        for col, allowed_values in categorical_rules.items():
            if col not in df.columns:
                continue
            actual_values = set(
                df[col].dropna().astype(str).str.strip().str.lower().unique()
            )
            allowed_set = {v.lower() for v in allowed_values}
            invalid     = actual_values - allowed_set
            if invalid:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': invalid values found: {invalid}"
                )

        return self._store(report)

    # ── 6. DISTRIBUTION — statistical sanity checks ───────────────────────
    def validate_distribution(self,
                               df: pd.DataFrame,
                               distribution_rules: dict,
                               skewness_threshold: float = 1.0) -> dict:
        """
        Checks:
          - Column mean within expected range
          - Row count within expected range
          - Skewness flags (|skew| > threshold)
          - Cross-column consistency rules
        """
        report = self._make_report("Distribution")

        # 6a. Column mean checks
        for col, rules in distribution_rules.items():
            if col not in df.columns:
                continue
            actual_mean = df[col].mean()
            if "mean_min" in rules and actual_mean < rules["mean_min"]:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': mean {actual_mean:.2f} below "
                    f"expected min {rules['mean_min']}"
                )
            if "mean_max" in rules and actual_mean > rules["mean_max"]:
                report["passed"] = False
                report["issues"].append(
                    f"Column '{col}': mean {actual_mean:.2f} above "
                    f"expected max {rules['mean_max']}"
                )

        # 6b. Row count (checked once)
        for col, rules in distribution_rules.items():
            if "row_min" in rules and len(df) < rules["row_min"]:
                report["passed"] = False
                report["issues"].append(
                    f"Row count {len(df):,} below expected minimum {rules['row_min']:,}"
                )
            if "row_max" in rules and len(df) > rules["row_max"]:
                report["passed"] = False
                report["issues"].append(
                    f"Row count {len(df):,} above expected maximum {rules['row_max']:,}"
                )
            break

        # 6c. Skewness flags
        for col in df.select_dtypes(include="number").columns:
            skew = df[col].skew()
            if abs(skew) > skewness_threshold:
                report["issues"].append(
                    f"Column '{col}': high skewness ({skew:+.3f}) — "
                    f"consider log/sqrt transform"
                )

        # 6d. Cross-column consistency rules
        if "yearOfRegistration" in df.columns and "kilometer" in df.columns:
            old_low_km = (
                (df["yearOfRegistration"] < 1980) & (df["kilometer"] < 100)
            ).sum()
            if old_low_km:
                report["issues"].append(
                    f"Cross-column: {old_low_km} pre-1980 cars with < 100 km (suspicious)"
                )

        if "fuelType" in df.columns and "yearOfRegistration" in df.columns:
            early_ev = (
                (df["fuelType"].str.lower().str.strip() == "electro") &
                (df["yearOfRegistration"] < 2000)
            ).sum()
            if early_ev:
                report["issues"].append(
                    f"Cross-column: {early_ev} electric cars registered before 2000"
                )

        if "vehicleType" in df.columns and "power" in df.columns:
            weak_bus = (
                (df["vehicleType"].str.lower().str.strip() == "bus") &
                (df["power"] < 50)
            ).sum()
            if weak_bus:
                report["issues"].append(
                    f"Cross-column: {weak_bus} buses with power < 50 hp (suspicious)"
                )

        return self._store(report)

    # ── generate_report() — aggregate & print all results ─────────────────
    def generate_report(self) -> dict:
        """
        Aggregates all validate_*() results.
        Returns dict: total, passed, failed, success_rate, details.
        sum(1 for r in ... if r['passed']) — memory-efficient generator.
        """
        total        = len(self.validation_results)
        passed       = sum(1 for r in self.validation_results if r["passed"])
        failed       = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0.0

        print(f"\n{'─'*60}")
        print(f"  Validation Report  |  {passed}/{total} checks passed "
              f"({success_rate:.1f}%)")
        print(f"{'─'*60}")

        for result in self.validation_results:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"\n[{status}]  {result['check_type']} Check  "
                  f"(@ {result['timestamp']})")
            for issue in result["issues"]:
                print(f"         → {issue}")

        return {
            "total":        total,
            "passed":       passed,
            "failed":       failed,
            "success_rate": success_rate,
            "details":      self.validation_results,
        }


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — GREAT EXPECTATIONS v1.x
# (follows the 5-step GX workflow from the slides)
# ════════════════════════════════════════════════════════════════════════════

def run_great_expectations(df: pd.DataFrame) -> None:
    """
    GX v1.x Modern API — 5 steps:
      1. DataContext (ephemeral)
      2. DataSource -> DataAsset -> BatchDefinition -> Batch
      3. ExpectationSuite with all 6 quality dimensions
      4. ValidationDefinition
      5. Run & print report
    """
    try:
        import great_expectations as gx
    except ImportError:
        print("  ⚠  great_expectations not installed.")
        print("     Run: pip install great-expectations")
        print("  Skipping GX section — Pandas results above are complete.")
        return

    sep("PART 2 — GREAT EXPECTATIONS v1.x")

    # Step 1 — Context (ephemeral = no config files written to disk)
    context = gx.get_context(mode="ephemeral")

    # Step 2 — DataSource -> DataAsset -> BatchDefinition -> Batch
    data_source = context.data_sources.add_pandas(name="car_listings_source")
    data_asset  = data_source.add_dataframe_asset(name="car_listings_asset")
    batch_def   = data_asset.add_batch_definition_whole_dataframe("full_batch")

    # Step 3 — ExpectationSuite (all 6 dimensions)
    suite = context.suites.add(
        gx.ExpectationSuite(name="car_listings_suite")
    )
    E = gx.expectations   # alias

    # SCHEMA (Consistency)
    suite.add_expectation(
        E.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS,
            exact_match=False,
        )
    )
    for col in EXPECTED_COLUMNS:
        suite.add_expectation(E.ExpectColumnToExist(column=col))

    # COMPLETENESS
    for col in REQUIRED_COLUMNS:
        suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column=col))

    # ACCURACY / RANGES
    for col, rules in RANGE_RULES.items():
        suite.add_expectation(
            E.ExpectColumnValuesToBeBetween(
                column=col,
                min_value=rules.get("min"),
                max_value=rules.get("max"),
            )
        )

    # UNIQUENESS
    for col in UNIQUE_COLUMNS:
        suite.add_expectation(E.ExpectColumnValuesToBeUnique(column=col))

    # VALIDITY (Categorical)
    for col, allowed in CATEGORICAL_RULES.items():
        suite.add_expectation(
            E.ExpectColumnValuesToBeInSet(column=col, value_set=allowed)
        )

    # DISTRIBUTION
    suite.add_expectation(
        E.ExpectTableRowCountToBeBetween(
            min_value=DISTRIBUTION_RULES["price"]["row_min"],
            max_value=DISTRIBUTION_RULES["price"]["row_max"],
        )
    )
    suite.add_expectation(
        E.ExpectColumnMeanToBeBetween(
            column="price",
            min_value=DISTRIBUTION_RULES["price"]["mean_min"],
            max_value=DISTRIBUTION_RULES["price"]["mean_max"],
        )
    )
    suite.add_expectation(
        E.ExpectColumnMeanToBeBetween(
            column="power",
            min_value=DISTRIBUTION_RULES["power"]["mean_min"],
            max_value=DISTRIBUTION_RULES["power"]["mean_max"],
        )
    )

    # Step 4 — ValidationDefinition
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="car_listings_validation",
            data=batch_def,
            suite=suite,
        )
    )

    # Step 5 — Run
    results = validation_def.run(batch_parameters={"dataframe": df})
    _print_gx_report(results)

    # To generate HTML Data Docs, switch to mode='file' and run:
    # context.build_data_docs()
    # context.open_data_docs()


def _print_gx_report(results) -> None:
    """
    Parse GX ValidationResult and print a clean summary.
    Mirrors _print_report() from the slides.
    .expectation_config.type  -> expectation class name
    .expectation_config.kwargs -> rule parameters
    .result                   -> stats dict (unexpected_count, partial_unexpected_list)
    """
    total       = len(results.results)
    n_passed    = sum(1 for r in results.results if r.success)
    success_pct = round(n_passed / total * 100, 1) if total else 0

    print(f"\n{'─'*60}")
    print(f"  GX Report  |  {n_passed}/{total} expectations passed ({success_pct}%)")
    print(f"  Overall success: {results.success}")
    print(f"{'─'*60}")

    for exp_result in results.results:
        exp_type = exp_result.expectation_config.type
        col      = exp_result.expectation_config.kwargs.get("column", "table-level")
        status   = "✓ PASS" if exp_result.success else "✗ FAIL"
        print(f"\n  [{status}]  {exp_type}  |  column: {col}")

        if not exp_result.success and exp_result.result:
            r = exp_result.result
            if r.get("unexpected_count"):
                print(f"            Issues : {r['unexpected_count']} unexpected values")
            if r.get("partial_unexpected_list"):
                print(f"            Sample : {r['partial_unexpected_list'][:3]}")


# ════════════════════════════════════════════════════════════════════════════
# PART 3 — DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    pass