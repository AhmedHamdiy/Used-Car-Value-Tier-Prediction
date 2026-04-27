"""
validate_data.py
================
Comprehensive data validation for the merged_data.csv car-listings dataset.

Covers all 8 quality dimensions from the course PDF:
  Dim 1 — Schema / Consistency   (column names, data types)
  Dim 2 — Completeness            (null / missing values)
  Dim 3 — Uniqueness              (duplicate rows / key columns)
  Dim 4 — Validity / Accuracy     (range rules, regex patterns, categorical sets)
  Dim 5 — Outlier Detection       (Z-Score, IQR, Isolation Forest)
  Dim 6 — Timeliness              (freshness, chronological order, gap detection)
  Dim 7 — Distribution Profile    (mean/median/mode, skewness, kurtosis, KS test)
  Dim 8 — Relationships           (Pearson correlation, Spearman correlation)

Usage
-----
    import pandas as pd
    from validate_data import DataValidator

    df = pd.read_csv("merged_data.csv")
    validator = DataValidator()

    validator.validate_schema(df)
    validator.validate_completeness(df)
    validator.validate_uniqueness(df)
    validator.validate_validity(df)
    validator.validate_outliers_zscore(df)
    validator.validate_outliers_iqr(df)
    validator.validate_outliers_isolation_forest(df)
    validator.validate_timeliness(df)
    validator.validate_distribution(df)
    validator.validate_relationships(df)

    summary = validator.generate_report()
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset schema definition (derived from the notebook exploration)
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: list[str] = [
    "dateCrawled",
    "name",
    "seller",
    "offerType",
    "price",
    "abtest",
    "vehicleType",
    "yearOfRegistration",
    "gearbox",
    "power",
    "model",
    "kilometer",
    "monthOfRegistration",
    "fuelType",
    "brand",
    "notRepairedDamage",
    "dateCreated",
    "nrOfPictures",
    "postalCode",
    "lastSeen",
]

EXPECTED_DTYPES: dict[str, str] = {
    "price": "int64",
    "yearOfRegistration": "int64",
    "power": "int64",
    "kilometer": "int64",
    "monthOfRegistration": "int64",
    "nrOfPictures": "int64",
    "postalCode": "int64",
}

# Dim 4 — range rules: {column: {"min": val, "max": val}}
RANGE_RULES: dict[str, dict[str, float]] = {
    "price":               {"min": 0,    "max": 1_000_000},
    "power":               {"min": 1,    "max": 4_999},
    "kilometer":           {"min": 0,    "max": 300_000},
    "yearOfRegistration":  {"min": 1950, "max": 2023},
    "monthOfRegistration": {"min": 0,    "max": 12},
}

# Dim 4 — categorical validity rules
CATEGORICAL_RULES: dict[str, list[str]] = {
    "seller": ["private", "dealer"],
    "offerType": ["Angebot", "Gesuch"],
    "abtest": ["test", "control"],
    "gearbox": ["manuell", "automatik"],
    "fuelType": ["benzin", "diesel", "lpg", "cng", "hybrid", "elektro", "andere"],
    "notRepairedDamage": ["nein", "ja"],
}

# Dim 4 — regex validity rules
REGEX_RULES: dict[str, str] = {
    "dateCrawled":  r"\d{4}-\d{2}-\d{2}",
    "dateCreated":  r"\d{4}-\d{2}-\d{2}",
    "lastSeen":     r"\d{4}-\d{2}-\d{2}",
}

# Dim 2 — required (must-not-be-null) columns
REQUIRED_COLUMNS: list[str] = [
    "price", "yearOfRegistration", "power", "kilometer", "brand",
]

# Dim 6 — timeliness config
DATE_COL = "dateCreated"            # column to use for freshness / order checks
FRESHNESS_THRESHOLD_DAYS = 365 * 2  # warn if newest record > 2 years old
EXPECTED_FREQ_DAYS = 7              # expect at least one record per 7 days gap

# Dim 7 — distribution: columns to profile
NUMERIC_COLS: list[str] = ["price", "power", "kilometer", "yearOfRegistration"]

# Dim 7 — distribution bounds for mean sanity checks (Dim 1 summary table row)
MEAN_BOUNDS: dict[str, tuple[float, float]] = {
    "price":              (500,  30_000),
    "power":              (50,   300),
    "kilometer":          (10_000, 200_000),
    "yearOfRegistration": (1990, 2020),
}

# Dim 8 — pairs expected to have strong positive / negative correlation
EXPECTED_CORRELATIONS: list[tuple[str, str, float, float]] = [
    # (col_a, col_b, min_r, max_r)
    ("kilometer", "yearOfRegistration", -0.9, 0.0),   # older → more km driven
]

# ---------------------------------------------------------------------------
# DataValidator class
# ---------------------------------------------------------------------------


class DataValidator:
    """Runs all 8 data-quality dimension checks and accumulates a report."""

    def __init__(self) -> None:
        self.validation_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_report(self, check_type: str) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "check_type": check_type,
            "passed": True,
            "issues": [],
            "stats": {},
        }

    def _fail(self, report: dict, msg: str) -> None:
        report["passed"] = False
        report["issues"].append(msg)

    def _store(self, report: dict) -> dict:
        self.validation_results.append(report)
        return report

    # ------------------------------------------------------------------
    # Dim 1 — Schema / Consistency
    # ------------------------------------------------------------------

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: list[str] | None = None,
        expected_dtypes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Check column presence and data types match the specification."""
        expected_columns = expected_columns or EXPECTED_COLUMNS
        expected_dtypes = expected_dtypes or EXPECTED_DTYPES

        report = self._make_report("Schema")

        # 1a. Missing columns
        missing = set(expected_columns) - set(df.columns)
        if missing:
            self._fail(report, f"Missing columns: {sorted(missing)}")

        # 1b. Extra (unexpected) columns
        extra = set(df.columns) - set(expected_columns)
        if extra:
            report["issues"].append(f"Extra columns (not in spec): {sorted(extra)}")

        # 1c. Column count
        report["stats"]["column_count"] = len(df.columns)
        report["stats"]["expected_column_count"] = len(expected_columns)

        # 1d. Data-type mismatches
        for col, expected_type in expected_dtypes.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    self._fail(
                        report,
                        f"Column '{col}': expected dtype '{expected_type}', "
                        f"got '{actual_type}'",
                    )
        report["stats"]["dtype_checks"] = len(expected_dtypes)

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 2 — Completeness
    # ------------------------------------------------------------------

    def validate_completeness(
        self,
        df: pd.DataFrame,
        required_columns: list[str] | None = None,
        max_missing_pct: float = 0.05,
    ) -> dict[str, Any]:
        """
        Check that required columns stay below max_missing_pct null rate.

        Parameters
        ----------
        max_missing_pct : float
            Fraction (0–1) of allowed nulls per column, e.g. 0.05 = 5%.
        """
        required_columns = required_columns or REQUIRED_COLUMNS
        report = self._make_report("Completeness")

        total_rows = len(df)
        col_stats: dict[str, dict] = {}

        for col in required_columns:
            if col not in df.columns:
                self._fail(report, f"Column '{col}' not found in DataFrame.")
                continue

            missing_count = int(df[col].isnull().sum())
            missing_pct = missing_count / total_rows if total_rows > 0 else 0.0
            col_stats[col] = {
                "missing_count": missing_count,
                "missing_pct": round(missing_pct * 100, 4),
            }

            if missing_pct > max_missing_pct:
                self._fail(
                    report,
                    f"Column '{col}': {missing_pct * 100:.2f}% missing "
                    f"(threshold {max_missing_pct * 100:.0f}%)",
                )

        # Overall completeness score:  (non-null values / total values) * 100
        all_required = [c for c in required_columns if c in df.columns]
        if all_required:
            non_null = df[all_required].notnull().sum().sum()
            total_cells = len(df) * len(all_required)
            report["stats"]["completeness_score_pct"] = round(
                non_null / total_cells * 100, 2
            )
        report["stats"]["column_detail"] = col_stats

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 3 — Uniqueness
    # ------------------------------------------------------------------

    def validate_uniqueness(
        self,
        df: pd.DataFrame,
        unique_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Check for duplicate rows and duplicate values in key columns."""
        unique_columns = unique_columns or []
        report = self._make_report("Uniqueness")

        # 3a. Full-row duplicates
        dup_rows = int(df.duplicated().sum())
        dup_pct = dup_rows / len(df) * 100 if len(df) else 0.0
        report["stats"]["duplicate_rows"] = dup_rows
        report["stats"]["duplicate_rows_pct"] = round(dup_pct, 4)
        if dup_rows > 0:
            self._fail(
                report,
                f"Full-row duplicates: {dup_rows} ({dup_pct:.2f}%)",
            )

        # 3b. Per-column uniqueness
        col_stats: dict[str, int] = {}
        for col in unique_columns:
            if col not in df.columns:
                self._fail(report, f"Column '{col}' not found for uniqueness check.")
                continue
            dup_count = int(df[col].duplicated().sum())
            col_stats[col] = dup_count
            if dup_count > 0:
                self._fail(
                    report,
                    f"Column '{col}': {dup_count} duplicate value(s) found.",
                )
        report["stats"]["column_duplicates"] = col_stats

        # 3c. Uniqueness score: unique records / total * 100
        report["stats"]["uniqueness_score_pct"] = round(
            (len(df) - dup_rows) / len(df) * 100, 2
        ) if len(df) else 100.0

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 4 — Validity / Accuracy
    # ------------------------------------------------------------------

    def validate_validity(
        self,
        df: pd.DataFrame,
        range_rules: dict[str, dict[str, float]] | None = None,
        categorical_rules: dict[str, list[str]] | None = None,
        regex_rules: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Validity / Accuracy dimension:
          * Range checks  (min / max)
          * Categorical membership
          * Regex pattern matching
        """
        range_rules = range_rules or RANGE_RULES
        categorical_rules = categorical_rules or CATEGORICAL_RULES
        regex_rules = regex_rules or REGEX_RULES

        report = self._make_report("Validity")

        range_stats: dict[str, dict] = {}
        cat_stats: dict[str, Any] = {}
        regex_stats: dict[str, Any] = {}

        # --- Range checks ---
        for col, rules in range_rules.items():
            if col not in df.columns:
                continue
            col_clean = df[col].dropna()
            violations: dict[str, int] = {}

            if "min" in rules:
                below = int((col_clean < rules["min"]).sum())
                if below > 0:
                    violations["below_min"] = below
                    self._fail(
                        report,
                        f"Column '{col}': {below} value(s) below min ({rules['min']})",
                    )

            if "max" in rules:
                above = int((col_clean > rules["max"]).sum())
                if above > 0:
                    violations["above_max"] = above
                    self._fail(
                        report,
                        f"Column '{col}': {above} value(s) above max ({rules['max']})",
                    )

            total_valid = len(col_clean) - sum(violations.values())
            accuracy_pct = (
                round(total_valid / len(col_clean) * 100, 2) if len(col_clean) else 100.0
            )
            range_stats[col] = {
                "violations": violations,
                "accuracy_score_pct": accuracy_pct,
            }

        # --- Categorical checks ---
        for col, allowed in categorical_rules.items():
            if col not in df.columns:
                continue
            actual_values = set(df[col].dropna().str.lower().unique())
            allowed_lower = {v.lower() for v in allowed}
            invalid = actual_values - allowed_lower
            cat_stats[col] = {
                "invalid_values": sorted(invalid),
                "allowed_values": sorted(allowed),
            }
            if invalid:
                self._fail(
                    report,
                    f"Column '{col}': invalid categorical values found: {sorted(invalid)}",
                )

        # --- Regex checks ---
        for col, pattern in regex_rules.items():
            if col not in df.columns:
                continue
            series = df[col].dropna().astype(str)
            non_matching = int((~series.str.contains(pattern, regex=True)).sum())
            regex_stats[col] = {"non_matching": non_matching, "pattern": pattern}
            if non_matching > 0:
                self._fail(
                    report,
                    f"Column '{col}': {non_matching} value(s) do not match "
                    f"pattern '{pattern}'",
                )

        report["stats"]["range_checks"] = range_stats
        report["stats"]["categorical_checks"] = cat_stats
        report["stats"]["regex_checks"] = regex_stats

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 5 — Outlier Detection: Z-Score
    # ------------------------------------------------------------------

    def validate_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        """
        Z-Score outlier detection: |z| > threshold  →  outlier.
        Best for approximately normal distributions.

        Metric: (non-outlier records / total) * 100
        """
        columns = columns or NUMERIC_COLS
        report = self._make_report("Outliers_ZScore")
        col_stats: dict[str, dict] = {}

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 2:
                continue
            z_scores = np.abs(stats.zscore(series))
            outlier_count = int((z_scores > threshold).sum())
            outlier_pct = round(outlier_count / len(series) * 100, 4)
            quality_score = round(100.0 - outlier_pct, 2)
            col_stats[col] = {
                "threshold": threshold,
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "quality_score_pct": quality_score,
            }
            if outlier_count > 0:
                self._fail(
                    report,
                    f"Column '{col}': {outlier_count} Z-Score outlier(s) "
                    f"(|z| > {threshold}) — {outlier_pct:.2f}% of values",
                )

        report["stats"]["column_detail"] = col_stats
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 5 — Outlier Detection: IQR
    # ------------------------------------------------------------------

    def validate_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        factor: float = 1.5,
    ) -> dict[str, Any]:
        """
        IQR outlier detection: value < Q1 − factor*IQR  or  > Q3 + factor*IQR.
        Robust method that works well with skewed distributions.

        Metric: (non-outlier records / total) * 100
        """
        columns = columns or NUMERIC_COLS
        report = self._make_report("Outliers_IQR")
        col_stats: dict[str, dict] = {}

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            Q1 = float(series.quantile(0.25))
            Q3 = float(series.quantile(0.75))
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR

            outlier_mask = (series < lower) | (series > upper)
            outlier_count = int(outlier_mask.sum())
            outlier_pct = round(outlier_count / len(series) * 100, 4)
            quality_score = round(100.0 - outlier_pct, 2)

            col_stats[col] = {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "quality_score_pct": quality_score,
            }
            if outlier_count > 0:
                self._fail(
                    report,
                    f"Column '{col}': {outlier_count} IQR outlier(s) "
                    f"(outside [{lower:.1f}, {upper:.1f}]) — {outlier_pct:.2f}%",
                )

        report["stats"]["column_detail"] = col_stats
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 5 — Outlier Detection: Isolation Forest (multivariate)
    # ------------------------------------------------------------------

    def validate_outliers_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Isolation Forest: ML-based multivariate outlier detection.
        Output: +1 = inlier, -1 = outlier.
        Best for detecting anomalies caused by a *combination* of features.

        Metric: (non-outlier records / total) * 100
        """
        columns = columns or NUMERIC_COLS
        report = self._make_report("Outliers_IsolationForest")

        available = [c for c in columns if c in df.columns]
        if not available:
            self._fail(report, "No numeric columns available for Isolation Forest.")
            return self._store(report)

        feature_df = df[available].dropna()
        if len(feature_df) < 10:
            self._fail(report, "Not enough rows for Isolation Forest (need >= 10).")
            return self._store(report)

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        predictions = model.fit_predict(feature_df)
        outlier_count = int((predictions == -1).sum())
        outlier_pct = round(outlier_count / len(feature_df) * 100, 4)
        quality_score = round(100.0 - outlier_pct, 2)

        report["stats"] = {
            "features_used": available,
            "total_rows_evaluated": len(feature_df),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "quality_score_pct": quality_score,
            "contamination_param": contamination,
        }

        if outlier_count > 0:
            self._fail(
                report,
                f"Isolation Forest detected {outlier_count} multivariate outlier(s) "
                f"({outlier_pct:.2f}%) across {available}",
            )

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 6 — Timeliness
    # ------------------------------------------------------------------

    def validate_timeliness(
        self,
        df: pd.DataFrame,
        date_col: str = DATE_COL,
        freshness_threshold_days: int = FRESHNESS_THRESHOLD_DAYS,
        expected_gap_days: int = EXPECTED_FREQ_DAYS,
        reference_now: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        """
        Timeliness dimension checks:
          T1 — Freshness:         latest record age vs threshold
          T2 — Chronological order: timestamps must be monotonic-increasing
          T3 — Gap detection:     no gap between consecutive records > 1.5× expected
          T4 — Duplicate timestamps: same datetime appears more than once
        """
        report = self._make_report("Timeliness")

        if date_col not in df.columns:
            self._fail(report, f"Date column '{date_col}' not found.")
            return self._store(report)

        ts = pd.to_datetime(df[date_col], errors="coerce")
        null_ts = int(ts.isnull().sum())
        if null_ts:
            report["issues"].append(
                f"Column '{date_col}': {null_ts} unparseable timestamp(s) (treated as NaT)."
            )
        ts_clean = ts.dropna().sort_values()
        if len(ts_clean) == 0:
            self._fail(report, "No valid timestamps found — timeliness check aborted.")
            return self._store(report)

        now = reference_now or pd.Timestamp.now()

        # T1 — Freshness
        latest = ts_clean.max()
        age_days = (now - latest).days
        report["stats"]["latest_record"] = str(latest)
        report["stats"]["age_days"] = age_days
        report["stats"]["freshness_threshold_days"] = freshness_threshold_days
        if age_days > freshness_threshold_days:
            self._fail(
                report,
                f"Freshness: latest record is {age_days} days old "
                f"(threshold: {freshness_threshold_days} days)",
            )

        # T2 — Chronological order (on the original, unsorted series)
        ts_orig = pd.to_datetime(df[date_col], errors="coerce").dropna()
        is_ordered = bool(ts_orig.is_monotonic_increasing)
        report["stats"]["chronological_order"] = is_ordered
        if not is_ordered:
            out_of_order = int((ts_orig < ts_orig.shift(1)).sum())
            self._fail(
                report,
                f"Chronological order: {out_of_order} out-of-order timestamp(s) found "
                f"(is_monotonic_increasing=False)",
            )

        # T3 — Gap detection
        gap_series = ts_clean.diff().dt.total_seconds() / 86400  # days
        gap_series = gap_series.dropna()
        threshold_gap = expected_gap_days * 1.5
        excessive_gaps = gap_series[gap_series > threshold_gap]
        report["stats"]["max_gap_days"] = round(float(gap_series.max()), 2) if len(gap_series) else 0.0
        report["stats"]["expected_gap_days"] = expected_gap_days
        if len(excessive_gaps) > 0:
            self._fail(
                report,
                f"Gap detection: {len(excessive_gaps)} gap(s) exceed "
                f"{threshold_gap:.1f} days (1.5× expected {expected_gap_days}d). "
                f"Max gap: {gap_series.max():.1f} days",
            )

        # T4 — Duplicate timestamps
        dup_ts = int(ts_orig.duplicated().sum())
        report["stats"]["duplicate_timestamps"] = dup_ts
        if dup_ts > 0:
            self._fail(
                report,
                f"Duplicate timestamps: {dup_ts} identical datetime(s) found in '{date_col}'",
            )

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 7 — Distribution Profile
    # ------------------------------------------------------------------

    def validate_distribution(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        mean_bounds: dict[str, tuple[float, float]] | None = None,
        reference_df: pd.DataFrame | None = None,
        ks_threshold: float = 0.2,
    ) -> dict[str, Any]:
        """
        Distribution dimension:
          * Descriptive statistics: mean, median, mode, std, min, max, Q1–Q3
          * Skewness classification (symmetric / positive / negative)
          * Kurtosis classification (mesokurtic / leptokurtic / platykurtic)
          * Mean sanity bounds check
          * KS test against reference_df (data-drift detection) if provided

        KS statistic interpretation (from the PDF):
          ≈ 0        : distributions identical
          0.05–0.15  : small difference
          > 0.2      : significant difference  ← threshold used here
          → 1        : completely different distributions
        """
        columns = columns or NUMERIC_COLS
        mean_bounds = mean_bounds or MEAN_BOUNDS
        report = self._make_report("Distribution")
        col_stats: dict[str, dict] = {}

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 2:
                continue

            mean_val = float(series.mean())
            median_val = float(series.median())
            mode_val = float(series.mode().iloc[0]) if len(series.mode()) > 0 else float("nan")
            std_val = float(series.std())
            skew_val = float(stats.skew(series))
            kurt_val = float(stats.kurtosis(series))  # excess kurtosis (normal = 0)

            # Skewness classification
            if abs(skew_val) < 0.5:
                skew_label = "Symmetric"
            elif skew_val > 0:
                skew_label = "Positive Skew (right tail)"
            else:
                skew_label = "Negative Skew (left tail)"

            # Kurtosis classification (excess kurtosis)
            if abs(kurt_val) < 0.5:
                kurt_label = "Mesokurtic (≈ normal)"
            elif kurt_val > 0:
                kurt_label = "Leptokurtic (heavy tails)"
            else:
                kurt_label = "Platykurtic (light tails)"

            col_stat: dict[str, Any] = {
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "mode": round(mode_val, 4),
                "std": round(std_val, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "Q1": round(float(series.quantile(0.25)), 4),
                "Q3": round(float(series.quantile(0.75)), 4),
                "skewness": round(skew_val, 4),
                "skewness_label": skew_label,
                "kurtosis_excess": round(kurt_val, 4),
                "kurtosis_label": kurt_label,
            }

            # Mean bounds check
            if col in mean_bounds:
                lo, hi = mean_bounds[col]
                col_stat["mean_bounds"] = (lo, hi)
                if not (lo <= mean_val <= hi):
                    self._fail(
                        report,
                        f"Column '{col}': mean={mean_val:.2f} outside expected "
                        f"bounds [{lo}, {hi}]",
                    )

            # KS test against reference distribution
            if reference_df is not None and col in reference_df.columns:
                ref_series = reference_df[col].dropna()
                ks_stat, ks_pvalue = stats.ks_2samp(series, ref_series)
                col_stat["ks_stat"] = round(ks_stat, 4)
                col_stat["ks_pvalue"] = round(ks_pvalue, 6)
                if ks_stat > ks_threshold:
                    self._fail(
                        report,
                        f"Column '{col}': KS statistic={ks_stat:.3f} > {ks_threshold} "
                        f"— significant distribution drift detected (p={ks_pvalue:.4f})",
                    )

            col_stats[col] = col_stat

        report["stats"]["column_detail"] = col_stats
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 8 — Relationships
    # ------------------------------------------------------------------

    def validate_relationships(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        expected_correlations: list[tuple[str, str, float, float]] | None = None,
        corr_method: str = "pearson",
    ) -> dict[str, Any]:
        """
        Relationship dimension:
          * Pearson correlation matrix (linear relationships)
          * Spearman correlation matrix (monotonic, rank-based)
          * Expected correlation bound checks

        Parameters
        ----------
        expected_correlations : list of (col_a, col_b, min_r, max_r)
            Each tuple defines a pair whose correlation is expected to fall
            within [min_r, max_r]. Deviations trigger a validation failure.
        corr_method : 'pearson' or 'spearman'
            Primary method for the expected-bounds check.
        """
        columns = columns or NUMERIC_COLS
        expected_correlations = expected_correlations or EXPECTED_CORRELATIONS
        report = self._make_report("Relationships")

        available = [c for c in columns if c in df.columns]
        feature_df = df[available].dropna()
        if len(feature_df) < 2:
            self._fail(report, "Not enough rows to compute correlations.")
            return self._store(report)

        pearson_matrix = feature_df.corr(method="pearson").round(4)
        spearman_matrix = feature_df.corr(method="spearman").round(4)

        report["stats"]["pearson_matrix"] = pearson_matrix.to_dict()
        report["stats"]["spearman_matrix"] = spearman_matrix.to_dict()
        report["stats"]["columns_used"] = available

        primary_matrix = pearson_matrix if corr_method == "pearson" else spearman_matrix

        # Check expected correlation bounds
        for col_a, col_b, min_r, max_r in expected_correlations:
            if col_a not in available or col_b not in available:
                continue
            r = float(primary_matrix.loc[col_a, col_b])
            status = "within bounds" if min_r <= r <= max_r else "OUT OF BOUNDS"
            report["stats"].setdefault("expected_correlation_checks", {})[
                f"{col_a}_vs_{col_b}"
            ] = {
                "correlation": r,
                "expected_range": (min_r, max_r),
                "status": status,
                "method": corr_method,
            }
            if status == "OUT OF BOUNDS":
                self._fail(
                    report,
                    f"Correlation '{col_a}' vs '{col_b}': r={r:.3f} not in "
                    f"expected range [{min_r}, {max_r}] (method={corr_method})",
                )

        return self._store(report)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> dict[str, Any]:
        """
        Aggregate all stored results and print a summary to stdout.

        Returns
        -------
        dict with keys:
            total, passed, failed, success_rate, details
        """
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r["passed"])
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0.0

        print("\n" + "=" * 65)
        print("  DATA VALIDATION REPORT")
        print("=" * 65)
        print(f"  Total checks : {total}")
        print(f"  Passed       : {passed}")
        print(f"  Failed       : {failed}")
        print(f"  Success rate : {success_rate:.1f}%")
        print("=" * 65)

        for result in self.validation_results:
            status = "PASS" if result["passed"] else "FAIL"
            marker = "✓" if result["passed"] else "✗"
            print(f"\n  [{status}] {marker} {result['check_type']} Check")
            for issue in result["issues"]:
                print(f"       ⚠  {issue}")

        print("\n" + "=" * 65 + "\n")

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "details": self.validation_results,
        }

    def run_all(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Convenience method: run every dimension check in order and return the
        aggregated report dict.
        """
        self.validate_schema(df)
        self.validate_completeness(df)
        self.validate_uniqueness(df)
        self.validate_validity(df)
        self.validate_outliers_zscore(df)
        self.validate_outliers_iqr(df)
        self.validate_outliers_isolation_forest(df)
        self.validate_timeliness(df)
        self.validate_distribution(df, reference_df=reference_df)
        self.validate_relationships(df)
        return self.generate_report()



if __name__ == "__main__":
    import os

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "clean", "merged_data.csv"
    )

    if not os.path.exists(data_path):
        print(f"[INFO] Demo: generating a synthetic 500-row dataset (real CSV not found).\n")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame(
            {
                "dateCrawled": pd.date_range("2016-01-01", periods=n, freq="h")
                .strftime("%Y-%m-%d %H:%M:%S"),
                "name": [f"Car_{i}" for i in range(n)],
                "seller": np.random.choice(["private", "dealer"], n),
                "offerType": "Angebot",
                "price": np.random.randint(500, 30000, n),
                "abtest": np.random.choice(["test", "control"], n),
                "vehicleType": np.random.choice(
                    ["limousine", "suv", "kleinwagen", "kombi"], n
                ),
                "yearOfRegistration": np.random.randint(1995, 2020, n),
                "gearbox": np.random.choice(["manuell", "automatik"], n),
                "power": np.random.randint(60, 300, n),
                "model": [f"model_{i % 20}" for i in range(n)],
                "kilometer": np.random.randint(5000, 200000, n),
                "monthOfRegistration": np.random.randint(0, 13, n),
                "fuelType": np.random.choice(["benzin", "diesel", "lpg"], n),
                "brand": np.random.choice(
                    ["volkswagen", "bmw", "mercedes_benz", "audi", "opel"], n
                ),
                "notRepairedDamage": np.random.choice(["nein", "ja"], n),
                "dateCreated": pd.date_range("2016-01-01", periods=n, freq="h")
                .strftime("%Y-%m-%d"),
                "nrOfPictures": np.zeros(n, dtype=int),
                "postalCode": np.random.randint(10000, 99999, n),
                "lastSeen": pd.date_range("2016-03-01", periods=n, freq="h")
                .strftime("%Y-%m-%d"),
            }
        )
        # Inject intentional errors
        df.loc[0, "price"] = -500          # below min
        df.loc[1, "power"] = 9999          # above max
        df.loc[2, "yearOfRegistration"] = 1900  # below min
        df.loc[3, "fuelType"] = "wood"     # invalid category
        df.loc[4, "price"] = None          # completeness
    else:
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    validator = DataValidator()
    validator.run_all(df)