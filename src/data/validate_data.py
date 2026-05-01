from __future__ import annotations

import json
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.ensemble import IsolationForest


warnings.filterwarnings("ignore")
EXPECTED_COLUMNS: list[str] = [
    "brand",
    "model",
    "vehicleType",
    "power",
    "gearbox",
    "kilometer",
    "fuelType",
    "yearOfRegistration",
    "seller",
    "dataSource",
    "price_reference_year",
    "price",
]

EXPECTED_DTYPES: dict[str, str] = {
    # Pandas typically represents string columns as dtype=object.
    "brand": "object",
    "model": "object",
    "vehicleType": "object",
    "power": "float64",
    "gearbox": "object",
    "kilometer": "float64",
    "fuelType": "object",
    "yearOfRegistration": "int64",
    "seller": "str",
    "dataSource": "str",
    "price_reference_year": "int64",
    "price": "int64",
}

RANGE_RULES: dict[str, dict[str, float]] = {
    "price": {"min": 500, "max": 5_000_000},
    "power": {"min": 5, "max": 5_000},
    "kilometer": {"min": 0, "max": 300_000},
    "yearOfRegistration": {"min": 1900, "max": 2026},
    "price_reference_year": {"min": 2014, "max": 2026},
}

CATEGORICAL_RULES: dict[str, list[str]] = {
    "seller": ["private", "dealer"],
    "dataSource": ["kaggle", "crawled"],
    "gearbox": ["manual", "automatic", "semi-automatic"],
    "fuelType": ["gasoline", "diesel", "lpg",
                 "cng", "hybrid", "electric", "other"],
    "vehicleType": [
        "sedan",
        "compact",
        "station_wagon",
        "van",
        "convertible",
        "coupe",
        "suv",
        "other",
    ],
}

REQUIRED_COLUMNS: list[str] = [
    "price",
    "yearOfRegistration",
    "power",
    "kilometer",
    "model",
    "vehicleType",
    "fuelType",
    "gearbox",
    "dataSource",
    "price_reference_year",
]

FRESHNESS_THRESHOLD_DAYS = 365 * 2  # warn if newest record > 2 years old
EXPECTED_FREQ_DAYS = 7  # expect at least one record per 7 days gap

NUMERIC_COLS: list[str] = [
    "price",
    "power",
    "kilometer",
    "yearOfRegistration",
    "price_reference_year",
]

MEAN_BOUNDS: dict[str, tuple[float, float]] = {
    "price": (500, 30_000),
    "power": (50, 250),
    "kilometer": (10_000, 150_000),
    "yearOfRegistration": (1990, 2020),
    "price_reference_year": (2014, 2026),
}

EXPECTED_CORRELATIONS: list[tuple[str, str, float, float]] = [
    # (col_a, col_b, min_r, max_r)
    ("kilometer", "yearOfRegistration", -0.9, 0.0),
    ("yearOfRegistration", "price", 0.0, 0.9),
    ("kilometer", "price", -0.9, 0.0),
    ("power", "price", 0.0, 0.9),
]


class DataValidator:
    def __init__(self) -> None:
        self.validation_results: list[dict[str, Any]] = []

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
    ) -> dict[str, Any]:
        report = self._make_report("Schema")

        # 1a. Missing columns
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            self._fail(report, f"Missing columns: {sorted(missing)}")

        # 1b. Extra (unexpected) columns
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        if extra:
            report["issues"].append(f"Extra columns: {sorted(extra)}")

        # 1c. Column count
        report["stats"]["col_cnt"] = len(df.columns)
        report["stats"]["expected_col_cnt"] = len(EXPECTED_COLUMNS)

        # 1d. Data-type mismatches
        for col, expected_type in EXPECTED_DTYPES.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    self._fail(
                        report,
                        f"Column '{col}': expected dtype '{expected_type}', "
                        f"got '{actual_type}'",
                    )
        report["stats"]["dtype_checks"] = len(EXPECTED_DTYPES)

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 2 — Completeness
    # ------------------------------------------------------------------

    def validate_completeness(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        report = self._make_report("Completeness")

        total_rows = len(df)
        col_stats: dict[str, dict] = {}

        # 1. Check if required columns are completely missing
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                self._fail(report, f"Required column '{col}' is missing.")

        # 2. Calculate missing stats for ALL columns
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            missing_pct = missing_count / total_rows if total_rows > 0 else 0.0

            # Store stats for every column
            col_stats[col] = {
                "missing_count": missing_count,
                "missing_pct": round(missing_pct * 100, 4),
            }

            # 3. if a REQUIRED column exceeds the 5% threshold
            if col in REQUIRED_COLUMNS and missing_pct > 0.05:
                self._fail(
                    report,
                    f"Column '{col}': {missing_pct * 100:.2f}% missing "
                    f"(threshold {0.05 * 100:.0f}%)",
                )

        # 4. Calculate global completeness score across the entire DataFrame
        if total_rows > 0 and len(df.columns) > 0:
            non_null = int(df.notnull().sum().sum())
            total_cells = df.size
            report["stats"]["completeness_score_pct"] = round(
                non_null / total_cells * 100, 2
            )
        else:
            report["stats"]["completeness_score_pct"] = 0.0

        report["stats"]["column_detail"] = col_stats

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 3 — Uniqueness
    # ------------------------------------------------------------------

    def validate_uniqueness(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
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

        # 3b. Uniqueness score: unique records / total * 100
        report["stats"]["uniqueness_score_pct"] = 100.0
        if len(df):
            report["stats"]["uniqueness_score_pct"] = round(
                (len(df) - dup_rows) / len(df) * 100, 2
            )
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 4 — Validity / Accuracy
    # ------------------------------------------------------------------

    def validate_validity(self, df: pd.DataFrame) -> dict[str, Any]:
        report = self._make_report("Validity")

        range_stats: dict[str, dict] = {}
        cat_stats: dict[str, Any] = {}

        # --- Range checks ---
        for col, rules in RANGE_RULES.items():
            if col not in df.columns:
                continue
            col_clean = df[col].dropna()
            violations: dict[str, int] = {}
            line_1 = f"Validating range for column '{col}'"
            line_2 = f" with {len(col_clean)} non-null values..."
            print(line_1 + line_2)

            below = int((col_clean < rules["min"]).sum())
            if below > 0:
                violations["below_min"] = below
                fst_l = f"Column '{col}': {below} value(s)"
                scnd_l = f"below min ({rules['min']})"
                fail_line = fst_l + " " + scnd_l
                self._fail(report, fail_line)

            above = int((col_clean > rules["max"]).sum())
            if above > 0:
                violations["above_max"] = above
                fst_l = f"Column '{col}': {above} value(s)"
                scnd_l = f"above max ({rules['max']})"
                fail_line = fst_l + " " + scnd_l
                self._fail(report, fail_line)

            total_valid = len(col_clean) - sum(violations.values())
            if len(col_clean):
                accuracy_pct = round(total_valid / len(col_clean) * 100, 2)
            else:
                accuracy_pct = 100.0

            range_stats[col] = {
                "violations": violations,
                "accuracy_score_pct": accuracy_pct,
            }

        # --- Categorical checks ---
        for col, allowed in CATEGORICAL_RULES.items():
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
                fail_line = f""" Column '{col}': invalid categorical values,
                found: {sorted(invalid)}"""
                self._fail(report, fail_line)

        report["stats"]["range_checks"] = range_stats
        report["stats"]["categorical_checks"] = cat_stats

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 5 — Outlier Detection: Z-Score
    # ------------------------------------------------------------------

    def validate_outliers_zscore(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        report = self._make_report("Outliers_ZScore")
        col_stats: dict[str, dict] = {}

        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 2:
                continue
            z_scores = np.abs(stats.zscore(series))
            outlier_count = int((z_scores > 3.0).sum())
            outlier_pct = round(outlier_count / len(series) * 100, 4)
            quality_score = round(100.0 - outlier_pct, 2)
            col_stats[col] = {
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "quality_score_pct": quality_score,
            }
            if outlier_count > 0:
                self._fail(
                    report,
                    f"Column '{col}': {outlier_count} Z-Score outlier(s) "
                    f"(|z| > {3.0}) — {outlier_pct:.2f}% of values",
                )

        report["stats"]["column_detail"] = col_stats
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 5 — Outlier Detection: IQR
    # ------------------------------------------------------------------

    def validate_outliers_iqr(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        report = self._make_report("Outliers_IQR")
        col_stats: dict[str, dict] = {}

        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            Q1 = float(series.quantile(0.25))
            Q3 = float(series.quantile(0.75))
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

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
                count_line = f"Column '{col}': {outlier_count} IQR outlier(s) "
                bound_line = f"""(outside [{lower:.1f}, {upper:.1f}])"""
                pcnt_line = f""" -> {outlier_pct:.2f}%"""
                self._fail(report, count_line + bound_line + pcnt_line)

        report["stats"]["column_detail"] = col_stats
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 5 — Outlier Detection: Isolation Forest (multivariate)
    # ------------------------------------------------------------------

    def validate_outliers_isolation_forest(
        self,
        df: pd.DataFrame,
        contamination: float = 0.12,
        random_state: int = 42,
    ) -> dict[str, Any]:
        report = self._make_report("Outliers_IsolationForest")
        feature_df = df[NUMERIC_COLS].dropna()

        if len(feature_df) < 10:
            fail_line = "Not enough rows for Isolation Forest (need >= 10)."
            self._fail(report, fail_line)
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
            "features_used": NUMERIC_COLS,
            "total_rows_evaluated": len(feature_df),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "quality_score_pct": quality_score,
            "contamination_param": contamination,
        }

        if outlier_count > 0:
            cnt_line = f"Isolation Forest detected {outlier_count} outlier(s) "
            pct_line = f"({outlier_pct:.2f}%) across {NUMERIC_COLS}"
            self._fail(report, cnt_line + pct_line)

        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 7 — Distribution Profile
    # ------------------------------------------------------------------

    def validate_distribution(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        report = self._make_report("Distribution")
        col_stats: dict[str, dict] = {}

        for col in NUMERIC_COLS:
            series = df[col].dropna()
            if len(series) < 2:
                continue

            mean_val = float(series.mean())
            median_val = float(series.median())
            mode_val = float("nan")
            std_val = float(series.std())
            skew_val = float(stats.skew(series))
            kurt_val = float(stats.kurtosis(series))
            if len(series.mode()) > 0:
                mode_val = float(series.mode().iloc[0])

            # Skewness classification
            if abs(skew_val) < 0.5:
                skew_label = "Symmetric"
            elif skew_val > 0:
                skew_label = "Positive Skew (right tail)"
            else:
                skew_label = "Negative Skew (left tail)"

            # Kurtosis classification (excess kurtosis)
            if abs(kurt_val) < 0.5:
                kurt_label = "Mesokurtic (normal)"
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
            lo, hi = MEAN_BOUNDS[col]
            col_stat["mean_bounds"] = (lo, hi)
            if not (lo <= mean_val <= hi):
                fail_line = f"""Column '{col}': mean={mean_val:.2f}
                    outside expected bounds [{lo}, {hi}]"""
                self._fail(report, fail_line)
            # KS test against normal distribution
            ks_stat, ks_p = stats.kstest(series, "norm",
                                         args=(mean_val, std_val))
            col_stat["ks_statistic"] = round(ks_stat, 4)
            col_stat["ks_pvalue"] = round(ks_p, 4)
            if ks_p < 0.05:
                col_stat["ks_normality"] = "Reject normality (p < 0.05)"
                ks_line = f"Column '{col}': KS test p={ks_p:.4f} < 0.05"
                fail_line = "(distribution differs from normal)"
                self._fail(report, ks_line + fail_line)
            else:
                col_stat[
                    "ks_normality"
                ] = """Fail to reject normality
                (p >= 0.05)"""

            col_stats[col] = col_stat

        report["stats"]["column_detail"] = col_stats
        return self._store(report)

    # ------------------------------------------------------------------
    # Dim 8 — Relationships
    # ------------------------------------------------------------------

    def validate_relationships(self, df: pd.DataFrame) -> dict[str, Any]:
        report = self._make_report("Relationships")
        feature_df = df[NUMERIC_COLS].dropna()

        pearson_matrix = feature_df.corr(method="pearson").round(4)
        spearman_matrix = feature_df.corr(method="spearman").round(4)

        report["stats"]["pearson_matrix"] = pearson_matrix.to_dict()
        report["stats"]["spearman_matrix"] = spearman_matrix.to_dict()
        report["stats"]["columns_used"] = NUMERIC_COLS

        primary_matrix = spearman_matrix

        # Check expected correlation bounds
        for col_a, col_b, min_r, max_r in EXPECTED_CORRELATIONS:
            r = float(primary_matrix.loc[col_a, col_b])
            if min_r <= r <= max_r:
                status = "within bounds"
            else:
                status = "OUT OF BOUNDS"
            report["stats"].setdefault("expected_correlation_checks", {})[
                f"{col_a}_vs_{col_b}"
            ] = {
                "correlation": r,
                "expected_range": (min_r, max_r),
                "status": status,
                "method": "spearman",
            }
            if status == "OUT OF BOUNDS":
                fail_line = f"""Correlation '{col_a}' vs '{col_b}': r={r:.3f}
                not in expected range [{min_r}, {max_r}]
                (method={"spearman"})"""
                self._fail(report, fail_line)

        return self._store(report)

    # -------------------------------
    # Report generation
    # -------------------------------

    def generate_report(
        self, output_prefix: str = "validation_report"
    ) -> dict[str, Any]:
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r["passed"])
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0.0

        output_lines = []
        output_lines.append("\n" + "=" * 65)
        output_lines.append("  DATA VALIDATION REPORT")
        output_lines.append("=" * 65)
        output_lines.append(f"  Total checks : {total}")
        output_lines.append(f"  Passed       : {passed}")
        output_lines.append(f"  Failed       : {failed}")
        output_lines.append(f"  Success rate : {success_rate:.1f}%")
        output_lines.append("=" * 65)

        for result in self.validation_results:
            status = "PASS" if result["passed"] else "FAIL"
            marker = "✅" if result["passed"] else "❌"
            status_l = f"\n  [{status}] {marker}"
            ckeck_l = f"{result['check_type']} Check"
            output_lines.append(status_l + " " + ckeck_l)
            is_completness = result["check_type"] == "Completeness"
            if is_completness and "column_detail" in result["stats"]:
                all_score = result["stats"].get("completeness_score_pct", 0)
                output_lines.append(f" 📊 Overall Completeness: {all_score}%")
                output_lines.append(" 📊 Column-level Completeness:")
                for c, s in result["stats"]["column_detail"].items():
                    miss = s.get("missing_pct", 0.0)
                    comp = s.get("completeness_pct", 100.0 - miss)
                    complete_l = f" - {c}: {comp:.2f}% complete"
                    missing_l = f" ({miss:.2f}% missing)"
                    output_lines.append(complete_l + missing_l)

            # Format distribution statistics cleanly
            is_distribution = result["check_type"] == "Distribution"
            if is_distribution and "column_detail" in result["stats"]:
                for c, s in result["stats"]["column_detail"].items():
                    output_lines.append(f"📊 Stats for {c}:")
                    mean_l = f" Mean: {s.get('mean')}"
                    median_l = f"Median: {s.get('median')}"
                    mode_l = f"Mode: {s.get('mode')}"
                    std_l = f"Std: {s.get('std')}"
                    fst_line = (
                        mean_l + " | " + median_l + " | "
                        + mode_l + " | " + std_l
                    )
                    q1_l = f"Q1: {s.get('Q1')}"
                    q3_l = f"Q3: {s.get('Q3')}"
                    min_l = f"Min: {s.get('min')}"
                    max_l = f"Max: {s.get('max')}"
                    scnd_line = (
                        q1_l + " | " + q3_l + " | "
                        + min_l + " | " + max_l
                    )
                    skew_l = f"Skewness: {s.get('skewness')}"
                    skw_lbl_l = f"({s.get('skewness_label')})"
                    thrd_line = skew_l + " " + skw_lbl_l
                    kurt_l = f"Kurtosis: {s.get('kurtosis_excess')}"
                    kurt_lbl_l = f"({s.get('kurtosis_label')})"
                    frth_line = kurt_l + " " + kurt_lbl_l
                    output_lines.append(fst_line)
                    output_lines.append(scnd_line)
                    output_lines.append(thrd_line)
                    output_lines.append(frth_line)

            for issue in result["issues"]:
                output_lines.append(f"       ⚠️  {issue}")

        output_lines.append("\n" + "=" * 65 + "\n")
        report_text = "\n".join(output_lines)

        # Print to console
        print(report_text)

        # Build dictionary to return and export as JSON
        report_dict = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "details": self.validation_results,
        }

        # Save to TXT
        txt_path = Path(f"{output_prefix}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report summary written to: {txt_path.absolute()}")

        # Save to JSON
        json_path = Path(f"{output_prefix}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)
        print(f"Detailed JSON stats written to: {json_path.absolute()}")

        return report_dict

    def run_all(self, df: pd.DataFrame, output_file: str) -> dict[str, Any]:
        self.validate_schema(df)
        self.validate_completeness(df)
        self.validate_uniqueness(df)
        self.validate_validity(df)
        self.validate_outliers_zscore(df)
        self.validate_outliers_iqr(df)
        self.validate_outliers_isolation_forest(df)
        self.validate_distribution(df)
        self.validate_relationships(df)
        return self.generate_report(output_prefix=output_file)
