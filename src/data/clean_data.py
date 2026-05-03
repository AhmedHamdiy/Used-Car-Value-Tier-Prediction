from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import json

warnings.filterwarnings("ignore")


def _log_step(msg: str) -> None:
    print(f" [DEBUG] {msg}")


def _log_ok(msg: str) -> None:
    print(f"  [INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

# ========================= CONFIGURATION =========================

with open("src/data/mappings/missing_placeholders.json") as f:
    PLACEHOLDERS: set[str] = set(json.load(f))

with open("src/data/mappings/brand_aliases.json") as f:
    BRAND_ALIASES: dict[str, str] = json.load(f)

with open("src/data/mappings/model_aliases.json") as f:
    MODEL_ALIASES: dict[str, str] = json.load(f)

with open("src/data/mappings/fuel_aliases.json") as f:
    FUEL_ALIASES: dict[str, str] = json.load(f)

with open("src/data/mappings/vt_aliases.json") as f:
    VT_ALIASES: dict[str, str] = json.load(f)

with open("src/data/mappings/seller_aliases.json") as f:
    SELLER_ALIASES: dict[str, str] = json.load(f)

with open("src/data/mappings/gear_aliases.json") as f:
    GEAR_ALIASES: dict[str, str] = json.load(f)

with open("src/data/schemas/schema.json") as f:
    SCHEMA: dict[str, dict[str, Any]] = json.load(f)

BRANDS_TO_DROP: set[str] = {"sonstige-autos", "unbekannt", "keine_angabe"}

MODELS_TO_DROP: set[str] = {"sonstige", "sonstige_autos", "keine_angabe"}


# Categorical columns (after cleaning)
CATEGORICAL_COLS: list[str] = [
    "seller",
    "vehicleType",
    "gearbox",
    "model",
    "brand",
    "fuelType",
    "dataSource",
]

# Expected columns for raw data (input)
INPUT_COLS: list[str] = [
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
    "price",
    "price_tier",
]


# Fixed capping bounds (domain knowledge)
CAP_BOUNDS: dict[str, dict[str, float]] = {
    "power": {"lower": SCHEMA["power"]["min"], "upper": SCHEMA["power"]["max"]},
    "price": {"lower": SCHEMA["price"]["min"], "upper": SCHEMA["price"]["max"]},
    "kilometer": {"lower": SCHEMA["kilometer"]["min"], "upper": SCHEMA["kilometer"]["max"]},
    "yearOfRegistration": {"lower": SCHEMA["yearOfRegistration"]["min"], "upper": SCHEMA["yearOfRegistration"]["max"]},
}

# ========================= HELPER FUNCTIONS =========================


def replace_placeholders(series: pd.Series) -> pd.Series:
    """Replace common placeholder strings with NaN."""
    if series.dtype == "object":
        stripped = series.astype(str).str.strip()
        return stripped.where(~stripped.isin(PLACEHOLDERS), np.nan)
    return series


def clean_brand(series: pd.Series) -> pd.Series:
    """Normalise brand names using aliases and drop unwanted brands."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_]", "-", s)
        s = re.sub(r"[^\w\-]", "", s)
        # drop explicitly unwanted brands
        if s in BRANDS_TO_DROP:
            return np.nan
        # apply alias mapping
        return BRAND_ALIASES.get(s, s)

    return series.apply(_clean_one)


def clean_model(series: pd.Series) -> pd.Series:
    """Normalise model names using aliases and pattern removal."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_]", "-", s)
        if s in MODELS_TO_DROP:
            return np.nan
        # apply alias mapping
        s = MODEL_ALIASES.get(s, s)
        # remove any remaining non‑alphanumeric characters except hyphen
        s = re.sub(r"[^\w\-]", "", s)
        return s if s else np.nan

    return series.apply(_clean_one)


def clean_with_aliases(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """Normalise categorical values using a provided alias mapping."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_ ]", "-", s)
        return aliases.get(s, s)

    return series.apply(_clean_one)


def validate_schema(df: pd.DataFrame, schema: dict) -> list[str]:
    """Validate a DataFrame against a schema (tutorial pattern)."""
    violations = []
    for col, rules in schema.items():
        if col not in df.columns:
            violations.append(f"MISSING COLUMN: {col}")
            continue
        # Null check
        null_count = df[col].isnull().sum()
        if not rules.get("nullable", True) and null_count > 0:
            v = f"{col}: {null_count} nulls in non-nullable column"
            violations.append(v)
        # Min/Max
        if "min" in rules:
            bad = (df[col].dropna() < rules["min"]).sum()
            if bad:
                v = f"{col}: {bad} values below min={rules['min']}"
                violations.append(v)
        if "max" in rules:
            bad = (df[col].dropna() > rules["max"]).sum()
            if bad:
                v = f"{col}: {bad} values above max={rules['max']}"
                violations.append(v)
        # Allowed set (if present)
        if "allowed" in rules:
            allowed_set = set(rules["allowed"])
            na_allowed = pd.isna(allowed_set) if allowed_set else False
            data = df[col].dropna() if not na_allowed else df[col]
            bad = (~data.isin(allowed_set)).sum()
            if bad:
                v = f"{col}: {bad} values not in {rules['allowed']}"
                violations.append(v)
    return violations



# ========================= CLEANING STEPS =========================


def load_and_coerce(path: str | Path) -> pd.DataFrame:
    """Load CSV, keep only expected columns, coerce types, and clean
    strings."""
    _log_step(f"Loading data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    _log_ok(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # Keep only expected columns
    missing_cols = set(INPUT_COLS) - set(df.columns)
    if missing_cols:
        line = f"Raw file is missing required columns: {missing_cols}"
        raise ValueError(line)
    df = df[INPUT_COLS].copy()

    # Coerce numeric columns (errors → NaN)
    for col in ["power", "kilometer", "yearOfRegistration", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # yearOfRegistration -> nullable integer
    df["yearOfRegistration"] = df["yearOfRegistration"].astype("Int64")

    # String columns – strip whitespace, lower-case, replace placeholders
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = replace_placeholders(df[col])

    _log_ok("Schema coercion and placeholder replacement complete.")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid values based on domain constraints."""
    _log_step("Removing invalid rows …")
    before = len(df)

    # price
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= SCHEMA["price"]["min"]) & (df["price"] <= SCHEMA["price"]["max"])]

    # year
    year_valid = (
        df["yearOfRegistration"].notna()
        & (df["yearOfRegistration"] >= SCHEMA["yearOfRegistration"]["min"])
        & (df["yearOfRegistration"] <= SCHEMA["yearOfRegistration"]["max"])
    )
    df = df[year_valid]

    # kilometer
    km_valid = (
        df["kilometer"].notna()
        & (df["kilometer"] >= SCHEMA["kilometer"]["min"])
        & (df["kilometer"] <= SCHEMA["kilometer"]["max"])
    )
    df = df[km_valid]

    # power
    power_valid = (
        df["power"].notna()
        & (df["power"] >= SCHEMA["power"]["min"])
        & (df["power"] <= SCHEMA["power"]["max"])
    )
    df = df[power_valid]

    after = len(df)
    removed = before - after
    _log_ok(f"Removed {removed:,} invalid rows. Remaining: {after:,}.")
    return df.reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    _log_step("Dropping full-row duplicates …")
    before = len(df)
    df = df.drop_duplicates(keep="first")
    after = len(df)
    removed = before - after
    _log_ok(f"Dropped {removed:,} duplicate rows. Remaining: {after:,}.")
    return df.reset_index(drop=True)


def _mode_or_fallback(series: pd.Series, fallback: str) -> str:
    mode_vals = series.dropna().mode()
    return str(mode_vals.iloc[0]) if len(mode_vals) > 0 else fallback


def impute_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing categorical values using brand-group mode,
    then a logical global fallback."""
    _log_step("Imputing missing categorical values …")

    df["brand"] = clean_brand(df["brand"])
    df["model"] = clean_model(df["model"])
    df["vehicleType"] = clean_with_aliases(df["vehicleType"], VT_ALIASES)
    df["fuelType"] = clean_with_aliases(df["fuelType"], FUEL_ALIASES)
    df["seller"] = clean_with_aliases(df["seller"], SELLER_ALIASES)
    df["gearbox"] = clean_with_aliases(df["gearbox"], GEAR_ALIASES)

    impute_cols = [
        c for c in CATEGORICAL_COLS if c not in ("brand",) and c in df.columns
    ]

    for col in impute_cols:
        # Group-wise mode imputation per brand (if brand exists)
        if "brand" in df.columns and df["brand"].notna().any():
            brand_modes = df.groupby("brand")[col].transform(
                _mode_or_fallback, fallback="unknown"
            )
            df[col] = df[col].fillna(brand_modes)

        # Remaining NaN -> global fallback
        if df[col].isna().any():
            if col in ["model", "brand"]:
                global_fallback = "unknown"
            else:
                global_fallback = _mode_or_fallback(df[col], "unknown")

            df[col] = df[col].fillna(global_fallback)
            _log_warn(
                f"'{col}': residual NaN filled with "
                f"global fallback '{global_fallback}'."
            )

    # Numeric imputation: median for power and year
    for num_col in ["power", "yearOfRegistration"]:
        if df[num_col].isna().any():
            med = df[num_col].median()
            df[num_col] = df[num_col].fillna(med)
            _log_warn(f"'{num_col}': NaN filled with median {med:.1f}.")

    _log_ok("Categorical imputation complete.")
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cap outliers using the IQR rule."""
    _log_step("Capping outliers using IQR …")
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        upper = Q3 + 1.5 * IQR

        if col in CAP_BOUNDS and "lower" in CAP_BOUNDS[col]:
            lower = CAP_BOUNDS[col]["lower"]
        else:
            lower = 0.0

        if pd.api.types.is_integer_dtype(df[col]):
            lower = int(np.floor(lower))
            upper = int(np.ceil(upper))

        before_lower = (df[col] < lower).sum()
        before_upper = (df[col] > upper).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        total = before_lower + before_upper

        _log_ok(
            f"'{col}': capped {total:,} IQR outliers "
            f"(lower={lower:.1f}, upper={upper:.1f})."
        )
    return df


def cap_outliers_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using fixed domain bounds."""
    _log_step("Capping outliers with fixed bounds …")
    for col, bounds in CAP_BOUNDS.items():
        if col not in df.columns:
            continue
        upper = bounds.get("upper")
        lower = bounds.get("lower")
        capped = 0
        if upper is not None:
            before = (df[col] > upper).sum()
            df[col] = df[col].clip(upper=upper)
            capped += before
        if lower is not None:
            before = (df[col] < lower).sum()
            df[col] = df[col].clip(lower=lower)
            capped += before
        values = "values " f"(upper={upper}, lower={lower})."
        _log_ok(f"'{col}': capped {capped:,} " + values)
    return df


# ========================= MAIN CLEANING FUNCTION =========================


def clean_data(
    raw_path: str | Path,
    use_iqr_capping: bool = True,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  DATA CLEANING PIPELINE")
    print("=" * 65)

    # Step 1 – Load, coerce and validate schema
    df = load_and_coerce(raw_path)
    raw_shape = df.shape
    violations = validate_schema(df, SCHEMA)
    if violations:
        _log_warn(f"Schema violations before cleaning: {violations}")

    # Step 2 – Remove invalid rows
    df = remove_invalid_rows(df)

    # Step 3 – Impute missing categoricals (also cleans brand/model)
    df = impute_categoricals(df)

    # Step 4 – Outlier handling
    df = cap_outliers_fixed(df)

    if use_iqr_capping:
        df = cap_outliers_iqr(df, ["power", "kilometer", "yearOfRegistration"])

    # Step 5 – Drop exact duplicates
    df = drop_duplicates(df)

    # Step 6 – Remove Nan rows
    df = df.dropna().reset_index(drop=True)

    # Final validation
    final_violations = validate_schema(df, SCHEMA)
    if final_violations:
        _log_warn(f"Post-cleaning schema violations: {final_violations}")
    else:
        _log_ok("All schema checks passed after cleaning.")

    # Report
    print("\n" + "=" * 65)
    print("  CLEANING REPORT")
    print("=" * 65)
    print(f"  Raw dataset : {raw_shape[0]:,} rows, {raw_shape[1]} cols")
    print(f"  Cleaned dataset : {len(df):,} rows, {len(df.columns)} cols")
    print(f"  Missing values : {df.isna().sum().sum():,}")
    print("=" * 65 + "\n")

    if output_path:
        df.to_csv(output_path, index=False)
        _log_ok(f"Cleaned data saved to {output_path}")

    return df


class DataCleaner:

    def __init__(self, use_iqr_capping: bool = True):
        self.use_iqr_capping = use_iqr_capping
        self.cleaned_df: pd.DataFrame | None = None

    def run(
        self, raw_path: str | Path, output_path: str | Path | None = None
    ) -> pd.DataFrame:
        self.cleaned_df = clean_data(
            raw_path, use_iqr_capping=self.use_iqr_capping,
            output_path=output_path
        )
        return self.cleaned_df
