from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ========================= CONFIGURATION =========================

# Missing value placeholders (implicit)
PLACEHOLDERS: set[str] = {
    "",
    " ",
    "?",
    "N/A",
    "n/a",
    "NA",
    "na",
    "null",
    "NULL",
    "None",
    "none",
    "unknown",
    "Unknown",
    "-",
    "--",
    "keine_angabe",
    "andere",
    "sonstige",
    "nan",
}

# Mapping dictionaries
FUEL_MAP: dict[str, str] = {
    "Gasoline": "gasoline",
    "benzin": "gasoline",
    "diesel": "diesel",
    "Electric": "electric",
    "elektro": "electric",
    "Electric/Gasoline": "hybrid",
    "Electric/Diesel": "hybrid",
    "hybrid": "hybrid",
    "lpg": "lpg",
    "cng": "cng",
    "andere": "other",
}

VEHICLE_MAP: dict[str, str] = {
    "Compact": "compact",
    "kleinwagen": "compact",
    "Station Wagon": "station_wagon",
    "kombi": "station_wagon",
    "SUV/Off-Road/Pick-Up": "suv",
    "suv": "suv",
    "Sedan": "sedan",
    "limousine": "sedan",
    "Van": "van",
    "bus": "van",
    "Convertible": "convertible",
    "cabrio": "convertible",
    "coupe": "coupe",
    "Other": "other",
    "andere": "other",
}

# brand mapping
BRAND_ALIASES: dict[str, str] = {
    "alfa": "alfa-romeo",
    "land": "land-rover",
    "aston": "aston-martin",
    "lynk": "lynk-co",
    "alpine": "alpina",
    "merc": "mercedes-benz",
    "vw": "volkswagen",
    "landrover": "land-rover",
}

BRANDS_TO_DROP: set[str] = {"sonstige-autos", "unbekannt", "keine_angabe"}

# Simplified model mapping
MODEL_ALIASES: dict[str, str] = {
    "kaefer": "beetle",
    "käfer": "beetle",
    "new-beetle": "beetle",
    "1er": "1-series",
    "2er": "2-series",
    "3er": "3-series",
    "4er": "4-series",
    "5er": "5-series",
    "6er": "6-series",
    "7er": "7-series",
    "8er": "8-series",
    "a-klasse": "a-class",
    "b-klasse": "b-class",
    "c-klasse": "c-class",
    "e-klasse": "e-class",
    "g-klasse": "g-class",
    "m-klasse": "m-class",
    "s-klasse": "s-class",
    "v-klasse": "v-class",
    "x-klasse": "x-class",
    "up!": "up",
    "e-up!": "e-up",
    "ceed": "ceed",
    "andere": "other",
    "unknown": "other",
}

MODELS_TO_DROP: set[str] = {
    "sonstige", "sonstige_autos", "keine_angabe"
    }

FUEL_ALIASES: dict[str, str] = {
    "benzin": "gasoline",
    "elektro": "electric",
    "Electric/Gasoline": "hybrid",
    "Electric/Diesel": "hybrid",
    "andere": "other",
}

VT_ALIASES: dict[str, str] = {
    "limousine": "sedan",
    "kleinwagen": "compact",
    "kombi": "station-wagon",
    "Station Wagon": "station-wagon",
    "bus": "van",
    "cabrio": "convertible",
    "SUV/Off-Road/Pick-Up": "suv",
    "andere": "other",
}

SELLER_ALIASES: dict[str, str] = {
    "privat": "private",
    "gewerblich": "dealer",
}

GEAR_ALIASES: dict[str, str] = {
    "manuell": "manual",
    "automatik": "automatic",
}

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
    "price_reference_year",
    "price",
]

# Domain constraints (based on validation report)
YEAR_RANGE: tuple[int, int] = (1900, 2026)
KM_RANGE: tuple[float, float] = (0.0, 300_000.0)
MIN_PRICE: float = 500.0
MAX_PRICE: float = 3_000_000.0
POWER_MIN: float = 5.0
POWER_MAX: float = 3000.0

# Fixed capping bounds (domain knowledge)
CAP_BOUNDS: dict[str, dict[str, float]] = {
    "power": {"lower": POWER_MIN, "upper": POWER_MAX},
    "price": {"lower": MIN_PRICE, "upper": MAX_PRICE},
    "kilometer": {"lower": KM_RANGE[0], "upper": KM_RANGE[1]},
    "yearOfRegistration": {"lower": YEAR_RANGE[0], "upper": YEAR_RANGE[1]},
}

# Schema definition for validation (tutorial golden rule)
SCHEMA: dict[str, dict[str, Any]] = {
    "price": {
        "dtype": "float64",
        "nullable": False,
        "min": MIN_PRICE,
        "max": MAX_PRICE,
    },
    "power": {
        "dtype": "float64",
        "nullable": False,
        "min": POWER_MIN,
        "max": POWER_MAX,
    },
    "kilometer": {
        "dtype": "float64",
        "nullable": False,
        "min": KM_RANGE[0],
        "max": KM_RANGE[1],
    },
    "yearOfRegistration": {
        "dtype": "int64",
        "nullable": False,
        "min": YEAR_RANGE[0],
        "max": YEAR_RANGE[1],
    },
    "brand": {"dtype": "object", "nullable": False},
    "model": {"dtype": "object", "nullable": True},
    "vehicleType": {"dtype": "object", "nullable": True},
    "gearbox": {"dtype": "object", "nullable": True},
    "fuelType": {"dtype": "object", "nullable": True},
    "seller": {"dtype": "object", "nullable": True},
    "dataSource": {"dtype": "object", "nullable": True},
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


def clean_vehicle_type(series: pd.Series) -> pd.Series:
    """Normalise vehicle types using aliases."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_ ]", "-", s)
        return VT_ALIASES.get(s, s)

    return series.apply(_clean_one)


def clean_fuel_type(series: pd.Series) -> pd.Series:
    """Normalise fuel types using aliases."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_ ]", "-", s)
        return FUEL_ALIASES.get(s, s)

    return series.apply(_clean_one)


def clean_seller(series: pd.Series) -> pd.Series:
    """Normalise seller types using aliases."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_ ]", "-", s)
        return SELLER_ALIASES.get(s, s)

    return series.apply(_clean_one)


def clean_gearbox(series: pd.Series) -> pd.Series:
    """Normalise gearbox types using aliases."""

    def _clean_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        s = re.sub(r"[-_ ]", "-", s)
        return GEAR_ALIASES.get(s, s)

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


def _log_step(msg: str) -> None:
    print(f"  #️⃣ {msg}")


def _log_ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def _log_warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


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
    df["yearOfRegistration"] = (
        df["yearOfRegistration"].fillna(-1).astype("int64").replace(-1, np.nan)
    )

    # String columns – strip whitespace, lower-case, replace placeholders
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str
                .strip().str.lower()
                .replace("nan", np.nan))
            df[col] = replace_placeholders(df[col])

    _log_ok("Schema coercion and placeholder replacement complete.")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid values based on domain constraints."""
    _log_step("Removing invalid rows …")
    before = len(df)

    # price
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)]

    # year
    year_valid = (
        df["yearOfRegistration"].notna()
        & (df["yearOfRegistration"] >= YEAR_RANGE[0])
        & (df["yearOfRegistration"] <= YEAR_RANGE[1])
    )
    df = df[year_valid]

    # kilometer
    km_valid = (
        df["kilometer"].notna()
        & (df["kilometer"] >= KM_RANGE[0])
        & (df["kilometer"] <= KM_RANGE[1])
    )
    df = df[km_valid]

    # power
    power_valid = (
        df["power"].notna() &
        (df["power"] >= POWER_MIN) &
        (df["power"] <= POWER_MAX)
    )
    df = df[power_valid]

    after = len(df)
    removed = before - after
    _log_ok(f"Removed {removed:,} invalid rows. Remaining: {after:,}.")
    return df.reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    _log_step("Dropping full‑row duplicates …")
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
    df["vehicleType"] = clean_vehicle_type(df["vehicleType"])
    df["fuelType"] = clean_fuel_type(df["fuelType"])
    df["seller"] = clean_seller(df["seller"])
    df["gearbox"] = clean_gearbox(df["gearbox"])

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
        _log_ok(
            f"'{col}': capped {capped:,} " + values
        )
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

    # Step 3 – Drop exact duplicates
    df = drop_duplicates(df)

    # Step 4 – Impute missing categoricals (also cleans brand/model)
    df = impute_categoricals(df)

    # Step 5 – Outlier handling
    df = cap_outliers_fixed(df)

    if use_iqr_capping:
        df = cap_outliers_iqr(df, ["power", "kilometer", "yearOfRegistration"])

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
