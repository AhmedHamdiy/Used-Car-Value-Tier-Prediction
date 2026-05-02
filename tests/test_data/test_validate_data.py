import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Assuming your class is in src/data/validate_data.py or similar
from src.data.validate_data import DataValidator

@pytest.fixture
def clean_dataframe():
    """
    Generates a baseline DataFrame of 20 rows that perfectly complies with
    all DataValidator rules, including isolation forest size requirements,
    mean bounds, categorical rules, and correlation constraints.
    """
    np.random.seed(42)
    n = 20

    # Establish linear relationships to pass the expected spearman bounds
    # year increases -> kilometer decreases -> price increases -> power increases
    years = np.linspace(2010, 2020, n).astype(int)
    kms = np.linspace(140000, 20000, n)
    prices = np.linspace(5000, 25000, n).astype(int)
    power = np.linspace(80, 200, n)

    df = pd.DataFrame({
        "brand": ["volkswagen"] * n,
        "model": ["golf"] * n,
        "vehicleType": ["sedan"] * n,
        "power": power,
        "gearbox": ["manual"] * n,
        "kilometer": kms,
        "fuelType": ["gasoline"] * n,
        "yearOfRegistration": years,
        "seller": ["private"] * n,
        "dataSource": ["kaggle"] * n,
        "price_reference_year": [2020] * n,
        "price": prices
    })

    # Ensure correct dtypes to pass validate_schema
    df["yearOfRegistration"] = df["yearOfRegistration"].astype("int64")
    df["price_reference_year"] = df["price_reference_year"].astype("int64")
    df["price"] = df["price"].astype("int64")
    df["kilometer"] = df["kilometer"].astype("float64")
    df["power"] = df["power"].astype("float64")

    return df


@pytest.fixture
def validator():
    return DataValidator()


# ------------------------------------------------------------------
# Dim 1 — Schema / Consistency
# ------------------------------------------------------------------

def test_validate_schema_success(validator, clean_dataframe):
    """Test schema validation passes with expected columns and dtypes."""
    report = validator.validate_schema(clean_dataframe)
    assert report["passed"] is True
    assert len(report["issues"]) == 0

def test_validate_schema_missing_column(validator, clean_dataframe):
    """Test schema fails when an expected column is missing."""
    df_missing = clean_dataframe.drop(columns=["price"])
    report = validator.validate_schema(df_missing)

    assert report["passed"] is False
    assert any("Missing columns" in issue for issue in report["issues"])

def test_validate_schema_wrong_dtype(validator, clean_dataframe):
    """Test schema fails when column types do not match EXPECTED_DTYPES."""
    df_wrong_type = clean_dataframe.copy()
    df_wrong_type["price"] = df_wrong_type["price"].astype(float) # Expected int64

    report = validator.validate_schema(df_wrong_type)
    assert report["passed"] is False
    assert any("expected dtype" in issue for issue in report["issues"])


# ------------------------------------------------------------------
# Dim 2 — Completeness
# ------------------------------------------------------------------

def test_validate_completeness_success(validator, clean_dataframe):
    """Test completeness passes when no required columns are heavily missing."""
    report = validator.validate_completeness(clean_dataframe)
    assert report["passed"] is True
    assert report["stats"]["completeness_score_pct"] == 100.0

def test_validate_completeness_threshold_exceeded(validator, clean_dataframe):
    """Test completeness fails when missing % in REQUIRED_COLUMNS > 5%."""
    df_missing = clean_dataframe.copy()
    # 2 missing values out of 20 = 10% missing, which exceeds the 0.05 threshold
    df_missing.loc[0:1, "price"] = np.nan

    report = validator.validate_completeness(df_missing)
    assert report["passed"] is False
    assert any("missing (threshold 5%)" in issue for issue in report["issues"])


# ------------------------------------------------------------------
# Dim 3 — Uniqueness
# ------------------------------------------------------------------

def test_validate_uniqueness_success(validator, clean_dataframe):
    """Test uniqueness passes when no full rows are duplicated."""
    report = validator.validate_uniqueness(clean_dataframe)
    assert report["passed"] is True
    assert report["stats"]["duplicate_rows"] == 0

def test_validate_uniqueness_duplicates(validator, clean_dataframe):
    """Test uniqueness fails and accurately counts duplicated rows."""
    df_dupes = pd.concat([clean_dataframe, clean_dataframe.iloc[[0, 1]]])
    report = validator.validate_uniqueness(df_dupes)

    assert report["passed"] is False
    assert report["stats"]["duplicate_rows"] == 2


# ------------------------------------------------------------------
# Dim 4 — Validity / Accuracy
# ------------------------------------------------------------------

def test_validate_validity_success(validator, clean_dataframe):
    """Test valid ranges and categorical variables pass."""
    report = validator.validate_validity(clean_dataframe)
    assert report["passed"] is True

def test_validate_validity_range_fail(validator, clean_dataframe):
    """Test validity fails when numerical boundaries are breached."""
    df_invalid = clean_dataframe.copy()
    df_invalid.loc[0, "price"] = 100  # Below min (500)
    df_invalid.loc[1, "power"] = 6000 # Above max (5000)

    report = validator.validate_validity(df_invalid)
    assert report["passed"] is False
    assert report["stats"]["range_checks"]["price"]["violations"]["below_min"] == 1
    assert report["stats"]["range_checks"]["power"]["violations"]["above_max"] == 1

def test_validate_validity_categorical_fail(validator, clean_dataframe):
    """Test validity fails with unrecognized categorical strings."""
    df_invalid = clean_dataframe.copy()
    df_invalid.loc[0, "gearbox"] = "hyper-drive" # Not in allowed list

    report = validator.validate_validity(df_invalid)
    assert report["passed"] is False
    assert "invalid categorical values" in report["issues"][0]


# ------------------------------------------------------------------
# Dim 5 — Outliers (Z-Score, IQR, Isolation Forest)
# ------------------------------------------------------------------

def test_validate_outliers_zscore(validator, clean_dataframe):
    """Test z-score flags values where |z| > 3.0."""
    df_outlier = clean_dataframe.copy()
    df_outlier.loc[0, "price"] = 5_000_000 # Extreme outlier

    report = validator.validate_outliers_zscore(df_outlier)
    assert report["passed"] is False
    assert report["stats"]["column_detail"]["price"]["outlier_count"] > 0

def test_validate_outliers_iqr(validator, clean_dataframe):
    """Test IQR flags values outside Q1-1.5*IQR to Q3+1.5*IQR bounds."""
    df_outlier = clean_dataframe.copy()
    df_outlier.loc[0, "kilometer"] = 800_000

    report = validator.validate_outliers_iqr(df_outlier)
    assert report["passed"] is False
    assert report["stats"]["column_detail"]["kilometer"]["outlier_count"] > 0

def test_validate_outliers_isolation_forest(validator, clean_dataframe):
    """Test Isolation Forest successfully evaluates multivariate distributions."""
    df_outlier = clean_dataframe.copy()
    # Inject an odd multivariate combination
    df_outlier.loc[0, "price"] = 45000
    df_outlier.loc[0, "kilometer"] = 250000

    report = validator.validate_outliers_isolation_forest(df_outlier, contamination=0.1)
    assert report["passed"] is False
    assert report["stats"]["outlier_count"] > 0


# ------------------------------------------------------------------
# Dim 7 & 8 — Distribution and Relationships
# ------------------------------------------------------------------

def test_validate_distribution_mean_bounds(validator, clean_dataframe):
    """Test distribution check fails if the column mean is outside EXPECTED_BOUNDS."""
    df_dist = clean_dataframe.copy()
    df_dist["price"] = df_dist["price"] + 50_000 # Shifts mean well above 30,000 max

    report = validator.validate_distribution(df_dist)
    assert report["passed"] is False
    assert any("outside expected bounds" in issue for issue in report["issues"])

def test_validate_relationships(validator, clean_dataframe):
    """Test spearman correlation constraints pass against the baseline trend."""
    report = validator.validate_relationships(clean_dataframe)
    assert report["passed"] is True

    # Inverse correlation check: break expected bounds
    df_broken = clean_dataframe.copy()
    df_broken["kilometer"] = df_broken["price"] # Forces positive correlation between km and price
    report_broken = validator.validate_relationships(df_broken)
    assert report_broken["passed"] is False


# ------------------------------------------------------------------
# Run All / Report Generation
# ------------------------------------------------------------------

def test_run_all_and_report_generation(validator, clean_dataframe, tmp_path):
    """Test comprehensive suite execution and JSON/TXT output generation."""
    output_prefix = tmp_path / "test_report"

    # Introduce one flaw to ensure both passed and failed states exist in the report
    clean_dataframe.loc[0, "price"] = -100
    summary = validator.run_all(clean_dataframe, str(output_prefix))
    assert summary["total"] == 9 # 9 distinct checks in run_all
    assert summary["failed"] > 0
    assert summary["passed"] > 0

    # Validate files were physically written
    assert Path(f"{output_prefix}.txt").exists()
    assert Path(f"{output_prefix}.json").exists()

    with open(f"{output_prefix}.json", "r") as f:
        data = json.load(f)
        assert "success_rate" in data
        assert len(data["details"]) == 9