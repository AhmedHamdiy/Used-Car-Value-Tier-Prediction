import pandas as pd
import numpy as np
import pytest
from src.data.validate_data import DataValidator

@pytest.fixture
def clean_data():
    """A perfectly clean dataset that should pass all basic validations."""
    return pd.DataFrame({
        "price": [5000, 15000, 25000, 10000, 12000],
        "yearOfRegistration": [2010, 2015, 2020, 2012, 2018],
        "power": [100, 150, 200, 120, 180],
        "kilometer": [150000, 80000, 30000, 120000, 60000],
        "brand": ["volkswagen", "bmw", "audi", "ford", "mercedes"],
        "seller": ["private", "dealer", "private", "dealer", "private"],
        "dateCreated": pd.date_range(end=pd.Timestamp.now(), periods=5, freq="D").strftime("%Y-%m-%d")
    })

@pytest.fixture
def dirty_data(clean_data):
    """A corrupted dataset designed to trigger validation failures."""
    df = clean_data.copy()
    
    # Inject Schema error: change dtype
    df["price"] = df["price"].astype(float)
    
    # Inject Completeness error: add nulls to a required column
    df.loc[0:2, "brand"] = np.nan
    
    # Inject Uniqueness error: duplicate a row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    # Inject Validity errors: out of range and invalid category
    df.loc[3, "price"] = -500  # Below min
    df.loc[4, "seller"] = "black_market"  # Invalid categorical
    
    return df

def test_validate_schema(clean_data, dirty_data):
    validator = DataValidator()
    
    # Happy Path
    report_clean = validator.validate_schema(
        clean_data, 
        expected_columns=["price", "power"], 
        expected_dtypes={"price": "int64", "power": "int64"}
    )
    assert report_clean["passed"] is True
    
    # Sad Path
    report_dirty = validator.validate_schema(
        dirty_data, 
        expected_columns=["price", "power", "missing_col"], # Force missing column
        expected_dtypes={"price": "int64"} # Dirty data has float64
    )
    assert report_dirty["passed"] is False
    assert any("Missing columns" in issue for issue in report_dirty["issues"])
    assert any("expected dtype" in issue for issue in report_dirty["issues"])

def test_validate_completeness(clean_data, dirty_data):
    validator = DataValidator()
    
    # Happy Path
    report_clean = validator.validate_completeness(clean_data, required_columns=["brand"])
    assert report_clean["passed"] is True
    
    # Sad Path: 3 out of 6 rows in dirty_data have NaN in 'brand' (50% missing)
    report_dirty = validator.validate_completeness(
        dirty_data, 
        required_columns=["brand"], 
        max_missing_pct=0.05
    )
    assert report_dirty["passed"] is False
    assert report_dirty["stats"]["column_detail"]["brand"]["missing_count"] == 4

def test_validate_uniqueness(clean_data, dirty_data):
    validator = DataValidator()
    
    # Happy Path
    report_clean = validator.validate_uniqueness(clean_data)
    assert report_clean["passed"] is True
    
    # Sad Path
    report_dirty = validator.validate_uniqueness(dirty_data)
    assert report_dirty["passed"] is False
    assert report_dirty["stats"]["duplicate_rows"] == 1

def test_validate_validity(clean_data, dirty_data):
    validator = DataValidator()
    
    # Happy Path
    report_clean = validator.validate_validity(
        clean_data,
        range_rules={"price": {"min": 0, "max": 100000}},
        categorical_rules={"seller": ["private", "dealer"]},
        regex_rules={}
    )
    assert report_clean["passed"] is True
    
    # Sad Path
    report_dirty = validator.validate_validity(
        dirty_data,
        range_rules={"price": {"min": 0, "max": 100000}},
        categorical_rules={"seller": ["private", "dealer"]},
        regex_rules={}
    )
    assert report_dirty["passed"] is False
    assert report_dirty["stats"]["range_checks"]["price"]["violations"]["below_min"] == 1
    assert "black_market" in report_dirty["stats"]["categorical_checks"]["seller"]["invalid_values"]

def test_validate_outliers_iqr(clean_data):
    validator = DataValidator()
    
    # Inject a massive outlier into clean data
    df = clean_data.copy()
    df.loc[0, "price"] = 9999999
    
    report = validator.validate_outliers_iqr(df, columns=["price"], factor=1.5)
    assert report["passed"] is False
    assert report["stats"]["column_detail"]["price"]["outlier_count"] > 0

def test_run_all_integration(clean_data):
    """Integration test to ensure the run_all aggregator functions correctly."""
    validator = DataValidator()
    
    # We pass empty expected lists to bypass strict schema checks for this integration test
    validator.EXPECTED_COLUMNS = list(clean_data.columns)
    
    summary = validator.run_all(clean_data)
    
    assert summary["total"] > 0
    # Because clean_data doesn't have all 20 columns, some specific dimension checks 
    # (like Timeliness on 'dateCrawled') might abort or fail depending on default kwargs,
    # but the aggregator itself should return a properly formatted dictionary.
    assert "passed" in summary
    assert "failed" in summary
    assert "success_rate" in summary