import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data.clean_data import (
    replace_placeholders,
    clean_brand,
    clean_model,
    clean_vehicle_type,
    clean_fuel_type,
    clean_seller,
    clean_gearbox,
    validate_schema,
    remove_invalid_rows,
    drop_duplicates,
    impute_categoricals,
    cap_outliers_iqr,
    cap_outliers_fixed,
    DataCleaner,
    SCHEMA,
)


# =====================================================================
# Unit Tests: Helper & String Cleaning Functions
# =====================================================================

def test_replace_placeholders():
    """Test that explicit placeholders are correctly converted to NaN."""
    series = pd.Series(["valid_data", "?", "N/A", "keine_angabe", "-", " "])
    result = replace_placeholders(series)
    assert result.iloc[0] == "valid_data"
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])
    assert pd.isna(result.iloc[5])

def test_clean_brand():
    """Test alias mapping, special character removal, and brand dropping."""
    series = pd.Series(["vw", "alfa", "sonstige-autos", "mercedes_benz", np.nan])
    result = clean_brand(series)
    assert result.iloc[0] == "volkswagen"  # Alias mapped
    assert result.iloc[1] == "alfa-romeo"  # Alias mapped
    assert pd.isna(result.iloc[2])         # Dropped brand
    assert result.iloc[3] == "mercedes-benz" # Char replacement
    assert pd.isna(result.iloc[4])

def test_clean_model():
    """Test model alias mapping, pattern removal, and model dropping."""
    series = pd.Series(["käfer", "3er", "sonstige_autos", "c_klasse", np.nan])
    result = clean_model(series)
    assert result.iloc[0] == "beetle"      # Alias mapped
    assert result.iloc[1] == "3-series"    # Alias mapped
    assert pd.isna(result.iloc[2])         # Dropped model
    assert result.iloc[3] == "c-class"     # Char replacement & alias
    assert pd.isna(result.iloc[4])

def test_clean_categorical_aliases():
    """Test aliases for vehicle type, fuel type, seller, and gearbox."""
    vt = clean_vehicle_type(pd.Series(["limousine", "kombi"]))
    assert vt.iloc[0] == "sedan" and vt.iloc[1] == "station-wagon"

    ft = clean_fuel_type(pd.Series(["benzin", "elektro", "electric/gasoline"]))
    assert ft.iloc[0] == "gasoline" and ft.iloc[1] == "electric" and ft.iloc[2] == "hybrid"

    seller = clean_seller(pd.Series(["privat", "gewerblich"]))
    assert seller.iloc[0] == "private" and seller.iloc[1] == "dealer"

    gear = clean_gearbox(pd.Series(["manuell", "automatik"]))
    assert gear.iloc[0] == "manual" and gear.iloc[1] == "automatic"


# =====================================================================
# Unit Tests: Row Operations & Outliers
# =====================================================================

@pytest.fixture
def sample_numeric_df():
    return pd.DataFrame({
        "price": [1000, 400, 4_000_000, 15000],
        "yearOfRegistration": [2010, 1800, 2030, 2015],
        "kilometer": [50000, -100, 400_000, 100000],
        "power": [100, 2, 4000, 150]
    })

def test_remove_invalid_rows(sample_numeric_df):
    result = remove_invalid_rows(sample_numeric_df)
    # Only the last row is fully valid across all four columns
    assert len(result) == 2
    assert result.iloc[0]["price"] == 15000

def test_drop_duplicates():
    df = pd.DataFrame({"id": [1, 2, 2, 3], "val": ["a", "b", "b", "c"]})
    result = drop_duplicates(df)
    assert len(result) == 3
    assert result["id"].tolist() == [1, 2, 3]

def test_cap_outliers_fixed(sample_numeric_df):
    result = cap_outliers_fixed(sample_numeric_df)

    assert result["price"].max() == 3_000_000.0
    assert result["price"].min() == 500.0
    assert result["power"].max() == 3000.0
    assert result["yearOfRegistration"].min() == 1900

def test_cap_outliers_iqr():
    df = pd.DataFrame({"kilometer": [10000, 12000, 11000, 10500, 250000]})
    result = cap_outliers_iqr(df.copy(), ["kilometer"])

    # 250000 is an extreme outlier and should be clipped down to the upper IQR boundary
    assert result["kilometer"].max() < 250000
    assert result["kilometer"].max() == (
        df["kilometer"].quantile(0.75) +
        1.5 * (df["kilometer"].quantile(0.75)
        - df["kilometer"].quantile(0.25))
        )


# =====================================================================
# Unit Tests: Imputation & Schema
# =====================================================================

def test_impute_categoricals():
    df = pd.DataFrame({
        "brand": ["volkswagen", "volkswagen", "volkswagen", np.nan],
        "model": ["golf", "golf", np.nan, np.nan],
        "power": [100, 120, np.nan, 140],
        "yearOfRegistration": [2010, 2012, np.nan, 2014],
        "vehicleType": ["sedan", "sedan", np.nan, np.nan],
        "fuelType": ["gasoline", "gasoline", "gasoline", np.nan],
        "seller": ["private", "private", np.nan, np.nan],
        "gearbox": ["manual", "manual", np.nan, np.nan],
        "dataSource": ["kaggle", "kaggle", "kaggle", "kaggle"]
    })

    result = impute_categoricals(df)

    # Group-wise mode fallback
    assert result.loc[2, "model"] == "golf"

    # Global fallback for missing brand and model
    assert pd.isna(result.loc[3, "brand"])
    assert result.loc[3, "model"] == "unknown"

    # Median imputation for numeric
    assert result.loc[2, "power"] == df["power"].median()
    assert result.loc[2, "yearOfRegistration"] == df["yearOfRegistration"].median()

def test_validate_schema():
    df = pd.DataFrame({
        "price": [10000, np.nan],
        "power": [100, 2],
        "brand": ["volkswagen", "bmw"]
    })

    violations = validate_schema(df, SCHEMA)
    assert any("MISSING COLUMN" in v for v in violations)
    assert any("nulls in non-nullable column" in v for v in violations)
    assert any("below min" in v for v in violations)


# =====================================================================
# Integration Tests: End-to-End Pipeline
# =====================================================================

@pytest.fixture
def full_raw_df():
    return pd.DataFrame({
        "brand": ["vw", "alfa", "sonstige", "vw", "bmw", "bmw"],
        "model": ["käfer", "unknown", "andere", "käfer", "3er", "3er"],
        "vehicleType": ["limousine", "bus", "kombi", "limousine", "limousine", "limousine"],
        "power": [100, 200, 50, 100, 150, 150],
        "gearbox": ["manuell", "automatik", "manuell", "manuell", "automatik", "automatik"],
        "kilometer": [50000, 150000, 300000, 50000, 100000, 100000],
        "fuelType": ["benzin", "elektro", "diesel", "benzin", "benzin", "benzin"],
        "yearOfRegistration": ["2015", "2018", "1800", "2015", "2020", "2020"],
        "seller": ["privat", "gewerblich", "privat", "privat", "privat", "privat"],
        "dataSource": ["kaggle", "kaggle", "crawled", "kaggle", "kaggle", "kaggle"],
        "price_reference_year": [2016, 2016, 2016, 2016, 2016, 2016],
        "price": ["15000", "20000", "100", "15000", "25000", "25000"]
    })

@patch("clean_data.pd.DataFrame.to_csv")
@patch("clean_data.pd.read_csv")
def test_clean_data_pipeline(mock_read_csv, mock_to_csv, full_raw_df):
    mock_read_csv.return_value = full_raw_df

    # Test via the wrapper class
    cleaner = DataCleaner(use_iqr_capping=True)
    result = cleaner.run(raw_path="dummy_input.csv", output_path="dummy_output.csv")

    # Assertions based on expected pipeline behavior
    mock_read_csv.assert_called_once_with("dummy_input.csv", low_memory=False)
    mock_to_csv.assert_called_once_with("dummy_output.csv", index=False)

    # Check that invalid row (year 1800, price 100) and dropped brand ("sonstige") was removed
    assert "sonstige" not in result["brand"].values

    # Check that exact duplicate row (vw käfer & bmw 3er) was dropped
    assert len(result) == 3

    # Check that string cleaning successfully occurred
    assert result["brand"].iloc[0] == "volkswagen"
    assert result["model"].iloc[0] == "beetle"
    assert result["price"].dtype == "float64"
    assert str(result["yearOfRegistration"].dtype) == "Int64"
