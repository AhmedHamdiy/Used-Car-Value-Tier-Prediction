import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data.merge_data import (
    _transform_kaggle,
    _transform_crawled,
    transform_data,
    merge_datasets,
    TARGET_COLS,
    REFERENCE_YEAR,
    KAGGLE_DATA_PATH,
    SCRAPED_DATA_PATH,
    MERGED_DATA_PATH
)

def test_transform_kaggle_success(sample_kaggle):
    """Test that Kaggle data is properly formatted and new columns are generated."""
    result = _transform_kaggle(sample_kaggle)

    # Verify column renaming and static additions
    assert "dataSource" in result.columns
    assert all(result["dataSource"] == "kaggle")
    assert "power" in result.columns
    assert "powerPS" not in result.columns

    # Verify year extraction from dateCrawled string
    assert "price_reference_year" in result.columns
    assert result["price_reference_year"].dtype == float
    assert all(result["price_reference_year"] == 2016.0)

def test_transform_crawled_success(sample_scraped):
    """Test that scraped data regex extractions and column renaming work."""
    result = _transform_crawled(sample_scraped)

    # Verify static additions and renaming
    assert "dataSource" in result.columns
    assert all(result["dataSource"] == "crawled")
    assert "price_reference_year" in result.columns
    assert all(result["price_reference_year"] == REFERENCE_YEAR)
    assert "kilometer" in result.columns
    assert "mileage" not in result.columns

    # Verify regex extraction for power and year
    assert result["power"].iloc[0] == 82.0
    assert result["yearOfRegistration"].iloc[0] == 2017.0

def test_transform_crawled_regex_edge_cases():
    """Test resilience against missing or malformed data during regex extraction."""
    df_edge = pd.DataFrame({
        "mileage": ["1000"],
        "power": ["No hp value", np.nan],
        "year": ["InvalidDate", np.nan]
    })

    result = _transform_crawled(df_edge)

    # Non-matching or missing regex targets should default to NaN
    assert pd.isna(result["power"].iloc[0])
    assert pd.isna(result["power"].iloc[1])
    assert pd.isna(result["yearOfRegistration"].iloc[0])
    assert pd.isna(result["yearOfRegistration"].iloc[1])

def test_transform_data_integration(sample_kaggle, sample_scraped):
    """Test that both datasets are transformed, filtered to TARGET_COLS, and combined."""
    result = transform_data(sample_kaggle, sample_scraped)

    # Verify strict column alignment
    assert list(result.columns) == TARGET_COLS

    # Verify total rows equal the sum of both datasets
    assert len(result) == len(sample_kaggle) + len(sample_scraped)

    # Verify the concatenation included records from both sources
    counts = result["dataSource"].value_counts()
    assert counts["kaggle"] == len(sample_kaggle)
    assert counts["crawled"] == len(sample_scraped)

@patch("src.data.merge_data.pd.DataFrame.to_csv")
@patch("src.data.merge_data.pd.read_csv")
def test_merge_datasets(mock_read_csv, mock_to_csv, sample_kaggle, sample_scraped):
    """Test the end-to-end load, process, and save workflow via mocking."""

    # Mock read_csv to return the fixture data based on file path arguments
    def read_csv_side_effect(path, **kwargs):
        if path == KAGGLE_DATA_PATH:
            return sample_kaggle
        elif path == SCRAPED_DATA_PATH:
            return sample_scraped
        raise ValueError(f"Unexpected path requested: {path}")

    mock_read_csv.side_effect = read_csv_side_effect

    # Execute the core function
    merge_datasets()

    # Validate correct arguments were passed to pd.read_csv
    mock_read_csv.assert_any_call(KAGGLE_DATA_PATH, encoding="latin-1")
    mock_read_csv.assert_any_call(SCRAPED_DATA_PATH)

    # Validate output was routed to to_csv correctly
    mock_to_csv.assert_called_once_with(MERGED_DATA_PATH, index=False)