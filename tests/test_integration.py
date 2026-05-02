import os
import pandas as pd
from unittest.mock import patch
from src.data.merge_data import merge_datasets
from src.data.validate_data import DataValidator
import pytest
from pathlib import Path
import json

from src.data.merge_data import transform_data
from src.data.clean_data import DataCleaner
from src.data.validate_data import DataValidator

@pytest.fixture
def raw_kaggle_data():
    """Generates synthetic Kaggle raw data with a mix of clean and dirty records."""
    return pd.DataFrame({
        "dateCrawled": ["2016-03-24 10:58:45", "2016-03-24 10:58:45", "2016-03-14 12:52:21"],
        "name": ["Golf_1.4", "A4_1.8", "Bad_Car"],
        "seller": ["privat", "gewerblich", "privat"],
        "offerType": ["Angebot"] * 3,
        "price": [5000.0, 15000.0, 10.0],  # 10.0 is an outlier (below min 500)
        "abtest": ["test", "control", "test"],
        "vehicleType": ["limousine", "kombi", "andere"],
        "yearOfRegistration": [2010, 2015, 1800], # 1800 is an outlier
        "gearbox": ["manuell", "automatik", "manuell"],
        "powerPS": [80.0, 150.0, 9000.0], # 9000 is an outlier
        "model": ["golf", "a4", "sonstige"], # 'sonstige' should be dropped
        "kilometer": [100000.0, 50000.0, 500000.0], # 500k is an outlier
        "monthOfRegistration": [10, 5, 1],
        "fuelType": ["benzin", "diesel", "elektro"],
        "brand": ["volkswagen", "audi", "sonstige_autos"], # should be dropped
        "notRepairedDamage": ["nein", "nein", "ja"],
        "dateCreated": ["2016-03-24", "2016-03-24", "2016-03-14"],
        "nrOfPictures": [0, 0, 0],
        "postalCode": [70435, 80331, 10115],
        "lastSeen": ["2016-04-07", "2016-04-07", "2016-03-15"],
    })

@pytest.fixture
def raw_scraped_data():
    """Generates synthetic scraped raw data."""
    return pd.DataFrame({
        'brand': ['BMW', 'Mercedes-Benz'],
        'fuelType': ['Gasoline', 'Electric/Gasoline'],
        'mileage': ['85000', '15000'],
        'year': ['10/2018', '05/2022'],
        'vehicleType': ['Sedan', 'SUV/Off-Road/Pick-Up'],
        'seller': ['Dealer', 'Private'],
        'power': ['110 kW (150 hp)', '220 kW (299 hp)'],
        'model': ['3er', 'c-klasse'],
        'price': [22000.0, 45000.0],
        'gearbox': ['Automatic', 'Automatic']
    })

def test_full_pipeline_integration(raw_kaggle_data, raw_scraped_data, tmp_path):

    # Setup temporary file paths for the pipeline handoffs
    merged_path = tmp_path / "merged_data.csv"
    cleaned_path = tmp_path / "cleaned_data.csv"
    report_prefix = tmp_path / "validation_report"

    # ==========================================
    # STAGE 1: MERGE DATA
    # ==========================================
    merged_df = transform_data(raw_kaggle_data, raw_scraped_data)

    # Verify the merge generated the correct columns
    assert "dataSource" in merged_df.columns
    assert "price_reference_year" in merged_df.columns
    assert len(merged_df) == 5  # 3 Kaggle + 2 Scraped

    # Save to disk as the cleaner expects a file path
    merged_df.to_csv(merged_path, index=False)
    assert merged_path.exists()

    # ==========================================
    # STAGE 2: CLEAN DATA
    # ==========================================
    cleaner = DataCleaner(use_iqr_capping=True)
    cleaned_df = cleaner.run(raw_path=merged_path, output_path=cleaned_path)

    # Verify cleaning operations occurred
    assert cleaned_path.exists()
    assert len(cleaned_df) < len(merged_df) # The "Bad_Car" from Kaggle should have been dropped

    # Verify string standardizations happened
    brands = cleaned_df["brand"].unique()
    assert "sonstige_autos" not in brands
    assert "bmw" in brands # Should be lowercased

    # Verify outliers were handled (e.g., max price capped, bad years removed)
    assert cleaned_df["price"].min() >= 500
    assert cleaned_df["yearOfRegistration"].min() >= 1900

    # ==========================================
    # STAGE 3: VALIDATE DATA
    # ==========================================
    validator = DataValidator()
    # Pass the cleaned dataframe into the validator
    validator.run_all(cleaned_df, str(report_prefix))

    # Verify the report was generated
    assert Path(f"{report_prefix}.json").exists()

    # Load the JSON report to verify pipeline success
    with open(f"{report_prefix}.json", "r") as f:
        report_data = json.load(f)

    # The cleaned data should pass the critical foundational checks
    details = {check["check_type"]: check["passed"] for check in report_data["details"]}

    # Assert that the data strictly conforms to the schema and validity rules post-cleaning
    assert details.get("Schema") is True, "Pipeline failed: Schema mismatch after cleaning."
    assert details.get("Completeness") is True, "Pipeline failed: Required columns missing."
    assert details.get("Validity") is True, "Pipeline failed: Out-of-bounds or invalid categorical data persisted."
