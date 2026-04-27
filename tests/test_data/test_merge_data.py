import pandas as pd
import pytest
from src.data.merge_data import transform_data

def test_transform_data_shape(sample_kaggle, sample_scraped):
    df_merged = transform_data(sample_kaggle, sample_scraped)
    
    assert len(df_merged) == 3  # 2 Kaggle + 1 scraped
    
    expected_cols = [
        'brand', 'fuelType', 'kilometer', 'yearOfRegistration', 
        'vehicleType', 'seller', 'power', 'model', 'price', 'data_source'
    ]
    assert list(df_merged.columns) == expected_cols

def test_transform_data_cleaning_logic(sample_kaggle, sample_scraped):
    df_merged = transform_data(sample_kaggle, sample_scraped)
    
    # Test text formatting (lowercase, no symbols)
    assert df_merged.loc[0, 'brand'] == 'volkswagen'
    assert df_merged.loc[1, 'brand'] == 'bmw'
    assert df_merged.loc[2, 'brand'] == 'peugeot'
    
    assert df_merged.loc[1, 'model'] == 'x5'
    
    # Test translations
    assert df_merged.loc[0, 'fuelType'] == 'gasoline'
    assert df_merged.loc[1, 'vehicleType'] == 'suv'
    assert df_merged.loc[0, 'seller'] == 'private'

    # Test regex power extraction from scraped data
    assert df_merged.loc[2, 'power'] == 82.0
    
    # Test year extraction
    assert df_merged.loc[2, 'yearOfRegistration'] == 2017.0