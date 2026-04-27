import os
import pandas as pd
from unittest.mock import patch
from src.data.merge_data import merge_datasets

@patch('src.data.merge_data.pd.DataFrame.to_csv')
@patch('src.data.merge_data.pd.read_csv')
def test_merge_datasets_integration(mock_read_csv, mock_to_csv, sample_kaggle, sample_scraped):
    # Mock the return values of pd.read_csv based on which file is requested
    def side_effect(filepath, **kwargs):
        if 'autos.csv' in filepath:
            return sample_kaggle
        elif 'cars_dataset.csv' in filepath:
            return sample_scraped
        return pd.DataFrame()
        
    mock_read_csv.side_effect = side_effect
    
    # Execute the main function
    merge_datasets()
    
    # Assert that read_csv was called twice (once for each dataset)
    assert mock_read_csv.call_count == 2
    
    # Assert that to_csv was called once to save the merged data
    assert mock_to_csv.call_count == 1
    
    # Verify the output directory was created
    assert os.path.exists('data/interim')