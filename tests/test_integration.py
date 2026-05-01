# import os
# import pandas as pd
# from unittest.mock import patch
# from src.data.merge_data import merge_datasets
# from src.data.validate_data import DataValidator


# # ADDED autospec=True here so the mock captures the DataFrame instance
# @patch('src.data.merge_data.pd.DataFrame.to_csv', autospec=True)
# @patch('src.data.merge_data.pd.read_csv')
# def test_merge_and_validate_integration(mock_read_csv, mock_to_csv, sample_kaggle, sample_scraped, tmp_path):
#     # 1. Mock the reading of the raw datasets
#     def side_effect(filepath, **kwargs):
#         if 'autos.csv' in filepath:
#             return sample_kaggle
#         elif 'cars_dataset.csv' in filepath:
#             return sample_scraped
#         return pd.DataFrame()
        
#     mock_read_csv.side_effect = side_effect
    
#     # 2. Execute the merge function
#     merge_datasets()
    
#     # Assertions for merge
#     assert mock_read_csv.call_count == 2
#     assert mock_to_csv.call_count == 1
    
#     # 3. Extract the merged DataFrame that was passed to to_csv
#     merged_df = mock_to_csv.call_args[0][0]

#     # Verify we successfully intercepted the DataFrame
#     assert isinstance(merged_df, pd.DataFrame)

#     # 4. Run the validation process on the intercepted DataFrame
#     validator = DataValidator()

#     # Use Pytest's tmp_path to avoid creating real report files during tests
#     test_report_prefix = str(tmp_path / "test_validation_report")
#     report = validator.run_all(merged_df, output_file=test_report_prefix)

#     # Assertions for validation
#     assert isinstance(report, dict)
#     assert "total" in report
#     assert "passed" in report
#     assert "failed" in report
#     assert report["total"] == 9  # Verify all 9 dimension checks ran

#     # Verify the report files were generated correctly in the temp directory
#     assert os.path.exists(f"{test_report_prefix}.txt")
#     assert os.path.exists(f"{test_report_prefix}.json")
