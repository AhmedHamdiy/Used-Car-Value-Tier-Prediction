from src.data.merge_data import transform_data


def test_transform_data_shape(sample_kaggle, sample_scraped):
    df_merged = transform_data(sample_kaggle, sample_scraped)

    assert len(df_merged) == 41  # 40 Kaggle + 1 scraped

    expected_cols = [
        'brand', 'model', 'vehicleType', 'power',
        'gearbox', 'kilometer', 'fuelType',
        'yearOfRegistration', 'seller', 'price'
    ]
    assert list(df_merged.columns) == expected_cols


def test_transform_data_cleaning_logic(sample_kaggle, sample_scraped):
    df_merged = transform_data(sample_kaggle, sample_scraped)

    # Test text formatting (lowercase, no symbols)
    assert df_merged.loc[0, 'brand'] == 'volkswagen'
    assert df_merged.loc[1, 'brand'] == 'volkswagen'
    assert df_merged.loc[40, 'brand'] == 'peugeot'

    assert df_merged.loc[1, 'model'] == 'golf'

    # Test translations
    assert df_merged.loc[0, 'fuelType'] == 'gasoline'
    assert df_merged.loc[1, 'vehicleType'] == 'sedan'
    assert df_merged.loc[0, 'seller'] == 'private'

    # Test regex power extraction from scraped data
    assert df_merged.loc[40, 'power'] == 82.0

    # Test year extraction
    assert df_merged.loc[40, 'yearOfRegistration'] == 2017.0
