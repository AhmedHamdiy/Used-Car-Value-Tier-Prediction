import pytest
import pandas as pd
from unittest.mock import patch, mock_open

_FAKE_CONFIG = {
    "data": {
        "kaggle_raw_data_path": "data/kaggle.csv",
        "scraped_raw_data_path": "data/scraped.csv",
        "merged_data_path": "data/merged.csv",
    }
}

with (
    patch("builtins.open", mock_open()),
    patch("tomllib.load", return_value=_FAKE_CONFIG),
):
    from src.data.merge_data import (
        CPI_USED_CARS,
        REFERENCE_YEAR,
        TARGET_COLS,
        normalize_price,
        extract_year,
        price_tier,
        _transform_kaggle,
        _transform_crawled,
        transform_data,
        merge_datasets,
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _make_kaggle_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid Kaggle-style DataFrame."""
    base = {
        "brand": ["BMW"],
        "model": ["3er"],
        "vehicleType": ["limousine"],
        "powerPS": [150],
        "gearbox": ["manuell"],
        "kilometer": [50000],
        "fuelType": ["benzin"],
        "yearOfRegistration": [2018],
        "seller": ["privat"],
        "price": [12000.0],
        "dateCrawled": ["2022-06-15 10:00:00"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_crawled_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid crawled/scraped-style DataFrame."""
    base = {
        "brand": ["Toyota"],
        "model": ["Corolla"],
        "vehicleType": ["sedan"],
        "power": ["110 kW (150 hp)"],
        "gearbox": ["automatic"],
        "mileage": [80000],
        "fuelType": ["petrol"],
        "year": ["2020"],
        "seller": ["dealer"],
        "price": [15000.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ===========================================================================
# normalize_price
# ===========================================================================

class TestNormalizePrice:

    def test_same_year_as_reference_returns_original_price(self):
        price = normalize_price(10_000.0, REFERENCE_YEAR)
        assert price == pytest.approx(10_000.0)

    def test_older_year_inflates_price(self):
        """Price from 2020 should be inflated relative to 2026."""
        price = normalize_price(10_000.0, 2020)
        expected = 10_000.0 * (
            CPI_USED_CARS[REFERENCE_YEAR] / CPI_USED_CARS[2020]
        )
        assert price == pytest.approx(expected)

    def test_year_with_lower_cpi_inflates_price(self):
        """2014 has the lowest CPI; price should be highest after norm."""
        price_2014 = normalize_price(10_000.0, 2014)
        price_2020 = normalize_price(10_000.0, 2020)
        assert price_2014 > price_2020

    def test_invalid_year_raises_value_error(self):
        with pytest.raises(ValueError, match="No CPI entry for year"):
            normalize_price(10_000.0, 1999)

    def test_zero_price_returns_zero(self):
        assert normalize_price(0.0, 2022) == pytest.approx(0.0)

    def test_all_supported_years_do_not_raise(self):
        for year in CPI_USED_CARS:
            result = normalize_price(1_000.0, year)
            assert result > 0

    def test_output_is_float(self):
        result = normalize_price(5_000.0, 2021)
        assert isinstance(result, float)

    @pytest.mark.parametrize("year,cpi", list(CPI_USED_CARS.items()))
    def test_formula_matches_for_each_year(self, year, cpi):
        price = 8_000.0
        expected = price * (CPI_USED_CARS[REFERENCE_YEAR] / cpi)
        assert normalize_price(price, year) == pytest.approx(expected)


# ===========================================================================
# extract_year
# ===========================================================================

class TestExtractYear:

    def test_four_digit_year_string(self):
        assert extract_year("2019") == 2019

    def test_year_embedded_in_date_string(self):
        assert extract_year("2021-03-15") == 2021

    def test_integer_year(self):
        assert extract_year(2020) == 2020

    def test_none_returns_none(self):
        assert extract_year(None) is None

    def test_nan_returns_none(self):
        assert extract_year(float("nan")) is None

    def test_pandas_nat_returns_none(self):
        assert extract_year(pd.NaT) is None

    def test_returns_first_four_digit_sequence(self):
        # "12" is not 4 digits; "2023" is
        assert extract_year("ID-2023-extra") == 2023

    def test_float_with_year(self):
        assert extract_year(2018.0) == 2018


# ===========================================================================
# price_tier
# ===========================================================================

class TestPriceTier:

    @pytest.mark.parametrize("price", [0.0, 1.0, 4999.99])
    def test_budget_tier(self, price):
        assert price_tier(price) == "budget"

    @pytest.mark.parametrize("price", [5000.0, 10000.0, 14999.99])
    def test_mid_range_tier(self, price):
        assert price_tier(price) == "mid-range"

    @pytest.mark.parametrize("price", [15000.0, 30000.0, 1_000_000.0])
    def test_luxury_tier(self, price):
        assert price_tier(price) == "luxury"

    def test_boundary_5000_is_mid_range(self):
        assert price_tier(5000.0) == "mid-range"

    def test_boundary_15000_is_luxury(self):
        assert price_tier(15000.0) == "luxury"

    def test_boundary_just_below_5000_is_budget(self):
        assert price_tier(4999.99) == "budget"

    def test_boundary_just_below_15000_is_mid_range(self):
        assert price_tier(14999.99) == "mid-range"


# ===========================================================================
# _transform_kaggle
# ===========================================================================

class TestTransformKaggle:

    def test_data_source_column_set_to_kaggle(self):
        df = _make_kaggle_df()
        result = _transform_kaggle(df)
        assert (result["dataSource"] == "kaggle").all()

    def test_powerPS_renamed_to_power(self):
        df = _make_kaggle_df()
        result = _transform_kaggle(df)
        assert "power" in result.columns
        assert "powerPS" not in result.columns

    def test_price_is_normalised(self):
        raw_price = 10_000.0
        year = 2022
        df = _make_kaggle_df(
            price=[raw_price],
            dateCrawled=[f"{year}-01-01 00:00:00"],
        )
        result = _transform_kaggle(df)
        expected = normalize_price(raw_price, year)
        assert result["price"].iloc[0] == pytest.approx(expected)

    def test_price_tier_column_created(self):
        df = _make_kaggle_df()
        result = _transform_kaggle(df)
        assert "price_tier" in result.columns

    def test_original_df_is_not_mutated(self):
        df = _make_kaggle_df()
        original_cols = set(df.columns)
        _transform_kaggle(df)
        assert set(df.columns) == original_cols

    def test_year_crawled_derived_from_date_crawled(self):
        df = _make_kaggle_df(dateCrawled=["2023-11-30 08:00:00"])
        # Should not raise even when year is in CPI dict
        result = _transform_kaggle(df)
        assert result is not None

    def test_multiple_rows_all_get_data_source(self):
        df = pd.DataFrame({
            "brand": ["BMW", "Audi"],
            "model": ["3er", "A4"],
            "vehicleType": ["limousine", "limousine"],
            "powerPS": [150, 180],
            "gearbox": ["manuell", "manuell"],
            "kilometer": [50000, 60000],
            "fuelType": ["benzin", "benzin"],
            "yearOfRegistration": [2018, 2019],
            "seller": ["privat", "privat"],
            "price": [12000.0, 18000.0],
            "dateCrawled": [
                "2022-06-15 10:00:00",
                "2023-06-15 10:00:00",
            ],
        })
        result = _transform_kaggle(df)
        assert len(result) == 2
        assert (result["dataSource"] == "kaggle").all()


# ===========================================================================
# _transform_crawled
# ===========================================================================

class TestTransformCrawled:

    def test_data_source_column_set_to_crawled(self):
        df = _make_crawled_df()
        result = _transform_crawled(df)
        assert (result["dataSource"] == "crawled").all()

    def test_mileage_renamed_to_kilometer(self):
        df = _make_crawled_df()
        result = _transform_crawled(df)
        assert "kilometer" in result.columns
        assert "mileage" not in result.columns

    def test_power_extracted_from_string(self):
        df = _make_crawled_df(power=["85 kW (115 hp)"])
        result = _transform_crawled(df)
        assert result["power"].iloc[0] == pytest.approx(115.0)

    def test_year_of_registration_extracted(self):
        df = _make_crawled_df(year=["2019-06"])
        result = _transform_crawled(df)
        assert result["yearOfRegistration"].iloc[0] == pytest.approx(2019.0)

    def test_price_tier_column_created(self):
        df = _make_crawled_df()
        result = _transform_crawled(df)
        assert "price_tier" in result.columns

    def test_original_df_is_not_mutated(self):
        df = _make_crawled_df()
        original_cols = set(df.columns)
        _transform_crawled(df)
        assert set(df.columns) == original_cols

    def test_power_none_when_pattern_absent(self):
        """No '(NNN hp)' pattern → power should be NaN."""
        df = _make_crawled_df(power=["85 kW"])
        result = _transform_crawled(df)
        assert pd.isna(result["power"].iloc[0])

    def test_multiple_rows_processed(self):
        df = pd.DataFrame({
            "brand": ["Toyota", "Honda"],
            "model": ["Corolla", "Civic"],
            "vehicleType": ["sedan", "sedan"],
            "power": ["110 kW (150 hp)", "100 kW (135 hp)"],
            "gearbox": ["automatic", "manual"],
            "mileage": [80000, 90000],
            "fuelType": ["petrol", "petrol"],
            "year": ["2020", "2019"],
            "seller": ["dealer", "dealer"],
            "price": [15000.0, 13000.0],
        })
        result = _transform_crawled(df)
        assert len(result) == 2


# ===========================================================================
# transform_data
# ===========================================================================

class TestTransformData:

    def test_output_contains_only_target_cols(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        assert list(result.columns) == TARGET_COLS

    def test_row_count_is_sum_of_both_sources(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        assert len(result) == len(df_k) + len(df_c)

    def test_both_data_sources_represented(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        sources = set(result["dataSource"].unique())
        assert sources == {"kaggle", "crawled"}

    def test_index_is_reset(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        assert list(result.index) == list(range(len(result)))

    def test_price_tier_values_valid(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        valid_tiers = {"budget", "mid-range", "luxury"}
        assert set(result["price_tier"].unique()).issubset(valid_tiers)

    def test_no_extra_columns_in_output(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        assert set(result.columns) == set(TARGET_COLS)


# ===========================================================================
# merge_datasets (mocked I/O)
# ===========================================================================

class TestMergeDatasets:

    def _run_with_mocked_io(self, df_kaggle, df_crawled):
        """
        Patch pd.read_csv and pd.DataFrame.to_csv so no filesystem
        access is required. Returns the DataFrame passed to to_csv.
        """
        saved = {}

        def fake_read_csv(path, **kwargs):
            if "kaggle" in str(path):
                return df_kaggle.copy()
            return df_crawled.copy()

        def fake_to_csv(self_df, path, **kwargs):
            saved["df"] = self_df
            saved["path"] = path

        with (
            patch("src.data.merge_data.pd.read_csv", side_effect=fake_read_csv),
            patch("pandas.DataFrame.to_csv", fake_to_csv),
        ):
            merge_datasets()

        return saved

    def test_output_saved_to_correct_path(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        saved = self._run_with_mocked_io(df_k, df_c)
        assert "merged" in saved["path"]

    def test_output_has_target_columns(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        saved = self._run_with_mocked_io(df_k, df_c)
        assert list(saved["df"].columns) == TARGET_COLS

    def test_output_combines_both_sources(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()
        saved = self._run_with_mocked_io(df_k, df_c)
        sources = set(saved["df"]["dataSource"].unique())
        assert sources == {"kaggle", "crawled"}

    def test_read_csv_called_twice(self):
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df()

        call_count = {"n": 0}

        def fake_read_csv(path, **kwargs):
            call_count["n"] += 1
            if "kaggle" in str(path):
                return df_k.copy()
            return df_c.copy()

        with (
            patch("src.data.merge_data.pd.read_csv", side_effect=fake_read_csv),
            patch("pandas.DataFrame.to_csv", lambda *a, **kw: None),
        ):
            merge_datasets()

        assert call_count["n"] == 2


# ===========================================================================
# Integration: end-to-end transform with realistic data
# ===========================================================================

class TestEndToEndTransform:

    def test_budget_car_gets_correct_tier(self):
        df_k = _make_kaggle_df(price=[3000.0])
        df_c = _make_crawled_df(price=[3000.0])
        result = transform_data(df_k, df_c)
        assert (result["price_tier"] == "budget").all()

    def test_luxury_car_gets_correct_tier(self):
        df_k = _make_kaggle_df(price=[50000.0])
        df_c = _make_crawled_df(price=[50000.0])
        result = transform_data(df_k, df_c)
        assert (result["price_tier"] == "luxury").all()

    def test_kaggle_price_is_normalised_in_output(self):
        raw_price = 10_000.0
        crawl_year = 2022
        df_k = _make_kaggle_df(
            price=[raw_price],
            dateCrawled=[f"{crawl_year}-03-01 00:00:00"],
        )
        df_c = _make_crawled_df()
        result = transform_data(df_k, df_c)
        kaggle_row = result[result["dataSource"] == "kaggle"].iloc[0]
        expected = normalize_price(raw_price, crawl_year)
        assert kaggle_row["price"] == pytest.approx(expected)

    def test_crawled_row_price_unchanged_in_output(self):
        raw_price = 12_000.0
        df_k = _make_kaggle_df()
        df_c = _make_crawled_df(price=[raw_price])
        result = transform_data(df_k, df_c)
        crawled_row = result[result["dataSource"] == "crawled"].iloc[0]
        assert crawled_row["price"] == pytest.approx(raw_price)
