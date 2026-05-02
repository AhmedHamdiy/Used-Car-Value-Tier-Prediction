# from __future__ import annotations

# import numpy as np
# import pandas as pd
# import pytest

# from clean_data import (
#     BRAND_ALIASES,
#     BRANDS_TO_DROP,
#     CAP_BOUNDS,
#     KM_RANGE,
#     MAX_PRICE,
#     MIN_PRICE,
#     MODEL_ALIASES,
#     MODELS_TO_DROP,
#     PLACEHOLDERS,
#     POWER_MAX,
#     POWER_MIN,
#     SCHEMA,
#     YEAR_RANGE,
#     DataCleaner,
#     cap_outliers_fixed,
#     cap_outliers_iqr,
#     clean_brand,
#     clean_model,
#     drop_duplicates,
#     impute_categoricals,
#     remove_invalid_rows,
#     replace_placeholders,
#     validate_schema,
# )

# # ===========================================================================
# # Shared helpers / fixtures
# # ===========================================================================

# INPUT_COLS = [
#     "brand", "model", "vehicleType", "power", "gearbox",
#     "kilometer", "fuelType", "yearOfRegistration", "seller",
#     "dataSource", "price_reference_year", "price", "price_normalized",
# ]


# def _make_valid_df(n: int = 50) -> pd.DataFrame:
#     """Return a fully valid DataFrame that passes all cleaning steps."""
#     np.random.seed(42)
#     df = pd.DataFrame({
#         "brand": ["volkswagen"] * n,
#         "model": ["golf"] * n,
#         "vehicleType": ["sedan"] * n,
#         "power": np.random.uniform(60, 200, n),
#         "gearbox": ["manual"] * n,
#         "kilometer": np.random.uniform(20_000, 120_000, n),
#         "fuelType": ["gasoline"] * n,
#         "yearOfRegistration": np.random.randint(
#             2000, 2018, n
#         ).astype("int64"),
#         "seller": ["private"] * n,
#         "dataSource": ["web"] * n,
#         "price_reference_year": [2020] * n,
#         "price": np.random.randint(3_000, 20_000, n).astype("float64"),
#         "price_normalized": np.random.uniform(3_000, 20_000, n),
#     })
#     return df


# @pytest.fixture()
# def valid_df() -> pd.DataFrame:
#     return _make_valid_df()


# # ===========================================================================
# # replace_placeholders
# # ===========================================================================


# class TestReplacePlaceholders:
#     def test_replaces_known_placeholder_with_nan(self) -> None:
#         series = pd.Series(["N/A", "ok", "null", "value"])
#         result = replace_placeholders(series)
#         assert pd.isna(result.iloc[0])
#         assert result.iloc[1] == "ok"
#         assert pd.isna(result.iloc[2])

#     def test_strips_whitespace_before_checking(self) -> None:
#         series = pd.Series(["  N/A  ", " ", "valid"])
#         result = replace_placeholders(series)
#         assert pd.isna(result.iloc[0])
#         assert pd.isna(result.iloc[1])
#         assert result.iloc[2] == "valid"

#     def test_non_object_series_returned_unchanged(self) -> None:
#         series = pd.Series([1.0, 2.0, 3.0])
#         result = replace_placeholders(series)
#         pd.testing.assert_series_equal(result, series)

#     def test_all_placeholders_become_nan(self) -> None:
#         series = pd.Series(list(PLACEHOLDERS))
#         result = replace_placeholders(series)
#         assert result.isna().all()

#     def test_empty_string_becomes_nan(self) -> None:
#         series = pd.Series([""])
#         result = replace_placeholders(series)
#         assert pd.isna(result.iloc[0])

#     def test_valid_strings_are_preserved(self) -> None:
#         series = pd.Series(["BMW", "Toyota", "Ford"])
#         result = replace_placeholders(series)
#         assert list(result) == ["BMW", "Toyota", "Ford"]


# # ===========================================================================
# # clean_brand
# # ===========================================================================


# class TestCleanBrand:
#     def test_known_alias_is_mapped(self) -> None:
#         series = pd.Series(["vw"])
#         result = clean_brand(series)
#         assert result.iloc[0] == "volkswagen"

#     def test_brand_to_drop_becomes_nan(self) -> None:
#         for bad_brand in BRANDS_TO_DROP:
#             series = pd.Series([bad_brand])
#             result = clean_brand(series)
#             assert pd.isna(result.iloc[0]), (
#                 f"Expected NaN for brand '{bad_brand}'"
#             )

#     def test_nan_input_returns_nan(self) -> None:
#         series = pd.Series([np.nan])
#         result = clean_brand(series)
#         assert pd.isna(result.iloc[0])

#     def test_unknown_brand_passed_through_lowercased(self) -> None:
#         series = pd.Series(["Honda"])
#         result = clean_brand(series)
#         assert result.iloc[0] == "honda"

#     def test_all_alias_keys_are_mapped(self) -> None:
#         for alias, expected in BRAND_ALIASES.items():
#             result = clean_brand(pd.Series([alias]))
#             assert result.iloc[0] == expected, (
#                 f"Alias '{alias}' not mapped to '{expected}'"
#             )

#     def test_mixed_series(self) -> None:
#         series = pd.Series(["vw", "sonstige-autos", np.nan, "BMW"])
#         result = clean_brand(series)
#         assert result.iloc[0] == "volkswagen"
#         assert pd.isna(result.iloc[1])
#         assert pd.isna(result.iloc[2])
#         assert result.iloc[3] == "bmw"


# # ===========================================================================
# # clean_model
# # ===========================================================================


# class TestCleanModel:
#     def test_known_alias_is_mapped(self) -> None:
#         series = pd.Series(["3er"])
#         result = clean_model(series)
#         assert result.iloc[0] == "3-series"

#     def test_model_to_drop_becomes_nan(self) -> None:
#         for bad_model in MODELS_TO_DROP:
#             result = clean_model(pd.Series([bad_model]))
#             assert pd.isna(result.iloc[0]), (
#                 f"Expected NaN for model '{bad_model}'"
#             )

#     def test_nan_input_returns_nan(self) -> None:
#         result = clean_model(pd.Series([np.nan]))
#         assert pd.isna(result.iloc[0])

#     def test_all_model_aliases_are_mapped(self) -> None:
#         for alias, expected in MODEL_ALIASES.items():
#             result = clean_model(pd.Series([alias]))
#             # MODELS_TO_DROP aliases map to NaN via 'other' etc. –
#             # only check aliases that don't map to drop targets
#             if expected not in MODELS_TO_DROP:
#                 assert result.iloc[0] == expected, (
#                     f"Model alias '{alias}' → expected '{expected}'"
#                 )

#     def test_special_chars_removed(self) -> None:
#         series = pd.Series(["some model!"])
#         result = clean_model(series)
#         # Should strip the '!'
#         assert "!" not in str(result.iloc[0])

#     def test_empty_after_cleanup_becomes_nan(self) -> None:
#         series = pd.Series(["!!!"])
#         result = clean_model(series)
#         assert pd.isna(result.iloc[0])


# # ===========================================================================
# # validate_schema
# # ===========================================================================


# class TestValidateSchema:
#     def _schema_df(self) -> pd.DataFrame:
#         """Minimal DataFrame that satisfies SCHEMA requirements."""
#         n = 10
#         return pd.DataFrame({
#             "price": np.full(n, 5_000.0),
#             "power": np.full(n, 100.0),
#             "kilometer": np.full(n, 50_000.0),
#             "yearOfRegistration": np.full(n, 2010, dtype="int64"),
#             "brand": ["volkswagen"] * n,
#             "model": ["golf"] * n,
#             "vehicleType": ["sedan"] * n,
#             "gearbox": ["manual"] * n,
#             "fuelType": ["gasoline"] * n,
#             "seller": ["private"] * n,
#             "dataSource": ["web"] * n,
#         })

#     def test_no_violations_for_valid_df(self) -> None:
#         violations = validate_schema(self._schema_df(), SCHEMA)
#         assert violations == []

#     def test_missing_column_reported(self) -> None:
#         df = self._schema_df().drop(columns=["price"])
#         violations = validate_schema(df, SCHEMA)
#         assert any("price" in v for v in violations)

#     def test_null_in_non_nullable_column_reported(self) -> None:
#         df = self._schema_df()
#         df.loc[0, "price"] = np.nan
#         violations = validate_schema(df, SCHEMA)
#         assert any("price" in v for v in violations)

#     def test_value_below_min_reported(self) -> None:
#         df = self._schema_df()
#         df.loc[0, "price"] = MIN_PRICE - 1
#         violations = validate_schema(df, SCHEMA)
#         assert any("price" in v for v in violations)

#     def test_value_above_max_reported(self) -> None:
#         df = self._schema_df()
#         df.loc[0, "price"] = MAX_PRICE + 1
#         violations = validate_schema(df, SCHEMA)
#         assert any("price" in v for v in violations)

#     def test_returns_list(self) -> None:
#         violations = validate_schema(self._schema_df(), SCHEMA)
#         assert isinstance(violations, list)

#     def test_multiple_violations_accumulate(self) -> None:
#         df = self._schema_df()
#         df.loc[0, "price"] = np.nan
#         df.loc[1, "power"] = np.nan
#         violations = validate_schema(df, SCHEMA)
#         assert len(violations) >= 2


# # ===========================================================================
# # remove_invalid_rows
# # ===========================================================================


# class TestRemoveInvalidRows:
#     def test_valid_df_unchanged_length(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         before = len(valid_df)
#         result = remove_invalid_rows(valid_df.copy())
#         assert len(result) == before

#     def test_removes_price_below_min(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = MIN_PRICE - 1
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_removes_price_above_max(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = MAX_PRICE + 1
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_removes_nan_price(self, valid_df: pd.DataFrame) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = np.nan
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_removes_year_out_of_range(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "yearOfRegistration"] = YEAR_RANGE[0] - 1
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_removes_km_below_zero(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "kilometer"] = -1.0
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_removes_power_below_min(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "power"] = POWER_MIN - 1
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_removes_power_above_max(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "power"] = POWER_MAX + 1
#         result = remove_invalid_rows(df)
#         assert len(result) == len(valid_df) - 1

#     def test_index_reset_after_removal(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = np.nan
#         result = remove_invalid_rows(df)
#         assert list(result.index) == list(range(len(result)))


# # ===========================================================================
# # drop_duplicates
# # ===========================================================================


# class TestDropDuplicates:
#     def test_unique_df_unchanged(self, valid_df: pd.DataFrame) -> None:
#         before = len(valid_df)
#         result = drop_duplicates(valid_df.copy())
#         assert len(result) == before

#     def test_exact_duplicates_removed(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = pd.concat(
#             [valid_df, valid_df.iloc[:5]], ignore_index=True
#         )
#         result = drop_duplicates(df)
#         assert len(result) == len(valid_df)

#     def test_index_reset_after_dedup(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = pd.concat(
#             [valid_df, valid_df.iloc[:3]], ignore_index=True
#         )
#         result = drop_duplicates(df)
#         assert list(result.index) == list(range(len(result)))

#     def test_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
#         result = drop_duplicates(valid_df.copy())
#         assert isinstance(result, pd.DataFrame)

#     def test_all_duplicates_removed(self) -> None:
#         df = pd.DataFrame({
#             "a": [1, 1, 1],
#             "b": ["x", "x", "x"],
#         })
#         result = drop_duplicates(df)
#         assert len(result) == 1


# # ===========================================================================
# # impute_categoricals
# # ===========================================================================


# class TestImputeCategoricals:
#     def test_no_nans_after_imputation(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         # Introduce NaN in a categorical column
#         df.loc[0, "gearbox"] = np.nan
#         result = impute_categoricals(df)
#         assert result["gearbox"].isna().sum() == 0

#     def test_brand_cleaned_via_alias(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "brand"] = "vw"
#         result = impute_categoricals(df)
#         assert result.loc[0, "brand"] == "volkswagen"

#     def test_model_cleaned_via_alias(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "model"] = "3er"
#         result = impute_categoricals(df)
#         assert result.loc[0, "model"] == "3-series"

#     def test_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
#         result = impute_categoricals(valid_df.copy())
#         assert isinstance(result, pd.DataFrame)

#     def test_columns_preserved(self, valid_df: pd.DataFrame) -> None:
#         result = impute_categoricals(valid_df.copy())
#         for col in valid_df.columns:
#             assert col in result.columns


# # ===========================================================================
# # cap_outliers_iqr
# # ===========================================================================


# class TestCapOutliersIQR:
#     def test_extreme_value_is_capped(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = 999_999_999.0
#         result = cap_outliers_iqr(df, ["price"])
#         assert result["price"].max() < 999_999_999.0

#     def test_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
#         result = cap_outliers_iqr(valid_df.copy(), ["price"])
#         assert isinstance(result, pd.DataFrame)

#     def test_missing_col_skipped_gracefully(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         # Should not raise even if a column is absent
#         result = cap_outliers_iqr(
#             valid_df.copy(), ["nonexistent_col"]
#         )
#         assert isinstance(result, pd.DataFrame)

#     def test_no_capping_on_clean_data(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         original_max = valid_df["power"].max()
#         result = cap_outliers_iqr(valid_df.copy(), ["power"])
#         # After IQR capping a clean dataset the max may be equal or less
#         assert result["power"].max() <= original_max * 2

#     def test_multiple_cols_capped(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = 999_999_999.0
#         df.loc[1, "kilometer"] = 999_999_999.0
#         result = cap_outliers_iqr(df, ["price", "kilometer"])
#         assert result["price"].max() < 999_999_999.0
#         assert result["kilometer"].max() < 999_999_999.0


# # ===========================================================================
# # cap_outliers_fixed
# # ===========================================================================


# class TestCapOutliersFixed:
#     def test_price_capped_at_max(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = MAX_PRICE + 100_000
#         result = cap_outliers_fixed(df)
#         assert result["price"].max() <= MAX_PRICE

#     def test_price_capped_at_min(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "price"] = MIN_PRICE - 1
#         result = cap_outliers_fixed(df)
#         assert result["price"].min() >= MIN_PRICE

#     def test_kilometer_upper_cap_applied(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "kilometer"] = KM_RANGE[1] + 1
#         result = cap_outliers_fixed(df)
#         assert result["kilometer"].max() <= KM_RANGE[1]

#     def test_power_upper_cap_applied(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "power"] = POWER_MAX + 500
#         result = cap_outliers_fixed(df)
#         assert result["power"].max() <= POWER_MAX

#     def test_power_lower_cap_applied(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         df = valid_df.copy()
#         df.loc[0, "power"] = POWER_MIN - 1
#         result = cap_outliers_fixed(df)
#         assert result["power"].min() >= POWER_MIN

#     def test_all_cap_bound_cols_present_in_output(
#         self, valid_df: pd.DataFrame
#     ) -> None:
#         result = cap_outliers_fixed(valid_df.copy())
#         for col in CAP_BOUNDS:
#             assert col in result.columns

#     def test_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
#         result = cap_outliers_fixed(valid_df.copy())
#         assert isinstance(result, pd.DataFrame)


# # ===========================================================================
# # DataCleaner — integration smoke test
# # ===========================================================================


# class TestDataCleaner:
#     def test_run_returns_dataframe(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         csv_path = tmp_path / "input.csv"
#         valid_df.to_csv(csv_path, index=False)
#         cleaner = DataCleaner(use_iqr_capping=True)
#         result = cleaner.run(csv_path)
#         assert isinstance(result, pd.DataFrame)
#         assert len(result) > 0

#     def test_cleaned_df_stored_on_instance(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         csv_path = tmp_path / "input.csv"
#         valid_df.to_csv(csv_path, index=False)
#         cleaner = DataCleaner()
#         cleaner.run(csv_path)
#         assert cleaner.cleaned_df is not None
#         assert isinstance(cleaner.cleaned_df, pd.DataFrame)

#     def test_output_csv_written_when_path_given(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         csv_path = tmp_path / "input.csv"
#         output_path = tmp_path / "output.csv"
#         valid_df.to_csv(csv_path, index=False)
#         DataCleaner().run(csv_path, output_path=output_path)
#         assert output_path.exists()

#     def test_no_iqr_capping_variant_runs(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         csv_path = tmp_path / "input.csv"
#         valid_df.to_csv(csv_path, index=False)
#         cleaner = DataCleaner(use_iqr_capping=False)
#         result = cleaner.run(csv_path)
#         assert isinstance(result, pd.DataFrame)

#     def test_missing_required_column_raises(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         df = valid_df.drop(columns=["price"])
#         csv_path = tmp_path / "bad_input.csv"
#         df.to_csv(csv_path, index=False)
#         with pytest.raises(ValueError, match="missing required columns"):
#             DataCleaner().run(csv_path)

#     def test_cleaned_output_has_no_price_below_min(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         csv_path = tmp_path / "input.csv"
#         valid_df.to_csv(csv_path, index=False)
#         result = DataCleaner().run(csv_path)
#         assert (result["price"] >= MIN_PRICE).all()

#     def test_cleaned_output_has_no_price_above_max(
#         self, valid_df: pd.DataFrame, tmp_path
#     ) -> None:
#         csv_path = tmp_path / "input.csv"
#         valid_df.to_csv(csv_path, index=False)
#         result = DataCleaner().run(csv_path)
#         assert (result["price"] <= MAX_PRICE).all()
