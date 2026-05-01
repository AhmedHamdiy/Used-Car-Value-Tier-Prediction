import pandas as pd
import tomllib
from pathlib import Path


with open(Path("configs/config.toml"), "rb") as f:
    config = tomllib.load(f)

KAGGLE_DATA_PATH = config["data"]["kaggle_raw_data_path"]
SCRAPED_DATA_PATH = config["data"]["scraped_raw_data_path"]
MERGED_DATA_PATH = config["data"]["merged_data_path"]

REFERENCE_YEAR: int = 2025

TARGET_COLS: list[str] = [
    "brand",
    "model",
    "vehicleType",
    "power",
    "gearbox",
    "kilometer",
    "fuelType",
    "yearOfRegistration",
    "seller",
    "dataSource",
    "price_reference_year",
    "price",
]


def _transform_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataSource"] = "kaggle"

    df = df.rename(columns={"powerPS": "power"})

    df["price_reference_year"] = (
        df["dateCrawled"].astype(str)
        .str[:4].astype(float)
        )

    return df


def _transform_crawled(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_reference_year"] = REFERENCE_YEAR
    df["dataSource"] = "crawled"

    df = df.rename(columns={"mileage": "kilometer"})

    df["power"] = (df["power"].astype(str)
                   .str.extract(r"\((\d+)\s*hp\)").astype(float))
    df["yearOfRegistration"] = (
        df["year"].astype(str).str.extract(r"(\d{4})").astype(float)
    )

    return df


def transform_data(
    df_kaggle: pd.DataFrame,
    df_crawled: pd.DataFrame,
) -> pd.DataFrame:
    df_k = _transform_kaggle(df_kaggle)
    df_c = _transform_crawled(df_crawled)

    df_k_final = df_k[TARGET_COLS].copy()
    df_c_final = df_c[TARGET_COLS].copy()

    return pd.concat([df_k_final, df_c_final], ignore_index=True)


def merge_datasets() -> None:
    print("Loading raw datasets...")
    df_kaggle = pd.read_csv(KAGGLE_DATA_PATH, encoding="latin-1")
    df_crawled = pd.read_csv(SCRAPED_DATA_PATH)

    print("Applying transformations...")
    df_merged = transform_data(df_kaggle, df_crawled)

    df_merged.to_csv(MERGED_DATA_PATH, index=False)
    print(f"Merging complete. Saved to {MERGED_DATA_PATH}")
