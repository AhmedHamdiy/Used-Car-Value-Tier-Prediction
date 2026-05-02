import pandas as pd
import tomllib
from pathlib import Path


with open(Path("configs/config.toml"), "rb") as f:
    config = tomllib.load(f)

KAGGLE_DATA_PATH = config["data"]["kaggle_raw_data_path"]
SCRAPED_DATA_PATH = config["data"]["scraped_raw_data_path"]
MERGED_DATA_PATH = config["data"]["merged_data_path"]

CPI_USED_CARS: dict[int, float] = {
    2014: 99.5,
    2015: 100.0,
    2016: 100.5,
    2017: 102.0,
    2018: 103.8,
    2019: 105.3,
    2020: 100.0,
    2021: 109.1,
    2022: 128.5,
    2023: 138.9,
    2024: 145.3,
    2025: 147.1,
    2026: 147.9,
}

REFERENCE_YEAR: int = 2026

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
    "price",
    "price_tier",
]


def normalize_price(price: float, source_year: int) -> float:
    if source_year not in CPI_USED_CARS:
        raise ValueError(f"No CPI entry for year {source_year}.")
    return price * (CPI_USED_CARS[REFERENCE_YEAR] / CPI_USED_CARS[source_year])


def extract_year(registration):
    if pd.isna(registration):
        return None
    match = pd.Series(str(registration)).str.extract(r"(\d{4})")
    return int(match[0]) if not match.empty else None


def price_tier(price: float) -> str:
    if price < 5000:
        return "budget"
    elif price < 15000:
        return "mid-range"
    else:
        return "luxury"


def _transform_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataSource"] = "kaggle"

    df = df.rename(columns={"powerPS": "power"})
    df["yearCrawled"] = df["dateCrawled"].str[:4].astype(int)
    df["price"] = df.apply(
        lambda row: normalize_price(row["price"], row["yearCrawled"]), axis=1
    )

    df["price_tier"] = df["price"].apply(price_tier)

    return df


def _transform_crawled(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dataSource"] = "crawled"
    df = df.rename(columns={"mileage": "kilometer"})

    df["power"] = (df["power"].astype(str).str
                   .extract(r"\((\d+)\s*hp\)").astype(float))
    df["yearOfRegistration"] = (
        df["year"].astype(str).str.extract(r"(\d{4})").astype(float)
    )
    df["price_tier"] = df["price"].apply(price_tier)

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
