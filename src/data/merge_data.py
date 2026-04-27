import os
import re

import pandas as pd
import tomllib
from pathlib import Path


with open(Path("configs/config.toml"), "rb") as f:
	config = tomllib.load(f)

KAGGLE_DATA_PATH = config["data"]["kaggle_raw_data_path"]
SCRAPED_DATA_PATH = config["data"]["scraped_raw_data_path"]
MERGED_DATA_PATH = config["data"]["merged_data_path"]

def transform_data(df_kaggle, df_crawled):
    """Applies standard transformations and merges the datasets."""
    # 1. brand col
    df_kaggle['brand'] = df_kaggle['brand'].astype(str).str.lower().str.replace(r'[-_]', '', regex=True)
    df_crawled['brand'] = df_crawled['brand'].astype(str).str.lower().str.replace(r'[-_]', '', regex=True)

    # 2. fuelType col
    fuel_map_kaggle = {
        'benzin': 'gasoline', 'diesel': 'diesel', 'hybrid': 'hybrid',
        'elektro': 'electric', 'lpg': 'lpg', 'cng': 'cng', 'andere': 'other'
    }
    fuel_map_crawled = {
        'Gasoline': 'gasoline', 'Diesel': 'diesel', 
        'Electric/Gasoline': 'hybrid', 'Electric/Diesel': 'hybrid',
        'Electric': 'electric', 'LPG': 'lpg', 'CNG': 'cng'
    }
    df_kaggle['fuelType'] = df_kaggle['fuelType'].str.lower().map(fuel_map_kaggle)
    df_crawled['fuelType'] = df_crawled['fuelType'].map(fuel_map_crawled)

    # 3. kilometer/mileage col
    df_crawled = df_crawled.rename(columns={'mileage': 'kilometer'})
    df_crawled['kilometer'] = pd.to_numeric(df_crawled['kilometer'], errors='coerce').astype('Int64')

    # 4. year col
    df_crawled['yearOfRegistration'] = df_crawled['year'].astype(str).str.extract(r'(\d{4})').astype(float)
    df_kaggle['yearOfRegistration'] = pd.to_numeric(df_kaggle['yearOfRegistration'], errors='coerce')

    # 5. vehicleType col
    vt_map_kaggle = {
        'limousine': 'sedan', 'kleinwagen': 'compact', 'kombi': 'station_wagon',
        'bus': 'van', 'cabrio': 'convertible', 'coupe': 'coupe', 'suv': 'suv', 'andere': 'other'
    }
    vt_map_crawled = {
        'SUV/Off-Road/Pick-Up': 'suv', 'Sedan': 'sedan', 'Compact': 'compact',
        'Station Wagon': 'station_wagon', 'Van': 'van', 'Convertible': 'convertible',
        'Coupe': 'coupe', 'Other': 'other'
    }
    df_kaggle['vehicleType'] = df_kaggle['vehicleType'].str.lower().map(vt_map_kaggle)
    df_crawled['vehicleType'] = df_crawled['vehicleType'].map(vt_map_crawled)

    # 6. seller col
    seller_map_kaggle = {'privat': 'private', 'gewerblich': 'dealer'}
    seller_map_crawled = {'Dealer': 'dealer'}
    df_kaggle['seller'] = df_kaggle['seller'].str.lower().map(seller_map_kaggle)
    df_crawled['seller'] = df_crawled['seller'].map(seller_map_crawled)

    # 7. power col
    df_kaggle = df_kaggle.rename(columns={'powerPS': 'power'})
    df_crawled['power'] = df_crawled['power'].astype(str).str.extract(r'\((\d+)\s*hp\)').astype(float)

    # 8. model col
    df_kaggle['model'] = df_kaggle['model'].astype(str).str.lower().str.replace(r'[-_ ]', '', regex=True)
    df_crawled['model'] = df_crawled['model'].astype(str).str.lower().str.replace(r'[-_ ]', '', regex=True)

    # 9. Target columns and merge
    target_cols = [
        'brand', 'fuelType', 'kilometer', 'yearOfRegistration', 
        'vehicleType', 'seller', 'power', 'model', 'price'
    ]

    df_k_final = df_kaggle[target_cols].copy()
    df_c_final = df_crawled[target_cols].copy()

    df_k_final['data_source'] = 'kaggle'
    df_c_final['data_source'] = 'crawled'

    return pd.concat([df_k_final, df_c_final], ignore_index=True)

def merge_datasets():    
    print("Loading raw datasets...")
    df_kaggle = pd.read_csv(f'{KAGGLE_DATA_PATH}', encoding='latin-1')
    df_crawled = pd.read_csv(f'{SCRAPED_DATA_PATH}')

    print("Applying transformations...")
    df_merged = transform_data(df_kaggle, df_crawled)

    df_merged.to_csv(MERGED_DATA_PATH, index=False)
    print(f"Data merging complete. Saved to {MERGED_DATA_PATH}")


if __name__ == "__main__":
    merge_datasets()
