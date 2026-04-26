import pandas as pd
import os


def merge_datasets():
    raw_dir = 'data/raw'
    clean_dir = 'data/clean'

    os.makedirs(clean_dir, exist_ok=True)

    # Load datasets
    print("Loading raw datasets...")
    df_kaggle = pd.read_csv(f'{raw_dir}/autos.csv', encoding='latin-1')
    df_crawled = pd.read_csv(f'{raw_dir}/cars_dataset.csv')

    # 1. brand col
    df_kaggle['brand'] = (df_kaggle['brand'].astype(str).str.lower()
                          .str.replace(r'[-_]', '', regex=True))
    df_crawled['brand'] = (df_crawled['brand'].astype(str).str.lower()
                           .str.replace(r'[-_]', '', regex=True))

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
    df_kaggle['fuelType'] = (df_kaggle['fuelType'].str.lower()
                             .map(fuel_map_kaggle))
    df_crawled['fuelType'] = df_crawled['fuelType'].map(fuel_map_crawled)

    # 3. kilometer/mileage col
    df_crawled.rename(columns={'mileage': 'kilometer'}, inplace=True)
    df_crawled['kilometer'] = (
        pd.to_numeric(df_crawled['kilometer'], errors='coerce').astype('Int64')
    )

    # 4. year col
    df_crawled['yearOfRegistration'] = (df_crawled['year'].astype(str).str
                                        .extract(r'(\d{4})').astype(float))

    # 5. vehicleType col
    vt_map_kaggle = {
        'limousine': 'sedan', 'kleinwagen': 'compact',
        'kombi': 'station_wagon', 'bus': 'van', 'cabrio': 'convertible',
        'coupe': 'coupe', 'suv': 'suv', 'andere': 'other'
    }
    vt_map_crawled = {
        'SUV/Off-Road/Pick-Up': 'suv', 'Sedan': 'sedan', 'Compact': 'compact',
        'Station Wagon': 'station_wagon', 'Van': 'van', 'Coupe': 'coupe',
        'Convertible': 'convertible',  'Other': 'other'
    }
    df_kaggle['vehicleType'] = (df_kaggle['vehicleType'].
                                str.lower().map(vt_map_kaggle))
    df_crawled['vehicleType'] = df_crawled['vehicleType'].map(vt_map_crawled)

    # 6. seller col
    seller_map_kaggle = {'privat': 'private', 'gewerblich': 'dealer'}
    seller_map_crawled = {'Dealer': 'dealer'}
    df_kaggle['seller'] = (df_kaggle['seller'].str
                           .lower().map(seller_map_kaggle))
    df_crawled['seller'] = df_crawled['seller'].map(seller_map_crawled)

    # 7. power col
    df_kaggle.rename(columns={'powerPS': 'power'}, inplace=True)
    df_crawled['power'] = (df_crawled['power'].astype(str).str
                           .extract(r'\((\d+)\s*hp\)').astype(float))

    # 8. model col
    df_kaggle['model'] = (
        df_kaggle['model'].astype(str).str.lower()
        .str.replace(r'[-_ ]', '', regex=True)
    )
    df_crawled['model'] = (
        df_crawled['model'].astype(str).str.lower()
        .str.replace(r'[-_ ]', '', regex=True)
    )

    target_cols = ['brand', 'model', 'vehicleType', 'power',
                   'yearOfRegistration', 'kilometer', 'fuelType',
                   'seller', 'price']

    df_kaggle_final = df_kaggle[target_cols].copy()
    df_crawled_final = df_crawled[target_cols].copy()

    print("Merging datasets...")
    df_merged = pd.concat([df_kaggle_final, df_crawled_final],
                          ignore_index=True)

    output_path = f'{clean_dir}/merged_data.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"Data merging complete. Saved to {output_path}")


if __name__ == "__main__":
    merge_datasets()
