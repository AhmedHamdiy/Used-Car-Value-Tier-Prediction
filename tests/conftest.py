import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_scraped():
    return pd.DataFrame({
        'brand': ['Peugeot-'],
        'fuelType': ['Electric/Gasoline'],
        'mileage': ['122566'],
        'year': ['02/2017'],
        'vehicleType': ['SUV/Off-Road/Pick-Up'],
        'seller': ['Dealer'],
        'power': ['60 kW (82 hp)'],
        'model': ['2008 '],
        'price': [7450],
        'gearbox': ['Automatic']
    })


@pytest.fixture()
def sample_kaggle() -> pd.DataFrame:
    np.random.seed(0)
    n = 40
    df = pd.DataFrame({
        "dateCrawled": pd.date_range("2016-01-01", periods=n, freq="h")
        .strftime("%Y-%m-%d %H:%M:%S"),
        "name": [f"car {i}" for i in range(n)],
        "seller": ["privat"] * n,
        "offerType": ["Angebot"] * n,
        "price": np.random.randint(1000, 50000, n).astype(float),
        "abtest": ["test"] * n,
        "vehicleType": ["limousine"] * n,
        "yearOfRegistration": np.random.randint(2000, 2018, n),
        "gearbox": ["manuell"] * n,
        "powerPS": np.random.randint(80, 200, n).astype(float),
        "model": ["golf"] * n,
        "kilometer": np.random.randint(10000, 150000, n).astype(float),
        "monthOfRegistration": np.random.randint(1, 13, n),
        "fuelType": ["benzin"] * n,
        "brand": ["volkswagen"] * n,
        "notRepairedDamage": ["nein"] * n,
        "dateCreated": pd.date_range("2016-01-01", periods=n, freq="h")
        .strftime("%Y-%m-%d"),
        "nrOfPictures": [0] * n,
        "postalCode": np.random.randint(10000, 99999, n),
        "lastSeen": pd.date_range("2016-03-01", periods=n, freq="h")
        .strftime("%Y-%m-%d"),
    })
    # generate some invalid values for testing validation logic
    for i in range(5):
        df.loc[i, "price"] = -500  # negative
        df.loc[i, "powerPS"] = 9999  # above max
        df.loc[i, "yearOfRegistration"] = 1900  # below min
        df.loc[i, "price"] = np.nan  # missing
        df.loc[i, "mileage"] = np.nan  # missing
    return df


@pytest.fixture
def transformed_data(sample_kaggle, sample_scraped):
    from src.data.merge_data import transform_data
    return transform_data(sample_kaggle, sample_scraped)
