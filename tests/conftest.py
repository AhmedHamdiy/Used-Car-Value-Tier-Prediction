import pandas as pd
import pytest

@pytest.fixture
def sample_kaggle():
    """Mock Kaggle data representing the pre-cleaned state."""
    return pd.DataFrame({
        'brand': ['VOLKSWAGEN', 'bmw_'],
        'fuelType': ['benzin', 'elektro'],
        'vehicleType': ['limousine', 'suv'],
        'seller': ['privat', 'gewerblich'],
        'powerPS': [100, 200],
        'model': ['golf', 'x_5'],
        'price': [1500, 30000],
        'kilometer': [150000, 50000],
        'yearOfRegistration': [2010, 2018]
    })

@pytest.fixture
def sample_scraped():
    """Mock AutoScout24 data representing the pre-cleaned state."""
    return pd.DataFrame({
        'brand': ['Peugeot-'],
        'fuelType': ['Electric/Gasoline'],
        'mileage': ['122566'],
        'year': ['02/2017'],
        'vehicleType': ['SUV/Off-Road/Pick-Up'],
        'seller': ['Dealer'],
        'power': ['60 kW (82 hp)'],
        'model': ['2008 '],
        'price': [7450]
    })