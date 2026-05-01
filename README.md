# Used-Car-Value-Tier-Prediction

Applied Data Science - Spring 2026 Project

## Team Members

- Ahmed Hamdy
- Ahmed Aladdin
- Asmaa Abozaid
- Shehab Khalid

## Project Description

Used car dealerships and online marketplaces receive large volumes of listings and trade-ins. Manually assigning each vehicle to a value tier is time-consuming and inconsistent across staff and branches. This project automates the classification of used cars into value tiers using historical market data.

Primary stakeholders: used car dealerships (pricing and inventory teams). Secondary stakeholders: online marketplaces (listing recommendations) and insurance underwriters (risk and valuation support).

## Target Variable

`price_tier` is a categorical target derived from the listing price:

- **Budget**: price <= 5,000
- **Mid-Range**: 5,000 < price <= 15,000
- **Luxury**: price > 15,000

Class distribution (from the proposal): 62.90% Budget, 25.57% Mid-Range, 8.62% Luxury.

## Data Sources

- **Kaggle**: "Uncovering Factors That Affect Used Car Prices" (autos.csv)
  - https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices
- **AutoScout24**: scraped listings to increase coverage and validate pricing patterns
  - https://www.autoscout24.com

## Features

Core fields used in modeling include:

- `seller`, `vehicleType`, `yearOfRegistration`, `gearbox`, `power`, `brand`, `kilometer`, `fuelType`

Planned engineered features:

- `vehicleAge = 2026 - yearOfRegistration`
- `kmPerYear = kilometer / (vehicleAge + 1)`
- `isAutomatic = (gearbox == "automatic")`
- `isVintageClassic` derived from age and brand
- `powerTierBin` derived from `power`

Note: after deriving `price_tier`, raw `price` is not used as an input feature to avoid target leakage.

## Repository Structure

- `data/`
  - `raw/` - original sources (Kaggle + scraping)
  - `interim/` - merged intermediate data
  - `processed/` - cleaned and validated data
- `scripts/`
  - `data/` - download, merge, and scraping helpers
- `src/`
  - `data/` - preprocessing and validation
  - `features/` - feature engineering
  - `models/` - training and prediction
  - `visualizations/` - plotting utilities
- `notebooks/` - EDA and analysis notebooks
- `reports/` - figures and validation reports
- `tests/` - unit and integration tests

## How to Run

### 1) Environment setup

This project uses Poetry and Python 3.11+.

```bash
poetry install
```

### 2) Data acquisition

Kaggle access requires API credentials in `~/.kaggle/kaggle.json`.

```bash
poetry run python scripts/data/download_kaggle.py
poetry run bash scripts/data/scrape_data.sh
```

### 3) Merge and preprocess

```bash
poetry run python scripts/data/merge_data.py
poetry run python src/data/preprocess_data.py
poetry run python src/data/validate_data.py
```

### 4) Run notebooks

```bash
poetry run jupyter lab
```

### 5) Tests

```bash
poetry run pytest tests/
```
