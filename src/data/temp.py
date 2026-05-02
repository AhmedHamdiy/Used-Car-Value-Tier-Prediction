# DATA TRANSFORMATION SECTION

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


def normalize_price(price: float, source_year: int) -> float:
    if source_year not in CPI_USED_CARS:
        raise ValueError(
            f"No CPI entry for year {source_year}."
        )
    return price * (CPI_USED_CARS[REFERENCE_YEAR] / CPI_USED_CARS[source_year])
