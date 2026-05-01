import tomllib
from pathlib import Path
from src.data.clean_data import DataCleaner


with open(Path("configs/config.toml"), "rb") as f:
    config = tomllib.load(f)

MERGED_DATA_PATH = config["data"]["merged_data_path"]

CLEAN_DATA_PATH = config["data"]["clean_data_path"]

if __name__ == "__main__":

    preprocessor = DataCleaner(use_iqr_capping=True)
    preprocessor.run(raw_path=MERGED_DATA_PATH, output_path=CLEAN_DATA_PATH)
