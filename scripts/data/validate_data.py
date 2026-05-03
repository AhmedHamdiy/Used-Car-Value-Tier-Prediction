import tomllib
from pathlib import Path
import sys
import pandas as pd
from src.data.validate_data import DataValidator

with open(Path("configs/config.toml"), "rb") as f:
    config = tomllib.load(f)

MERGED_DATA_PATH = config["data"]["merged_data_path"]

CLEAN_DATA_PATH = config["data"]["clean_data_path"]

VALIDATION_REPORT_PATH = config["data"]["validation_report_path"]


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in {"raw", "clean"}:
        raise ValueError("Usage: python validate_data.py [raw|clean]")

    if args[0] == "raw":
        df = pd.read_csv(MERGED_DATA_PATH)
    elif args[0] == "clean":
        df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    validator = DataValidator()
    df["kilometer"] = df["kilometer"].astype(float)
    validator.run_all(df, output_file=VALIDATION_REPORT_PATH)
