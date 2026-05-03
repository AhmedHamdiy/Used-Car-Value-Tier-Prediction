#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train_model import train_all_models  # : E402


def main():
    parser = argparse.ArgumentParser(
        description="Train models for used car price tier prediction"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/clean_data.csv",
        help="Path to the cleaned CSV data file (default: data/processed/clean_data.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reports/model_comparison.csv",
        help="Path to save results CSV (default: reports/model_comparison.csv)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking URI (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="used_car_price_tier",
        help="MLflow experiment name (default: used_car_price_tier)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to train (default: all available)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset types (default: none smote undersample)",
    )

    args = parser.parse_args()

    # Determine MLflow URI
    mlflow_uri = None if args.no_mlflow else args.mlflow_uri

    print("=" * 80)
    print("MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"MLflow URI: {mlflow_uri or 'Disabled (use --no-mlflow to disable)'}")
    print(f"Experiment: {args.experiment_name}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    if args.datasets:
        print(f"Datasets: {', '.join(args.datasets)}")
    print("=" * 80)

    train_all_models(
        data_path=args.data_path,
        output_path=args.output_path,
        models=args.models,
        datasets=args.datasets,
        experiment_name=args.experiment_name,
        mlflow_tracking_uri=mlflow_uri,
    )

    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
