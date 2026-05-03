#!/usr/bin/env python3
"""
CLI script for feature selection analysis.

Usage:
    python scripts/features/select_features.py
    python scripts/features/select_features.py --data data/processed/clean_data.csv --strategy xgboost
    python scripts/features/select_features.py --strategy variance --threshold 0.01
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.feature_selection import FeatureSelector, select_features
from src.models.train_model import prepare_data
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Select features using various strategies"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/clean_data.csv",
        help="Path to the cleaned CSV data file (default: data/processed/clean_data.csv)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="xgboost",
        choices=["xgboost", "variance", "correlation", "k_best", "mutual_info"],
        help="Feature selection strategy (default: xgboost)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for feature selection (strategy-dependent)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of features to select (for k_best, mutual_info strategies)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/selected_features.txt",
        help="Path to save selected features (default: reports/selected_features.txt)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/feature_importance_report.txt",
        help="Path to save feature importance report (default: reports/feature_importance_report.txt)",
    )
    parser.add_argument(
        "--export-data",
        type=str,
        default=None,
        help="Path to export selected feature data as CSV (optional)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to show in report (default: 30)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FEATURE SELECTION")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Strategy: {args.strategy}")
    if args.threshold:
        print(f"Threshold: {args.threshold}")
    if args.k:
        print(f"K: {args.k}")
    print("=" * 80)

    try:
        # Load data
        print(f"\nLoading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
        print(f"Loaded {len(df)} rows")

        # Prepare data
        print("Preparing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")

        # Apply feature selection
        print(f"\nApplying feature selection: {args.strategy}")
        X_train_sel, X_val_sel, X_test_sel, selector = select_features(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            X_test=X_test,
            strategy=args.strategy,
            threshold=args.threshold,
            k=args.k,
            random_state=42,
        )

        # Generate and save report
        print("\n" + selector.get_feature_importance_report(top_n=args.top_n))

        # Save report to file
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            f.write(selector.get_feature_importance_report(top_n=args.top_n))
        print(f"\nFeature importance report saved to: {args.report}")

        # Export selected features
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        selector.export_selected_features(args.output)

        # Export data if requested
        if args.export_data:
            print(f"\nExporting selected feature data...")
            Path(args.export_data).parent.mkdir(parents=True, exist_ok=True)

            # Combine with target for training data
            train_df = X_train_sel.copy()
            train_df["price_tier"] = y_train.values
            train_df["split"] = "train"

            val_df = X_val_sel.copy()
            val_df["price_tier"] = y_val.values
            val_df["split"] = "val"

            test_df = X_test_sel.copy()
            test_df["price_tier"] = y_test.values
            test_df["split"] = "test"

            # Combine all splits
            combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            combined_df.to_csv(args.export_data, index=False)
            print(f"Data exported to: {args.export_data}")

        print("\n" + "=" * 80)
        print("FEATURE SELECTION COMPLETE")
        print("=" * 80)
        print(f"Original features: {X_train.shape[1]}")
        print(f"Selected features: {X_train_sel.shape[1]}")
        print(f"Reduction: {(1 - X_train_sel.shape[1] / X_train.shape[1]) * 100:.1f}%")
        print("=" * 80)

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease prepare data first:")
        print("  make pipeline")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
