#!/usr/bin/env python3
"""
CLI script to select the best model from training results.

Usage:
    python scripts/models/select_model.py
    python scripts/models/select_model.py --results reports/model_comparison.csv
    python scripts/models/select_model.py --primary-metric test_f1 --secondary-metric luxury_recall
    python scripts/models/select_model.py --min-f1 0.80 --export-json reports/best_model.json
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.select_model import (  # noqa: E402
    ModelSelector,
    SelectionCriteria,
)


def main():
    parser = argparse.ArgumentParser(
        description="Select the best model from training results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="reports/model_comparison.csv",
        help="Path to model comparison results CSV (default: reports/model_comparison.csv)",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="luxury_recall",
        help="Primary metric for model selection (default: luxury_recall)",
    )
    parser.add_argument(
        "--secondary-metric",
        type=str,
        default="val_f1",
        help="Secondary metric for model selection (default: val_f1)",
    )
    parser.add_argument(
        "--primary-weight",
        type=float,
        default=0.7,
        help="Weight for primary metric (default: 0.7)",
    )
    parser.add_argument(
        "--secondary-weight",
        type=float,
        default=0.3,
        help="Weight for secondary metric (default: 0.3)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="Minimum test F1 threshold (optional)",
    )
    parser.add_argument(
        "--max-severe-misclass",
        type=float,
        default=None,
        help="Maximum severe misclassification rate (optional)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Filter by specific model type (e.g., 'xgboost', 'random_forest')",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="Filter by dataset type (e.g., 'none', 'smote', 'undersample')",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save selection report (optional)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Path to export best model info as JSON (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Show top k models (optional)",
    )
    parser.add_argument(
        "--compare-by-model",
        action="store_true",
        help="Show best config per model type",
    )
    parser.add_argument(
        "--analyze-datasets",
        action="store_true",
        help="Show dataset impact analysis",
    )

    args = parser.parse_args()

    # Validate weights sum to 1.0 (approximately)
    total_weight = args.primary_weight + args.secondary_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Weights sum to {total_weight:.2f}, normalizing to 1.0")
        args.primary_weight /= total_weight
        args.secondary_weight /= total_weight

    print("=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print(f"Primary metric: {args.primary_metric} (weight: {args.primary_weight:.2f})")
    print(
        f"Secondary metric: {args.secondary_metric} (weight: {args.secondary_weight:.2f})"
    )
    if args.min_f1:
        print(f"Min F1 threshold: {args.min_f1:.4f}")
    if args.max_severe_misclass:
        print(f"Max severe misclass rate: {args.max_severe_misclass:.4f}")
    if args.model_type:
        print(f"Model type filter: {args.model_type}")
    if args.dataset_type:
        print(f"Dataset type filter: {args.dataset_type}")
    print("=" * 80)

    try:
        # Create selection criteria
        criteria = SelectionCriteria(
            primary_metric=args.primary_metric,
            secondary_metric=args.secondary_metric,
            weight_primary=args.primary_weight,
            weight_secondary=args.secondary_weight,
            min_threshold=args.min_f1,
            max_severe_misclassification=args.max_severe_misclass,
        )

        # Initialize selector
        selector = ModelSelector(args.results, criteria)

        # Generate and print full report
        report = selector.generate_report(args.output_report, criteria)
        print(report)

        # Show top k models if requested
        if args.top_k:
            print(f"\n{'=' * 80}")
            print(f"TOP {args.top_k} MODELS:")
            print(f"{'=' * 80}")
            top_k = selector.get_top_k_models(args.top_k)
            print(top_k.to_string())

        # Compare by model type if requested
        if args.compare_by_model:
            print(f"\n{'=' * 80}")
            print("BEST CONFIG PER MODEL TYPE:")
            print(f"{'=' * 80}")
            best_per_type = selector.compare_models_by_type()
            display_cols = [
                "model",
                "dataset",
                "test_f1",
                "luxury_recall",
                "severe_misclassification_rate",
                "composite_score",
            ]
            available_cols = [c for c in display_cols if c in best_per_type.columns]
            print(best_per_type[available_cols].to_string())

        # Analyze dataset impact if requested
        if args.analyze_datasets:
            print(f"\n{'=' * 80}")
            print("DATASET IMPACT ANALYSIS:")
            print(f"{'=' * 80}")
            dataset_impact = selector.analyze_dataset_impact()
            print(dataset_impact.to_string())

        # Export best model info if requested
        if args.export_json:
            selector.export_best_model_info(args.export_json, criteria)

        # Get and display best model
        best = selector.select_best_model(
            criteria=criteria,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
        )

        if "error" in best:
            print(f"\nERROR: {best['error']}")
            return 1

        print(f"\n{'=' * 80}")
        print("SELECTED BEST MODEL:")
        print(f"{'=' * 80}")
        print(f"Model: {best['model_name']}")
        print(f"Dataset: {best['dataset_type']}")
        print("\nMetrics:")
        for metric, value in best["metrics"].items():
            print(f"  {metric:25s}: {value:.4f}")
        print(f"{'=' * 80}")

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run model training first:")
        print("  make train")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
