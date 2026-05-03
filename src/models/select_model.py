"""
Model selection module for choosing the best model from training results.

This module provides functionality to:
1. Load and analyze model comparison results
2. Select the best model based on configurable criteria
3. Export best model recommendations and reports
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class SelectionCriteria:
    """Configuration for model selection criteria."""

    primary_metric: str = "luxury_recall"
    secondary_metric: Optional[str] = "val_f1"
    min_threshold: Optional[float] = None
    max_severe_misclassification: Optional[float] = None
    weight_primary: float = 0.7
    weight_secondary: float = 0.3


class ModelSelector:
    """
    Selects the best model from training results based on configurable criteria.

    Attributes:
        results_df: DataFrame containing model comparison results
        criteria: SelectionCriteria object defining how to select the best model
    """

    def __init__(self, results_path: str, criteria: Optional[SelectionCriteria] = None):
        """
        Initialize the model selector.

        Args:
            results_path: Path to the model comparison CSV file
            criteria: Selection criteria configuration
        """
        self.results_path = Path(results_path)
        self.criteria = criteria or SelectionCriteria()
        self.results_df = self._load_results()

    def _load_results(self) -> pd.DataFrame:
        """Load results from CSV file."""
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_path}\n"
                "Please run model training first with: make train"
            )

        df = pd.read_csv(self.results_path)

        # Validate required columns
        required_cols = ["model", "dataset", "test_f1"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in results: {missing_cols}")

        return df

    def calculate_composite_score(
        self, df: pd.DataFrame, criteria: Optional[SelectionCriteria] = None
    ) -> pd.DataFrame:
        """
        Calculate a composite score for ranking models.

        Args:
            df: Results DataFrame
            criteria: Selection criteria (uses self.criteria if None)

        Returns:
            DataFrame with added 'composite_score' column
        """
        criteria = criteria or self.criteria
        df = df.copy()

        # Normalize primary metric to 0-1 scale
        primary_values = df[criteria.primary_metric]
        primary_min, primary_max = primary_values.min(), primary_values.max()
        if primary_max > primary_min:
            primary_normalized = (primary_values - primary_min) / (
                primary_max - primary_min
            )
        else:
            primary_normalized = pd.Series(1.0, index=df.index)

        # Calculate composite score
        if criteria.secondary_metric and criteria.secondary_metric in df.columns:
            secondary_values = df[criteria.secondary_metric]
            secondary_min, secondary_max = (
                secondary_values.min(),
                secondary_values.max(),
            )
            if secondary_max > secondary_min:
                secondary_normalized = (secondary_values - secondary_min) / (
                    secondary_max - secondary_min
                )
            else:
                secondary_normalized = pd.Series(1.0, index=df.index)

            df["composite_score"] = (
                criteria.weight_primary * primary_normalized
                + criteria.weight_secondary * secondary_normalized
            )
        else:
            df["composite_score"] = primary_normalized

        return df

    def filter_models(
        self, df: pd.DataFrame, criteria: Optional[SelectionCriteria] = None
    ) -> pd.DataFrame:
        """
        Filter models based on threshold criteria.

        Args:
            df: Results DataFrame
            criteria: Selection criteria

        Returns:
            Filtered DataFrame
        """
        criteria = criteria or self.criteria

        # Apply minimum threshold on primary metric
        if criteria.min_threshold is not None:
            df = df[df[criteria.primary_metric] >= criteria.min_threshold]

        # Apply maximum threshold on severe misclassification rate
        if criteria.max_severe_misclassification is not None:
            if "severe_misclassification_rate" in df.columns:
                df = df[
                    df["severe_misclassification_rate"]
                    <= criteria.max_severe_misclassification
                ]

        return df

    def select_best_model(
        self,
        criteria: Optional[SelectionCriteria] = None,
        model_type: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Select the best model based on criteria.

        Args:
            criteria: Selection criteria (uses self.criteria if None)
            model_type: Filter by specific model type (e.g., 'xgboost')
            dataset_type: Filter by dataset type (e.g., 'none', 'smote')

        Returns:
            Dictionary with best model information
        """
        criteria = criteria or self.criteria
        df = self.results_df.copy()

        # Apply filters
        if model_type:
            df = df[df["model"] == model_type]
        if dataset_type:
            df = df[df["dataset"] == dataset_type]

        # Filter by thresholds
        df = self.filter_models(df, criteria)

        if len(df) == 0:
            return {
                "error": "No models meet the selection criteria",
                "criteria": criteria.__dict__,
            }

        # Calculate composite scores
        df = self.calculate_composite_score(df, criteria)

        # Sort by composite score (descending)
        df = df.sort_values("composite_score", ascending=False)

        # Get best model
        best = df.iloc[0]

        return {
            "model_name": best["model"],
            "dataset_type": best["dataset"],
            "metrics": {
                "test_f1": float(best.get("test_f1", 0)),
                "val_f1": float(best.get("val_f1", 0)),
                "test_bal_acc": float(best.get("test_bal_acc", 0)),
                "luxury_recall": float(best.get("luxury_recall", 0)),
                "severe_misclassification_rate": float(
                    best.get("severe_misclassification_rate", 0)
                ),
                "composite_score": float(best.get("composite_score", 0)),
            },
            "rank": 1,
            "total_models_considered": len(df),
        }

    def get_top_k_models(self, k: int = 5) -> pd.DataFrame:
        """
        Get the top k models by test_f1 score.

        Args:
            k: Number of top models to return

        Returns:
            DataFrame with top k models
        """
        df = self.calculate_composite_score(self.results_df)
        return df.sort_values("composite_score", ascending=False).head(k)

    def compare_models_by_type(self) -> pd.DataFrame:
        """
        Compare the best performing configuration for each model type.

        Returns:
            DataFrame with best config per model type
        """
        df = self.calculate_composite_score(self.results_df)

        # Group by model and get best performing variant
        best_per_model = df.loc[df.groupby("model")["composite_score"].idxmax()]

        return best_per_model.sort_values("composite_score", ascending=False)

    def analyze_dataset_impact(self) -> pd.DataFrame:
        """
        Analyze the impact of different dataset preprocessing methods.

        Returns:
            DataFrame showing performance by dataset type
        """
        summary = (
            self.results_df.groupby("dataset")
            .agg(
                {
                    "test_f1": ["mean", "std", "min", "max"],
                    "luxury_recall": ["mean", "std"],
                    "severe_misclassification_rate": ["mean", "std"],
                }
            )
            .round(4)
        )

        return summary

    def generate_report(
        self,
        output_path: Optional[str] = None,
        criteria: Optional[SelectionCriteria] = None,
    ) -> str:
        """
        Generate a comprehensive model selection report.

        Args:
            output_path: Path to save the report (optional)
            criteria: Selection criteria

        Returns:
            Report string
        """
        criteria = criteria or self.criteria

        report_lines = [
            "=" * 80,
            "MODEL SELECTION REPORT",
            "=" * 80,
            "",
            f"Results file: {self.results_path}",
            f"Total models evaluated: {len(self.results_df)}",
            "",
            "SELECTION CRITERIA:",
            f"  Primary metric: {criteria.primary_metric}",
            f"  Secondary metric: {criteria.secondary_metric}",
            f"  Primary weight: {criteria.weight_primary}",
            f"  Secondary weight: {criteria.weight_secondary}",
        ]

        if criteria.min_threshold:
            report_lines.append(f"  Minimum threshold: {criteria.min_threshold}")
        if criteria.max_severe_misclassification:
            report_lines.append(
                f"  Max severe misclassification: {criteria.max_severe_misclassification}"
            )

        report_lines.extend(
            [
                "",
                "-" * 80,
                "BEST MODEL:",
                "-" * 80,
            ]
        )

        best = self.select_best_model(criteria)
        if "error" in best:
            report_lines.append(f"ERROR: {best['error']}")
        else:
            report_lines.extend(
                [
                    f"  Model: {best['model_name']}",
                    f"  Dataset: {best['dataset_type']}",
                    f"  Rank: {best['rank']} of {best['total_models_considered']}",
                    "",
                    "  Metrics:",
                    f"    Test F1:          {best['metrics']['test_f1']:.4f}",
                    f"    Validation F1:    {best['metrics']['val_f1']:.4f}",
                    f"    Test Balanced Acc: {best['metrics']['test_bal_acc']:.4f}",
                    f"    Luxury Recall:    {best['metrics']['luxury_recall']:.4f}",
                    f"    Severe Misclass Rate: "
                    f"{best['metrics']['severe_misclassification_rate']:.4f}",
                    f"    Composite Score:  {best['metrics']['composite_score']:.4f}",
                ]
            )

        report_lines.extend(
            [
                "",
                "-" * 80,
                "TOP 5 MODELS:",
                "-" * 80,
            ]
        )

        top5 = self.get_top_k_models(5)
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            report_lines.append(
                f"{idx}. {row['model']:15s} ({row['dataset']:12s}) | "
                f"F1={row['test_f1']:.4f} | "
                f"LuxuryRec={row.get('luxury_recall', 0):.4f} | "
                f"SevereErr={row.get('severe_misclassification_rate', 0):.4f}"
            )

        report_lines.extend(
            [
                "",
                "-" * 80,
                "BEST CONFIG PER MODEL TYPE:",
                "-" * 80,
            ]
        )

        best_per_type = self.compare_models_by_type()
        for _, row in best_per_type.iterrows():
            report_lines.append(
                f"  {row['model']:15s}: {row['dataset']:12s} | "
                f"F1={row['test_f1']:.4f} | "
                f"Composite={row['composite_score']:.4f}"
            )

        report_lines.extend(
            [
                "",
                "-" * 80,
                "DATASET IMPACT ANALYSIS:",
                "-" * 80,
            ]
        )

        dataset_impact = self.analyze_dataset_impact()
        report_lines.append(dataset_impact.to_string())

        report_lines.extend(
            [
                "",
                "=" * 80,
            ]
        )

        report = "\n".join(report_lines)

        if output_path:
            Path(output_path).write_text(report)
            print(f"Report saved to: {output_path}")

        return report

    def export_best_model_info(
        self, output_path: str, criteria: Optional[SelectionCriteria] = None
    ) -> None:
        """
        Export best model information to JSON.

        Args:
            output_path: Path to save JSON file
            criteria: Selection criteria
        """
        criteria = criteria or self.criteria
        best = self.select_best_model(criteria)

        output = {
            "best_model": best,
            "selection_criteria": criteria.__dict__,
            "top_5_models": self.get_top_k_models(5).to_dict("records"),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        Path(output_path).write_text(json.dumps(output, indent=2))
        print(f"Best model info exported to: {output_path}")


def select_best_model_from_results(
    results_path: str = "reports/model_comparison.csv",
    primary_metric: str = "test_f1",
    secondary_metric: str = "luxury_recall",
    min_test_f1: Optional[float] = None,
    output_report: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to select the best model from results file.

    Args:
        results_path: Path to results CSV
        primary_metric: Primary metric for selection
        secondary_metric: Secondary metric for selection
        min_test_f1: Minimum test F1 threshold
        output_report: Path to save report (optional)

    Returns:
        Dictionary with best model information
    """
    criteria = SelectionCriteria(
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        min_threshold=min_test_f1,
    )

    selector = ModelSelector(results_path, criteria)

    if output_report:
        selector.generate_report(output_report, criteria)

    return selector.select_best_model(criteria)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "reports/model_comparison.csv"

    selector = ModelSelector(results_path)
    report = selector.generate_report()
    print(report)
