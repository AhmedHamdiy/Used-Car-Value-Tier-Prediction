"""
Visualization utilities for the car price tier prediction project.

This module provides reusable plotting functions for:
1. Data exploration and EDA
2. Model performance visualization
3. Feature importance plots
4. Business metrics dashboards
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set default style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def plot_price_distribution(
    df: pd.DataFrame,
    column: str = "price",
    title: str = "Price Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of prices with tier boundaries.

    Args:
        df: DataFrame with price data
        column: Price column name
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots()

    sns.histplot(df[column], bins=50, kde=True, ax=ax)
    ax.axvline(5000, color="red", linestyle="--", label="Budget/Mid boundary")
    ax.axvline(15000, color="red", linestyle="--", label="Mid/Luxury boundary")

    ax.set_xlabel("Price (€)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_price_tier_distribution(
    df: pd.DataFrame,
    column: str = "price_tier",
    title: str = "Price Tier Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of price tiers.

    Args:
        df: DataFrame with price tier data
        column: Price tier column name
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots()

    tier_counts = df[column].value_counts()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, Orange, Red

    bars = ax.bar(tier_counts.index, tier_counts.values, color=colors)

    # Add percentage labels
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = height / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Count")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance horizontal bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort and take top N
    df_plot = importance_df.sort_values("importance", ascending=True).tail(top_n)

    # Clean feature names
    name_mapping = {
        "yearOfRegistration": "Registration Year",
        "kilometer": "Kilometers Driven",
        "power": "Horsepower (PS)",
        "vehicleType_suv": "Is SUV",
        "gearbox_automatic": "Has Automatic Gearbox",
        "vehicleType_convertible": "Is Convertible",
        "vehicleType_coupe": "Is Coupe",
        "vehicleType_sedan": "Is Sedan",
        "vehicleType_van": "Is Van",
        "vehicleType_station-wagon": "Is Station Wagon",
        "fuelType_diesel": "Diesel Fuel",
        "fuelType_gasoline": "Gasoline Fuel",
        "fuelType_hybrid": "Hybrid Fuel",
        "fuelType_electric": "Electric Fuel",
        "seller_private": "Private Seller",
        "brand_freq": "Brand Frequency",
        "model_freq": "Model Frequency",
        "vehicleAge": "Vehicle Age",
        "kmPerYear": "Km Per Year",
        "km_power_ratio": "Km/Power Ratio",
        "log_km": "Log Kilometers",
    }
    df_plot["feature_clean"] = (
        df_plot["feature"].map(name_mapping).fillna(df_plot["feature"])
    )

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df_plot)))
    bars = ax.barh(df_plot["feature_clean"], df_plot["importance"], color=colors)

    ax.set_xlabel("Importance")
    ax.set_title(title)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "test_f1",
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot model comparison bar chart.

    Args:
        results_df: DataFrame with model results
        metric: Metric column to plot
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by metric
    df_plot = results_df.sort_values(metric, ascending=True)

    # Create labels
    df_plot["label"] = df_plot["model"] + " (" + df_plot["dataset"] + ")"

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_plot)))

    bars = ax.barh(df_plot["label"], df_plot[metric], color=colors)

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlim(0, 1)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.3f}",
            ha="left",
            va="center",
        )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = ["budget", "mid-range", "luxury"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        df: DataFrame with numeric features
        columns: Columns to include (None = all numeric)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if columns:
        df = df[columns]

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    corr = df_numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_scatter_by_tier(
    df: pd.DataFrame,
    x_col: str = "kilometer",
    y_col: str = "price",
    tier_col: str = "price_tier",
    title: str = "Price vs Mileage by Tier",
    save_path: Optional[str] = None,
    sample_size: int = 1500,
) -> plt.Figure:
    """
    Plot scatter plot colored by price tier.

    Args:
        df: DataFrame with car data
        x_col: X-axis column
        y_col: Y-axis column
        tier_col: Price tier column
        title: Plot title
        save_path: Optional path to save figure
        sample_size: Max points to plot (for performance)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots()

    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Plot each tier
    tiers = df[tier_col].unique()
    colors = {"budget": "green", "mid-range": "orange", "luxury": "red"}

    for tier in tiers:
        tier_df = df[df[tier_col] == tier]
        ax.scatter(
            tier_df[x_col],
            tier_df[y_col],
            c=colors.get(tier, "blue"),
            label=tier,
            alpha=0.5,
            s=20,
        )

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_eda_report(
    df: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> Dict[str, str]:
    """
    Generate a full EDA report with multiple plots.

    Args:
        df: DataFrame with car data
        output_dir: Directory to save figures

    Returns:
        Dictionary mapping plot names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Price distribution
    fig = plot_price_distribution(df, save_path=output_path / "price_distribution.png")
    plots["price_distribution"] = str(output_path / "price_distribution.png")
    plt.close(fig)

    # Tier distribution
    fig = plot_price_tier_distribution(
        df, save_path=output_path / "price_tier_distribution.png"
    )
    plots["price_tier_distribution"] = str(output_path / "price_tier_distribution.png")
    plt.close(fig)

    # Scatter plot
    fig = plot_scatter_by_tier(df, save_path=output_path / "price_vs_mileage.png")
    plots["price_vs_mileage"] = str(output_path / "price_vs_mileage.png")
    plt.close(fig)

    print(f"EDA report generated in: {output_dir}")
    return plots


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/processed/clean_data.csv")

    # Generate EDA report
    plots = create_eda_report(df)
    print(f"Generated plots: {list(plots.keys())}")
