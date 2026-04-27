"""Train and track price-tier classification models with MLflow.

Implements report sections:
- 5.6 Model Development
- 5.7 Experiment Tracking with MLflow
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
DEFAULT_DATA_PATH = "data/clean/validated_data.csv"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_EXPERIMENT_NAME = "used_car_price_tier_classification"
DEFAULT_TRACKING_URI = "file:./mlruns"
PRICE_TIER_BINS = [0, 5000, 15000, float("inf")]
PRICE_TIER_LABELS = ["budget", "mid-range", "luxury"]


def derive_price_tier_from_price(price: pd.Series) -> pd.Series:
    """Create business target tiers from price according to proposal policy."""
    tier = pd.cut(
        price,
        bins=PRICE_TIER_BINS,
        labels=PRICE_TIER_LABELS,
        include_lowest=True,
        right=True,
    )
    return pd.Series(tier, index=price.index, dtype="object")


def business_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Business-oriented metrics for used car price-tier predictions."""
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)

    report = classification_report(y_true_series, y_pred_series, output_dict=True)
    luxury_recall = report.get("luxury", {}).get("recall", 0.0)

    severe_misclassification = (
        ((y_true_series == "budget") & (y_pred_series == "luxury"))
        | ((y_true_series == "luxury") & (y_pred_series == "budget"))
    ).mean()

    return {
        "luxury_recall": float(luxury_recall),
        "severe_misclassification_rate": float(severe_misclassification),
    }


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for mixed numeric/categorical features."""
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_model_configs() -> dict[str, dict]:
    """Define at least five models, including a baseline."""
    return {
        "baseline_dummy": {
            "estimator": DummyClassifier(),
            "param_distributions": {
                "model__strategy": ["most_frequent", "prior", "stratified"]
            },
            "n_iter": 3,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_distributions": {
                "model__C": np.logspace(-2, 1, 8),
                "model__penalty": ["l1", "l2"],
            },
            "n_iter": 8,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            "param_distributions": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 20, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "n_iter": 10,
        },
        "extra_trees": {
            "estimator": ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "param_distributions": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 20, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "n_iter": 10,
        },
        "linear_svc": {
            "estimator": LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_distributions": {
                "model__C": np.logspace(-3, 1, 9),
                "model__loss": ["hinge", "squared_hinge"],
            },
            "n_iter": 8,
        },
    }


def run_training(
    data_path: str,
    experiment_name: str,
    tracking_uri: str,
    reports_dir: str,
) -> None:
    """Train, tune, evaluate, and log all models to MLflow."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column to derive business target tiers.")

    df = df.copy()
    price_series = pd.Series(df["price"])
    df["price_tier_business"] = derive_price_tier_from_price(price_series)
    df = df.dropna(subset=["price_tier_business"]).copy()

    if "price_tier" in df.columns:
        source_tier = df["price_tier"].astype(str).str.lower().str.strip()
        target_tier = df["price_tier_business"].astype(str)
        mismatch_rate = (source_tier != target_tier).mean() * 100
        print(
            "Price-tier mismatch vs business thresholds "
            f"(<=5000, <=15000): {mismatch_rate:.2f}%"
        )

    target = df["price_tier_business"].astype(str)
    features = df.drop(columns=["price_tier", "price"], errors="ignore")

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    preprocess = build_preprocessor(x_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    configs = get_model_configs()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: list[dict] = []
    best_estimators: dict[str, Pipeline] = {}

    print(f"Dataset rows: {len(df):,}")
    print("Train/Val/Test:", len(x_train), len(x_val), len(x_test))

    for model_name, config in configs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", config["estimator"]),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["param_distributions"],
            n_iter=config["n_iter"],
            scoring={
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            },
            refit="macro_f1",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            return_train_score=False,
            verbose=1,
        )

        with mlflow.start_run(run_name=model_name):
            print(f"\nTraining model: {model_name}")
            search.fit(x_train, y_train)

            best_pipeline = search.best_estimator_
            best_estimators[model_name] = best_pipeline

            best_idx = search.best_index_
            cv_macro_f1 = search.cv_results_["mean_test_macro_f1"][best_idx]
            cv_bal_acc = search.cv_results_["mean_test_balanced_accuracy"][best_idx]

            y_val_pred = best_pipeline.predict(x_val)
            val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
            val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
            biz_metrics = business_metrics(y_val, y_val_pred)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_rows", len(x_train))
            mlflow.log_param("val_rows", len(x_val))
            mlflow.log_param("cv_folds", cv.get_n_splits())
            mlflow.log_param(
                "tier_policy",
                "budget<=5000,mid-range<=15000,luxury>15000",
            )
            for param_name, param_value in search.best_params_.items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_metrics(
                {
                    "cv_macro_f1": float(cv_macro_f1),
                    "cv_balanced_accuracy": float(cv_bal_acc),
                    "val_macro_f1": float(val_macro_f1),
                    "val_balanced_accuracy": float(val_bal_acc),
                    "val_luxury_recall": biz_metrics["luxury_recall"],
                    "val_severe_misclassification_rate": biz_metrics[
                        "severe_misclassification_rate"
                    ],
                }
            )

            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                report_file = tmp_path / f"{model_name}_val_classification_report.txt"
                report_text = classification_report(y_val, y_val_pred, zero_division=0)
                report_file.write_text(report_text, encoding="utf-8")

                labels = sorted(y_train.astype(str).unique().tolist())
                cm = confusion_matrix(y_val, y_val_pred, labels=labels)
                label_index = pd.Index(labels)
                cm_df = pd.DataFrame(cm, index=label_index, columns=label_index)
                cm_file = tmp_path / f"{model_name}_val_confusion_matrix.csv"
                cm_df.to_csv(cm_file, index=True)

                mlflow.log_artifact(str(report_file))
                mlflow.log_artifact(str(cm_file))

            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

            results.append(
                {
                    "model": model_name,
                    "cv_macro_f1": float(cv_macro_f1),
                    "cv_balanced_accuracy": float(cv_bal_acc),
                    "val_macro_f1": float(val_macro_f1),
                    "val_balanced_accuracy": float(val_bal_acc),
                    "val_luxury_recall": biz_metrics["luxury_recall"],
                    "val_severe_misclassification_rate": biz_metrics[
                        "severe_misclassification_rate"
                    ],
                }
            )

    results_df = pd.DataFrame(results).sort_values("cv_macro_f1", ascending=False)
    comparison_path = reports_path / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)
    print(f"\nSaved model comparison table: {comparison_path}")

    best_model_name = str(results_df.iloc[0]["model"])
    best_pipeline = clone(best_estimators[best_model_name])
    best_pipeline.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))

    y_test_pred = best_pipeline.predict(x_test)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_business = business_metrics(y_test, y_test_pred)

    test_summary = {
        "best_model": best_model_name,
        "test_macro_f1": float(test_macro_f1),
        "test_balanced_accuracy": float(test_bal_acc),
        "test_luxury_recall": test_business["luxury_recall"],
        "test_severe_misclassification_rate": test_business[
            "severe_misclassification_rate"
        ],
    }

    summary_path = reports_path / "best_model_test_metrics.json"
    summary_path.write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    print(f"Saved final test metrics: {summary_path}")
    print(f"Best model selected by CV: {best_model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model development and MLflow tracking for price-tier prediction."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to validated CSV with price_tier target.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (example: file:./mlruns).",
    )
    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
        help="Directory for result summary files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Input dataset not found: {args.data_path}")

    run_training(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        reports_dir=args.reports_dir,
    )
