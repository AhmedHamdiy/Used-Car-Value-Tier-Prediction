from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features.encoding import FrequencyEncoder
from src.features.engineering import FeatureEngineer


DEFAULT_DATA_PATH = "data/processed/validated_data.csv"
DEFAULT_EXPERIMENT_NAME = "used_car_price_tier_classification"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
DEFAULT_REPORTS_DIR = "reports/results"
DEFAULT_MODEL_DIR = "models"
RANDOM_STATE = 42

TIER_TO_INT = {"budget": 0, "mid-range": 1, "luxury": 2}


def business_metrics(y_true, y_pred) -> dict[str, float]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    luxury_recall = float(report.get("2", {}).get("recall", 0.0))

    y_t = pd.Series(y_true).reset_index(drop=True)
    y_p = pd.Series(y_pred).reset_index(drop=True)
    severe = ((y_t == 0) & (y_p == 2)) | ((y_t == 2) & (y_p == 0))
    severe_rate = float(severe.mean())

    return {
        "luxury_recall": luxury_recall,
        "severe_misclassification_rate": severe_rate,
    }


def get_model_configs() -> dict[str, dict]:
    return {
        "baseline_dummy": {
            "estimator": DummyClassifier(),
            "param_distributions": {
                "model__strategy": ["most_frequent", "prior", "stratified"]
            },
            "n_iter": 3,
        },
        "log_reg": {
            "estimator": LogisticRegression(
                max_iter=3000,
                solver="saga",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_distributions": {
                "model__C": np.logspace(-2, 1, 8),
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 10,
        },
        "linear_svc": {
            "estimator": LinearSVC(random_state=RANDOM_STATE),
            "param_distributions": {
                "model__C": np.logspace(-3, 2, 8),
                "model__loss": ["hinge", "squared_hinge"],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 10,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_distributions": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
                "model__class_weight": [None, "balanced_subsample"],
            },
            "n_iter": 10,
        },
        "extra_trees": {
            "estimator": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_distributions": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 10,
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_distributions": {
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__criterion": ["gini", "entropy"],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 10,
        },
        "xgboost": {
            "estimator": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="mlogloss",
            ),
            "param_distributions": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
                "model__gamma": [0, 1, 5],
            },
            "n_iter": 10,
        },
    }


def build_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", drop="first"),
                        ),
                    ]
                ),
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="drop",
    )


def build_pipeline(
    estimator,
    preprocess: ColumnTransformer,
    strategy: str,
    current_year: int | None,
    use_feature_selection: bool,
):
    steps = [
        ("fe", FeatureEngineer(current_year=current_year)),
        ("freq", FrequencyEncoder(["brand", "model"])),
        (
            "drop_cols",
            FunctionTransformer(
                lambda X: X.drop(columns=["brand", "model"], errors="ignore"),
                validate=False,
            ),
        ),
        ("preprocess", preprocess),
    ]

    if use_feature_selection:
        selector = SelectFromModel(
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            threshold="median",
        )
        steps.append(("feature_selection", selector))

    if strategy == "smote":
        steps.append(("resample", SMOTE(random_state=RANDOM_STATE)))
    elif strategy == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=RANDOM_STATE)))
    elif strategy != "none":
        raise ValueError("strategy must be one of: none, smote, undersample")

    steps.append(("model", estimator))
    return Pipeline(steps)


def run_experiments(
    model_configs: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str,
    preprocess: ColumnTransformer,
    cv: StratifiedKFold,
    current_year: int | None,
    use_feature_selection: bool,
    reports_dir: Path,
    model_dir: Path,
):
    results: list[dict] = []
    best_estimators: dict[str, Pipeline] = {}

    for model_name, cfg in model_configs.items():
        pipe = build_pipeline(
            cfg["estimator"],
            preprocess=preprocess,
            strategy=strategy,
            current_year=current_year,
            use_feature_selection=use_feature_selection,
        )

        search = RandomizedSearchCV(
            pipe,
            cfg["param_distributions"],
            n_iter=cfg["n_iter"],
            scoring={
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            },
            refit="macro_f1",
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

        with mlflow.start_run(run_name=f"{model_name}_{strategy}"):
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_idx = int(search.best_index_)
            cv_f1 = float(search.cv_results_["mean_test_macro_f1"][best_idx])
            cv_bal = float(search.cv_results_["mean_test_balanced_accuracy"][best_idx])

            y_pred_train = best_model.predict(X_train)
            train_macro_f1 = float(f1_score(y_train, y_pred_train, average="macro"))
            train_bal_acc = float(balanced_accuracy_score(y_train, y_pred_train))
            train_biz = business_metrics(y_train, y_pred_train)

            y_pred = best_model.predict(X_test)
            test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
            test_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            biz = business_metrics(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{model_name} - {strategy}")
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches="tight")
                mlflow.log_artifact(tmp.name, "confusion_matrix")
            plt.close()

            report_text = classification_report(y_test, y_pred, zero_division=0)
            report_path = (
                reports_dir / f"classification_report_{model_name}_{strategy}.txt"
            )
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path), "reports")

            train_report_text = classification_report(
                y_train, y_pred_train, zero_division=0
            )
            train_report_path = (
                reports_dir / f"classification_report_train_{model_name}_{strategy}.txt"
            )
            train_report_path.write_text(train_report_text, encoding="utf-8")
            mlflow.log_artifact(str(train_report_path), "reports")

            mlflow.log_param("model", model_name)
            mlflow.log_param("strategy", strategy)
            mlflow.log_param(
                "current_year", current_year if current_year is not None else "system"
            )
            mlflow.log_param("feature_selection", bool(use_feature_selection))
            for k, v in search.best_params_.items():
                mlflow.log_param(k, v)
            mlflow.log_param("cv_folds", cv.get_n_splits())

            mlflow.log_metric("cv_macro_f1", cv_f1)
            mlflow.log_metric("cv_balanced_accuracy", cv_bal)
            mlflow.log_metric("train_macro_f1", train_macro_f1)
            mlflow.log_metric("train_balanced_accuracy", train_bal_acc)
            mlflow.log_metric("train_luxury_recall", train_biz["luxury_recall"])
            mlflow.log_metric(
                "train_severe_misclassification_rate",
                train_biz["severe_misclassification_rate"],
            )
            mlflow.log_metric("test_macro_f1", test_macro_f1)
            mlflow.log_metric("test_balanced_accuracy", test_bal_acc)
            mlflow.log_metric("test_luxury_recall", biz["luxury_recall"])
            mlflow.log_metric(
                "test_severe_misclassification_rate",
                biz["severe_misclassification_rate"],
            )

            mlflow.sklearn.log_model(best_model, "model")

            model_out = model_dir / f"{model_name}_{strategy}.pkl"
            mlflow.sklearn.save_model(best_model, path=str(model_out))

            results.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "cv_macro_f1": cv_f1,
                    "cv_balanced_accuracy": cv_bal,
                    "train_macro_f1": train_macro_f1,
                    "train_balanced_accuracy": train_bal_acc,
                    "train_luxury_recall": train_biz["luxury_recall"],
                    "train_severe_misclassification_rate": train_biz[
                        "severe_misclassification_rate"
                    ],
                    "test_macro_f1": test_macro_f1,
                    "test_balanced_accuracy": test_bal_acc,
                    "test_luxury_recall": biz["luxury_recall"],
                    "test_severe_misclassification_rate": biz[
                        "severe_misclassification_rate"
                    ],
                }
            )
            best_estimators[f"{model_name}_{strategy}"] = best_model

    return pd.DataFrame(results), best_estimators


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated model training + MLflow tracking"
    )
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument(
        "--strategy", default="none", choices=["none", "smote", "undersample"]
    )
    p.add_argument("--feature-selection", action="store_true")
    p.add_argument("--current-year", type=int, default=2026)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global RANDOM_STATE
    RANDOM_STATE = int(args.random_state)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "price_tier" not in df.columns:
        raise ValueError("Expected 'price_tier' column in validated dataset")

    df = df.copy()
    df["price_tier"] = df["price_tier"].astype(str).str.lower().map(TIER_TO_INT)
    df = df.dropna(subset=["price_tier"]).copy()

    X = df.drop(columns=["price_tier"], errors="ignore")
    y = df["price_tier"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocess = build_preprocess()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.end_run()

    model_configs = get_model_configs()
    results_df, best_estimators = run_experiments(
        model_configs=model_configs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        strategy=str(args.strategy),
        preprocess=preprocess,
        cv=cv,
        current_year=int(args.current_year) if args.current_year else None,
        use_feature_selection=bool(args.feature_selection),
        reports_dir=reports_dir,
        model_dir=model_dir,
    )

    out_csv = reports_dir / f"model_results_{args.strategy}.csv"
    results_df = results_df.sort_values(["test_macro_f1"], ascending=False)
    results_df.to_csv(out_csv, index=False)

    # 5.9 Results & Evaluation: pick best model and write evaluation artifacts
    best_row = results_df.iloc[0].to_dict()
    best_key = f"{best_row['model']}_{best_row['strategy']}"
    best_pipeline = best_estimators[best_key]

    y_test_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    cm_path = reports_dir / f"best_model_confusion_matrix_{args.strategy}.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False)

    # Error analysis focused on severe budget<->luxury mistakes
    y_test_s = pd.Series(y_test).reset_index(drop=True)
    y_pred_s = pd.Series(y_test_pred).reset_index(drop=True)
    severe_mask = ((y_test_s == 0) & (y_pred_s == 2)) | (
        (y_test_s == 2) & (y_pred_s == 0)
    )
    severe_examples_path = reports_dir / f"severe_errors_{args.strategy}.csv"
    severe_examples = X_test.reset_index(drop=True).copy()
    severe_examples["y_true"] = y_test_s
    severe_examples["y_pred"] = y_pred_s
    severe_examples = severe_examples.loc[severe_mask].head(5000)
    severe_examples.to_csv(severe_examples_path, index=False)

    evaluation_summary = {
        "best_model": str(best_row["model"]),
        "best_strategy": str(best_row["strategy"]),
        "selection_metric": "test_macro_f1",
        "cv_macro_f1": float(best_row["cv_macro_f1"]),
        "cv_balanced_accuracy": float(best_row["cv_balanced_accuracy"]),
        "train_macro_f1": float(best_row["train_macro_f1"]),
        "train_balanced_accuracy": float(best_row["train_balanced_accuracy"]),
        "test_macro_f1": float(best_row["test_macro_f1"]),
        "test_balanced_accuracy": float(best_row["test_balanced_accuracy"]),
        "test_luxury_recall": float(best_row["test_luxury_recall"]),
        "test_severe_misclassification_rate": float(
            best_row["test_severe_misclassification_rate"]
        ),
        "overfit_gap_train_test_macro_f1": float(
            best_row["train_macro_f1"] - best_row["test_macro_f1"]
        ),
        "severe_error_count": int(severe_mask.sum()),
        "severe_error_rate": float(severe_mask.mean()),
    }
    best_summary_path = reports_dir / f"best_model_summary_{args.strategy}.json"
    best_summary_path.write_text(
        json.dumps(evaluation_summary, indent=2),
        encoding="utf-8",
    )

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "rows": int(len(df)),
        "strategy": args.strategy,
        "feature_selection": bool(args.feature_selection),
        "current_year": int(args.current_year),
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "results_csv": str(out_csv),
        "best_model_summary": str(best_summary_path),
        "best_model_confusion_matrix": str(cm_path),
        "severe_errors_csv": str(severe_examples_path),
    }
    (reports_dir / f"run_metadata_{args.strategy}.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results table to: {out_csv}")


if __name__ == "__main__":
    main()
