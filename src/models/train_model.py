import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGBoost model will be skipped.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Resampling will be skipped.")

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Logging will be disabled.")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to engineer features from car data."""

    def __init__(self, current_year: Optional[int] = None):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        year = pd.to_numeric(X.get("yearOfRegistration"), errors="coerce")
        km = pd.to_numeric(X.get("kilometer"), errors="coerce")
        power = pd.to_numeric(X.get("power"), errors="coerce")

        cy = self.current_year or datetime.now().year
        age = cy - year

        X["vehicleAge"] = age
        X["kmPerYear"] = km / (age + 1)
        X["km_power_ratio"] = km / (power + 1)
        X["log_km"] = np.log1p(km)

        return X.drop(columns=["yearOfRegistration"], errors="ignore")


class DropColumns(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified columns."""

    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols, errors="ignore")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer to encode categorical variables by frequency."""

    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.freq_maps_ = {}

        for col in self.cols:
            if col in X_df.columns:
                self.freq_maps_[col] = X_df[col].value_counts(normalize=True)
            else:
                self.freq_maps_[col] = pd.Series(dtype=float)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        for col in self.cols:
            freq = self.freq_maps_.get(col)

            if col in X_df.columns:
                X_df[f"{col}_freq"] = X_df[col].map(freq).fillna(0.0)
            else:
                X_df[f"{col}_freq"] = 0.0

        return X_df


def calculate_business_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate business-specific metrics for price tier prediction.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with luxury_recall and severe_misclassification_rate
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    luxury_recall = float(report.get("2", {}).get("recall", 0.0))

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    severe = ((y_true == 0) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 0))
    severe_rate = float(severe.mean())

    return {
        "luxury_recall": luxury_recall,
        "severe_misclassification_rate": severe_rate,
    }


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration for all models to be trained.

    Returns:
        Dictionary mapping model names to their configurations
    """
    configs = {
        "baseline": {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "params": {},
            "n_iter": 1,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
            "n_iter": 10,
        },
        "extra_trees": {
            "estimator": ExtraTreesClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "n_iter": 10,
        },
        "log_reg": {
            "estimator": LogisticRegression(solver="saga", max_iter=5000),
            "params": {
                "C": np.logspace(-2, 1, 5),
            },
            "n_iter": 5,
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"],
                "max_features": [None, "sqrt", "log2"],
            },
            "n_iter": 10,
        },
        "linear_svm": {
            "estimator": LinearSVC(random_state=42, max_iter=5000),
            "params": {
                "C": np.logspace(-3, 2, 6),
                "loss": ["hinge", "squared_hinge"],
            },
            "n_iter": 10,
        },
    }

    if XGBOOST_AVAILABLE:
        configs["xgboost"] = {
            "estimator": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                tree_method="hist",
                n_jobs=1,
                random_state=42,
            ),
            "params": {
                "n_estimators": [200, 300, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
            "n_iter": 10,
        }

    return configs


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "price_tier",
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
    current_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Prepare data for training by applying feature engineering and train/val/test split.

    Args:
        df: Input dataframe with features and target
        target_col: Name of the target column
        drop_cols: Columns to drop (defaults to [target_col, "price"])
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        current_year: Current year for age calculation

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if drop_cols is None:
        drop_cols = [target_col, "price"]

    # Map price tier to numeric
    df = df.copy()
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.lower()
        .map({"budget": 0, "mid-range": 1, "luxury": 2})
    )

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].astype(int)

    # Feature engineering
    fe = FeatureEngineer(current_year=current_year or datetime.now().year)
    freq = FrequencyEncoder(cols=["brand", "model"])
    dropper = DropColumns(["brand", "model"])

    X = fe.fit_transform(X)
    X = freq.fit_transform(X)
    X = dropper.fit_transform(X)

    # Train/val/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    # One-hot encode categorical variables
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_resampled_datasets(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Create resampled datasets using SMOTE and undersampling.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Dictionary with dataset types as keys and (X, y) tuples as values
    """
    datasets = {"none": (X_train, y_train)}

    if IMBLEARN_AVAILABLE:
        # SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        datasets["smote"] = (X_train_smote, y_train_smote)

        # Undersampling
        rus = RandomUnderSampler(random_state=random_state)
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
        datasets["undersample"] = (X_train_us, y_train_us)

    return datasets


def train_model(
    model_name: str,
    dataset_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_configs: Optional[Dict] = None,
    experiment_name: str = "used_car_price_tier",
    mlflow_tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a single model configuration and evaluate on test set.

    Args:
        model_name: Name of the model to train
        dataset_type: Type of dataset ("none", "smote", "undersample")
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        model_configs: Model configuration dictionary
        experiment_name: MLflow experiment name
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        Dictionary with training results and metrics
    """
    if model_configs is None:
        model_configs = get_model_configs()

    cfg = model_configs[model_name]

    run_name = f"{model_name}_{dataset_type}"
    print(f"\nRunning: {run_name}")

    best_model = None
    best_score = -1
    best_params = None

    param_list = list(
        ParameterSampler(cfg["params"], n_iter=cfg["n_iter"], random_state=42)
    )

    # Train without MLflow if not available
    if not MLFLOW_AVAILABLE or mlflow_tracking_uri is None:
        for params in param_list:
            model = cfg["estimator"].set_params(**params)
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            score = f1_score(y_val, val_pred, average="macro")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

        test_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        test_bal = balanced_accuracy_score(y_test, test_pred)
        biz_metrics = calculate_business_metrics(y_test, test_pred)

        print(f"Done: {run_name} | val_f1={best_score:.4f} | test_f1={test_f1:.4f}")

        return {
            "model": model_name,
            "dataset": dataset_type,
            "val_f1": best_score,
            "test_f1": test_f1,
            "test_bal_acc": test_bal,
            "luxury_recall": biz_metrics["luxury_recall"],
            "severe_misclassification_rate": biz_metrics[
                "severe_misclassification_rate"
            ],
            "best_params": best_params,
        }

    # Train with MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Search for best hyperparameters
        for params in param_list:
            model = cfg["estimator"].set_params(**params)

            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)

            score = f1_score(y_val, val_pred, average="macro")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

        # Test evaluation
        test_pred = best_model.predict(X_test)

        test_f1 = f1_score(y_test, test_pred, average="macro")
        test_bal = balanced_accuracy_score(y_test, test_pred)

        # Business metrics
        biz_metrics = calculate_business_metrics(y_test, test_pred)

        # MLflow logging
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset_type", dataset_type)
        mlflow.log_params(best_params)

        mlflow.log_metric("val_f1", best_score)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_balanced_acc", test_bal)

        mlflow.log_metric("luxury_recall", biz_metrics["luxury_recall"])
        mlflow.log_metric(
            "severe_misclassification_rate",
            biz_metrics["severe_misclassification_rate"],
        )

        mlflow.sklearn.log_model(best_model, "model")

    print(f"Done: {run_name} | val_f1={best_score:.4f} | test_f1={test_f1:.4f}")

    return {
        "model": model_name,
        "dataset": dataset_type,
        "val_f1": best_score,
        "test_f1": test_f1,
        "test_bal_acc": test_bal,
        "luxury_recall": biz_metrics["luxury_recall"],
        "severe_misclassification_rate": biz_metrics["severe_misclassification_rate"],
        "best_params": best_params,
    }


def train_all_models(
    data_path: str,
    output_path: Optional[str] = None,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    experiment_name: str = "used_car_price_tier",
    mlflow_tracking_uri: Optional[str] = None,
    feature_selection_strategy: Optional[str] = None,
    feature_selection_threshold: Optional[float] = None,
    feature_selection_k: Optional[int] = None,
    feature_selection_report_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Train all specified models on all specified datasets.

    Args:
        data_path: Path to the cleaned CSV data file
        output_path: Optional path to save results CSV
        models: List of model names to train (default: all available)
        datasets: List of dataset types (default: ["none", "smote", "undersample"])
        experiment_name: MLflow experiment name
        mlflow_tracking_uri: MLflow tracking URI
        feature_selection_strategy: Feature selection strategy (xgboost, variance, correlation, k_best, mutual_info)
        feature_selection_threshold: Threshold for feature selection
        feature_selection_k: Number of features to select (for k_best, mutual_info)
        feature_selection_report_path: Path to save feature selection report

    Returns:
        DataFrame with all results
    """
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    # Prepare data
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # Apply feature selection if requested
    feature_selector = None
    if feature_selection_strategy:
        try:
            from src.features.feature_selection import select_features

            print(f"\nApplying feature selection: {feature_selection_strategy}")
            X_train, X_val, X_test, feature_selector = select_features(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                X_test=X_test,
                strategy=feature_selection_strategy,
                threshold=feature_selection_threshold,
                k=feature_selection_k,
                random_state=42,
            )

            # Generate and save feature selection report
            if feature_selector:
                report = feature_selector.get_feature_importance_report()
                print(report)

                if feature_selection_report_path:
                    feature_selector.export_selected_features(
                        feature_selection_report_path
                    )
        except Exception as e:
            print(f"Warning: Feature selection failed: {e}")
            print("Continuing without feature selection...")

    # Create resampled datasets
    print("Creating resampled datasets...")
    datasets_dict = create_resampled_datasets(X_train, y_train)

    # Get model configs
    model_configs = get_model_configs()

    if models is None:
        models = list(model_configs.keys())
    if datasets is None:
        datasets = list(datasets_dict.keys())

    # Train all models
    all_results = []

    for model_name in models:
        if model_name not in model_configs:
            print(f"Warning: Model {model_name} not found in configs. Skipping.")
            continue

        for dataset_type in datasets:
            if dataset_type not in datasets_dict:
                print(f"Warning: Dataset {dataset_type} not available. Skipping.")
                continue

            X_tr, y_tr = datasets_dict[dataset_type]

            result = train_model(
                model_name=model_name,
                dataset_type=dataset_type,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                model_configs=model_configs,
                experiment_name=experiment_name,
                mlflow_tracking_uri=mlflow_tracking_uri,
            )

            all_results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("test_f1", ascending=False)

    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(results_df.to_string())
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/processed/clean_data.csv"

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = "reports/model_comparison.csv"

    train_all_models(
        data_path=data_path,
        output_path=output_path,
        mlflow_tracking_uri="http://127.0.0.1:5000",
    )
