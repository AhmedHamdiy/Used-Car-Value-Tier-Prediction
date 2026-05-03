# Modeling Rationale And Reporting Notes

This document converts the observed outcomes from `notebooks/eda.ipynb` and `notebooks/clean_data.ipynb` into explicit modeling rationale for `notebooks/modeling.ipynb` (per `docs/project_requirements.pdf`).

## Data Snapshot (Post-Cleaning)

Source used by modeling: `data/processed/clean_data.csv`.

- Shape: 253,776 rows x 12 columns.
- Numeric sanity checks (from the cleaned dataset):
  - `price`: min=500, p50=3,999, p90=16,490, p99=40,000, max=1,250,000
  - `kilometer`: min=0, p50=150,000, p90=150,000, p99=150,000, max=225,000
  - `power`: min=5, p50=118, p90=204, p99=258.5, max=258.5
  - `yearOfRegistration`: min=1910, p50=2005, p90=2013, p99=2018, max=2023
- Cardinalities (motivation for encoding choices):
  - `brand`: 73 unique values
  - `model`: 717 unique values
  - `vehicleType`: 8 unique values
  - `fuelType`: 7 unique values
  - `gearbox`: 3 unique values
  - `seller`: 2 unique values
  - `dataSource`: 2 unique values

## Target Variable (Tiering) And Class Imbalance

Modeling target in `notebooks/modeling.ipynb` is derived from `price`:

- `budget` (< 5,000) -> 0
- `mid-range` (5,000 to < 15,000) -> 1
- `luxury` (>= 15,000) -> 2

Observed class distribution on `data/processed/clean_data.csv` under these thresholds:

- budget: 57.13%
- mid-range: 31.16%
- luxury: 11.71%

Rationale implications:

- This is an imbalanced 3-class problem; optimizing plain accuracy would over-reward majority classes.
- Use stratified splitting and prioritize metrics that treat classes more evenly (macro-F1, balanced accuracy).
- Compare balancing strategies (none vs SMOTE vs undersampling) because luxury recall matters for stakeholder decisions.

## Cleaning Rationale (Why The Data Is Trusted)

The cleaning logic lives in `src/data/clean_data.py` and the cleaned output is inspected in `notebooks/clean_data.ipynb`.

- Placeholder handling: common placeholder strings ("unknown", "keine_angabe", "-", etc.) are normalized to missing values.
- Label standardization: aliases unify values across sources (e.g., brand/model/fuelType/vehicleType/seller/gearbox).
- Domain constraints / capping: numeric columns are bounded using domain rules (e.g., year range, km range, minimum price/power).

Modeling implication:

- After cleaning, we can safely apply global preprocessing transforms (imputation/scaling/encoding) without model instability due to obvious schema violations.

## EDA Findings That Drive Modeling Decisions

The EDA work (visual summaries in `notebooks/eda.ipynb` and numeric checks above) supports the following modeling choices.

### Tier-Wise Market Differences (Signal Exists)

On `data/processed/clean_data.csv` using the tier thresholds above:

- Budget (n=144,982): median price=1,999; median year=2001; median km=150,000; median power=102
- Mid-range (n=79,070): median price=8,250; median year=2008; median km=125,000; median power=140
- Luxury (n=29,724): median price=21,000; median year=2011; median km=70,000; median power=184

Rationale implications:

- Luxury listings are newer, lower-mileage, and higher-power on average; these patterns justify engineered features that combine age/usage/power.

### Monotonic Relationships (Useful For Both Linear And Tree Models)

Spearman correlations on the cleaned dataset:

- `price` vs `yearOfRegistration`: +0.616 (newer cars cost more)
- `price` vs `kilometer`: -0.430 (higher mileage associates with lower price)
- `price` vs `power`: +0.535 (higher power associates with higher price)

Rationale implications:

- Supports adding `vehicleAge`, `kmPerYear`, and ratio features (`km_power_ratio`, `power_per_age`) to help models capture these drivers.
- Suggests non-linear effects (e.g., mileage impact depends on age), motivating tree ensembles / boosting in the model shortlist.

## Validation Split And Leakage Controls

- Split strategy: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
- Rationale:
  - Stratification preserves class proportions between train and test, which is critical for imbalanced multi-class problems.
- Leakage controls:
  - Oversampling/undersampling is applied inside the pipeline as a step after preprocessing/feature selection and before the model, so it is performed only on training folds during cross-validation (not on the held-out test set).

## Preprocessing And Transformation Rationale

Implemented in `build_preprocess()` (a `ColumnTransformer`).

### Numeric Features

- Imputation: `SimpleImputer(strategy="median")`.
  - Rationale: median is robust to skewed distributions and outliers that are common in price-related datasets.
  - Note from cleaned data: core numeric columns have 0 missing values post-cleaning, but the imputer is kept for pipeline robustness and to guard against schema drift.
- Scaling: `StandardScaler()`.
  - Rationale: improves optimization and comparability for linear models (Logistic Regression, Linear SVC) and prevents features with large magnitudes from dominating.
  - Note: scaling is not strictly required for tree models, but keeping a single consistent preprocessing block simplifies experimentation.

### Categorical Features

- Imputation: `SimpleImputer(strategy="most_frequent")`.
  - Rationale: mode imputation is a simple, stable default for categorical missingness without introducing new categories.
- Encoding: `OneHotEncoder(handle_unknown="ignore", drop="first")`.
  - Rationale:
    - One-hot is a strong baseline for mixed-type tabular data.
    - `handle_unknown="ignore"` prevents runtime failures when unseen categories appear in validation/test.
    - `drop="first"` reduces redundant columns for linear models (mitigates multicollinearity).

### High-Cardinality Columns (`brand`, `model`)

- Approach: frequency encoding via `FrequencyEncoder(["brand", "model"])`, then the raw `brand` and `model` are dropped before one-hot.
- Rationale:
  - `brand` and especially `model` can have many distinct values; one-hot can explode dimensionality and hurt runtime and generalization.
  - Observed cardinalities after cleaning: `brand`=73, `model`=717.
  - Frequency features capture popularity/rarity information in a compact way and are robust to previously unseen categories (unseen -> 0).

## Feature Engineering Rationale

- `vehicleAge = current_year - yearOfRegistration` (clipped at 0).
  - Rationale: age is one of the strongest real-world determinants of used car value; it also converts a calendar year into a more directly meaningful feature.
- `kmPerYear = kilometer / (vehicleAge + 1)`.
  - Rationale: normalizes mileage by age to represent usage intensity; two cars with the same mileage but different ages should not be treated equivalently.
- `km_power_ratio = kilometer / (power + 1)`.
  - Rationale: interaction proxy between usage and engine power; helps models represent non-linear relationships without explicit polynomial expansion.
- `log_kilometer = log1p(kilometer)` and `log_power = log1p(power)`.
  - Rationale: mileage and power are right-skewed; log transforms compress outliers and can make class boundaries smoother.
  - Note from cleaned data: `kilometer` is concentrated at common discrete values (p50=p90=p99=150,000), so log transforms alone are not enough; ratio features like `kmPerYear` and `km_power_ratio` help recover additional signal.
- `power_per_age = power / (vehicleAge + 1)`.
  - Rationale: proxy for "performance relative to age"; may separate newer high-power listings from older high-power cars.
- `is_vintage = vehicleAge > vintage_age_years`.
  - Rationale: very old vehicles behave differently in the market (collectibles, condition-driven pricing) and benefit from an explicit regime indicator.
- `is_automatic` from `gearbox`.
  - Rationale: automatic transmission often correlates with tier/price; encoding it as a binary feature makes it usable by all models.

Implementation note: `yearOfRegistration` is dropped after engineering to avoid duplicate representations.

## Feature Selection Rationale

Optional step enabled by `use_feature_selection=True` in the notebook:

- Method: `SelectFromModel(RandomForestClassifier(...), threshold="median")`.
- Rationale:
  - Removes low-importance features after one-hot expansion, reducing noise and runtime.
  - A tree-based selector captures non-linear relevance and interactions better than simple variance filters.
  - `threshold="median"` is a simple, repeatable rule that avoids tuning an additional hyperparameter.
- Trade-offs:
  - May discard rare-but-informative categories.
  - Adds training time (fits an additional model) and can complicate interpretability.

## Data Balancing Strategy Rationale

Compared strategies: `none`, `smote`, `undersample`.

- Why compare:
  - The dataset is typically imbalanced across tiers; minority performance (especially `luxury`) matters for stakeholder decisions.
  - Different balancing methods shift the bias-variance trade-off differently.
- SMOTE (`SMOTE`):
  - Pros: improves minority recall by creating synthetic samples.
  - Cons: can create borderline/noisy samples; may increase confusion between adjacent tiers.
- Undersampling (`RandomUnderSampler`):
  - Pros: simpler, avoids synthetic points; can speed up training.
  - Cons: discards data; can hurt majority-class calibration and overall performance.

## Model Candidate Selection Rationale

The notebook evaluates at least five models plus a baseline:

- Baseline: `DummyClassifier`.
  - Rationale: sets a minimum expected performance; ensures improvements are meaningful.
- Linear models: `LogisticRegression(solver="saga")`, `LinearSVC`.
  - Rationale: strong baselines for tabular data, fast to train, often robust with one-hot features.
- Single tree: `DecisionTreeClassifier`.
  - Rationale: interpretable non-linear baseline; highlights whether simple rules can separate tiers.
- Bagging ensembles: `RandomForestClassifier`, `ExtraTreesClassifier`.
  - Rationale: capture non-linearities and interactions with good out-of-the-box performance.
- Gradient boosting: `XGBClassifier`.
  - Rationale: often state-of-the-art for structured/tabular data; handles complex patterns and feature interactions.

## Hyperparameter Search And Validation Rationale

- Search method: `RandomizedSearchCV`.
  - Rationale: more compute-efficient than exhaustive grid search, especially across multiple models and strategies.
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  - Rationale: preserves class proportions in each fold to stabilize estimates for minority classes.
- Refit metric: `macro_f1`.
  - Rationale: treats each class equally; avoids optimizing only for the majority class.

## Metrics And Business-Oriented Evaluation

### Standard Metrics (Logged)

- Macro F1 (`f1_macro`):
  - Rationale: averages F1 across classes, reflecting balanced performance across tiers.
- Balanced accuracy:
  - Rationale: accounts for class imbalance by averaging recall across classes.

### Business Metrics (Logged)

- `luxury_recall`:
  - Rationale: missing luxury listings can be costly (lost upsell, wrong prioritization, incorrect premium routing). Recall directly measures how often luxury is correctly detected.
- `severe_misclassification_rate`:
  - Defined as predicting `budget` as `luxury` or `luxury` as `budget`.
  - Rationale: these errors represent the highest business risk because they cause the most extreme decision mismatch.

## Model Selection Decision Rule (Recommended)

Use a multi-criteria decision rule instead of a single metric:

1) Primary: maximize CV `macro_f1` (generalization across tiers).
2) Constraint: keep `severe_misclassification_rate` low on the test set.
3) Tie-breakers: higher `luxury_recall`, simpler model (latency/maintainability), and stability (small CV-test gap).

## Reporting Artifacts To Include (Per Requirements)

The notebook already writes classification reports and logs confusion matrices to MLflow. For the final reporting package, include:

- A single comparison table (all models x strategies x feature-selection) with:
  - CV `macro_f1`, CV balanced accuracy
  - Train vs test `macro_f1` and balanced accuracy (overfitting check)
  - `luxury_recall` and `severe_misclassification_rate`
- Error analysis for the chosen best model:
  - Confusion matrix interpretation (which tier confusions dominate)
  - Examples of typical failure modes (e.g., very old low-mileage cars, high-mileage newer cars)
- Stakeholder interpretation:
  - What actions each predicted tier enables (e.g., prioritize review, adjust pricing band, route to premium sales)
- MLflow evidence:
  - Screenshot of MLflow experiment comparison showing all runs side by side.
