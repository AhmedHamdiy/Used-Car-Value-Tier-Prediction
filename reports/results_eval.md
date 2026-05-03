# 5.9 Results & Evaluation

## 1. Train vs Test Performance (Overfitting Check)

Across all models, **validation F1 and test F1 are very close**, which indicates **no major overfitting**.

* Example:

  * XGBoost (none):

    * Val F1 = 0.8591
    * Test F1 = 0.8590

### Conclusion

* Tree-based models generalize well.
* Logistic regression underfits (low scores overall).
* No extreme overfitting detected.

---

## 2. Model Comparison (Standard + Business Metrics)

### Best overall models

| Model                 | Test F1    | Test Balanced Acc | Luxury Recall            | Severe Error Rate |
| --------------------- | ---------- | ----------------- | ------------------------ | ----------------- |
| **XGBoost (none)**    | **0.8590** | 0.8566            | 0.8611                   | **0.0022**        |
| XGBoost (smote)       | 0.8560     | **0.8580**        | **0.8857**               | 0.0025            |
| XGBoost (undersample) | 0.8529     | 0.8586            | **0.9053 (best recall)** | 0.0028            |

---

## 3. Key Observations

### Best model overall

* **XGBoost (no resampling)** is the best trade-off:

  * Highest **F1 (0.859)**
  * Very strong **balanced accuracy**
  * Extremely low **severe misclassification rate (~0.22%)**

---

### Effect of SMOTE / undersampling

* SMOTE and undersampling:

  * Increase **luxury recall** (important class)
  * Slightly reduce overall F1
  * Slightly increase error rate

* Interpretation:

  * Resampling helps minority class detection
  * But slightly hurts global accuracy

---

### Weak models

#### Logistic Regression

* F1 ≈ 0.66 (low)
* Severe error rate very high (~2.1–2.7%)
* Struggles with nonlinear relationships

#### Baseline

* F1 ≈ 0.21 → useless baseline
* Confirms dataset is learnable

---

## 4. Business Metric Interpretation

### (A) Luxury Recall

* Measures: how well we detect luxury cars

Best:

* XGBoost undersample → **0.905**
* XGBoost SMOTE → **0.886**
* XGBoost none → **0.861**

* Meaning:

* If business cares about luxury cars:

  * SMOTE / undersampling may be preferred

---

### (B) Severe Misclassification Rate

(critical business risk: predicting budget as luxury or vice versa)

Best:

* XGBoost (none): **0.0022 (best)**
* All XGBoost variants: ~0.002–0.003

* Meaning:

* Model is very safe for business decisions
* Very few dangerous misclassifications

---

## 5. Error Analysis (Where model fails)

Best model: **XGBoost (none)**

### Main failure cases

1. **Mid-range vs luxury confusion**

   * hardest boundary
   * overlapping feature distributions

2. Rare cases:

   * luxury cars with low mileage but old age
   * budget cars with high power (outliers)

3. Most errors are:

   * class 1 ↔ class 2 (not extreme mistakes)

* Important insight:

* Model almost never confuses **budget ↔ luxury**, which is critical

---

## 6. Business Interpretation

### What the model enables

* Reliable segmentation of used car prices into tiers
* Very low risk of catastrophic pricing errors
* Strong detection of luxury segment

### Business actions

* Pricing automation system can be trusted for:

  * listing recommendations
  * market segmentation
* SMOTE variant can be used if:

  * luxury detection is priority (marketing / premium targeting)

---

## 7. Unsuccessful Approaches

### Logistic Regression

* Underfits complex relationships
* Poor F1 and recall
* Abandoned for final deployment

### SMOTE / Undersampling (partial success)

* Improved luxury recall
* Slightly reduced overall performance
* Not chosen as best global model

---

## Final Conclusion

* **Best model overall: XGBoost (no resampling)**
* Best trade-off between:

  * accuracy
  * stability
  * business safety (low severe errors)