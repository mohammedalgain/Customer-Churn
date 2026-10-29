# Customer Churn Prediction

A three-phase machine learning project on a synthetic customer churn dataset (3,000
customers, 24 original features): exploratory analysis and feature engineering, model
comparison, and hyperparameter tuning of the winning model.


## Results

| Metric | Baseline Logistic Regression | Tuned Final Model |
|---|---|---|
| Accuracy | 0.667 | 0.653 |
| Precision (churn) | 0.564 | 0.516 |
| **Recall (churn)** | 0.346 | **0.682** |
| **F1-score (churn)** | 0.429 | **0.587** |
| AUC-ROC | 0.701 | 0.700 |

The single highest-impact change was `class_weight="balanced"`, which roughly doubled
Recall. On an imbalanced dataset (63.8% retained / 36.2% churned) where a missed
churner is a lost customer, that trade-off — a bit less accuracy and precision for much
better churn detection — is the right one.

## Project structure

```
data/
  churn_dataset.csv      Original synthetic dataset (raw, before Phase 1)
notebooks/
  churn_modeling.ipynb   Phase 2 (model comparison) + Phase 3 (hyperparameter tuning)
requirements.txt         Python dependencies
```

Phase 1 (EDA, missing-value analysis, outlier analysis, statistical tests, feature
engineering) doesn't have a standalone notebook in this repo. Its approach and key
findings are summarised under Methodology below, and its output is what
`churn_modeling.ipynb` loads as its starting dataset.

## Methodology

**Phase 1 — Data preparation & EDA.** EDA was performed *before* any cleaning, per the
project's own rule, to avoid discarding informative patterns. Two features
(`engagement_score`, `support_tickets_90d`) showed Missing-Not-At-Random signals — the
missingness itself was predictive of churn — so those got a fill value plus a binary
missingness flag rather than simple imputation. Chi-square tests found `contract_type`
(χ²=170.2, p<0.001) and `auto_renew` (χ²=36.1, p<0.001) as the strongest categorical
predictors. Three features were engineered from the EDA findings, the strongest being
`disengagement_index = last_login_days_ago / (engagement_score + 1)`, which combines the
two numerically strongest correlates with churn into a single compound signal.

**Phase 2 — Model comparison.** Logistic Regression, Random Forest, and Gradient
Boosting were trained on identical preprocessing (scaling + one-hot encoding) and
compared on Recall and F1 for the churn class, since accuracy is a misleading metric on
this imbalanced dataset — predicting "retained" for everyone would still score ~64%
accuracy without learning anything. Logistic Regression won on both metrics; the
engineered features already captured the main variable interactions, so the extra
complexity of the ensemble methods didn't pay off.

**Phase 3 — Hyperparameter tuning.** Logistic Regression's `C`, `class_weight`,
`penalty`, and `solver` were each swept independently. `class_weight="balanced"` had by
far the largest effect, on its own doubling Recall. `C=1.0`, `penalty="l2"`, and
`solver="lbfgs"` were each confirmed as the best (or tied-for-best and simplest) choice.

## Setup

```bash
pip install -r requirements.txt
```

`data/churn_dataset.csv` is the original synthetic dataset issued for the course
(3,000 customers, 24 columns). The notebook itself loads the Phase 1 output
(`data/churn_phase1_v2.xlsx`, the cleaned, feature-engineered version described under
Methodology), which isn't included in the repository. To run the notebook, place that
file in `data/` or update `DATA_PATH` in the second cell of
`notebooks/churn_modeling.ipynb` to point at it.

## Tech stack

- **Pandas / NumPy** — data handling
- **Scikit-learn** — preprocessing pipeline, models, metrics
- **Matplotlib** — ROC curves

## Limitations

- The dataset is synthetic, so the results show how the models behave on this data rather than on real customers.
- Hyperparameters were compared on the same test set used for the final evaluation, so the final scores are slightly optimistic. Cross-validation on the training set would give a fully independent estimate.
