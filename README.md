# Regression on noisy 20-feature dataset
## End-to-end ML pipeline — Databricks + Spark + MLflow

**Goal.** Predict a continuous target from 20 numerical features.
**Theoretical R² ceiling = 0.92** (target has irreducible noise).

This repository implements the full data-science lifecycle on **Databricks**:

```
EDA + justification    →    Training + nested CV    →    Batch inference    →    Drift monitoring
   (00_eda.py)               (01_training.py)             (02_batch_inference.py)   (03_monitoring.py)
```

Designed to satisfy the must-have skills of a Senior Data Scientist role:
**Python + SQL + Spark + Databricks + MLflow + linear/clustering understanding +
A/B-testing-style validation methods + basic MLOps practices**.

---

## Repository layout

```
mlflow/
├── README.md
├── notebooks/
│   ├── 00_eda.py                  # EDA + clustering + quantitative justification
│   ├── 01_training.py             # Nested CV + Optuna + 2 non-redundant models
│   ├── 02_batch_inference.py      # Load champion from Registry → predict → Delta
│   └── 03_monitoring.py           # Drift detection with temporal simulation
└── configs/
    └── thresholds.yaml            # PSI/KS thresholds for drift alerts
```

All notebooks are in **Databricks source format** (`# Databricks notebook source`
header + `# COMMAND ----------` cell separators). Databricks renders them as
notebooks; GitHub renders them as readable Python.

---

## What each notebook does

### `00_eda.py` — Quantitative EDA and justification

Demonstrates **with measurements** that:

1. **Linear models are insufficient** — Spark ML compares Ridge vs RandomForest
   on standardized features; the gap (~5+ pp R²) confirms non-linear structure.
2. **PCA does not help** — variance is uniformly distributed across components
   (3-6% per PC, no concentration); top-3 PCs explain only ~15%.
3. **Latent groups exist** — K-means clustering with elbow + silhouette finds
   real structure; cluster-to-target η² quantifies how much variance is
   explained by group membership alone.

All Spark MLlib (Correlation, VectorAssembler, KMeans, RandomForestRegressor,
LinearRegression with CrossValidator).

### `01_training.py` — Senior ML pipeline

- **Two non-redundant non-linear models:** CatBoost (sequential boosting) vs
  RandomForest (parallel bagging) — opposite philosophies.
- **Leakage-free `sklearn.Pipeline`:** FE → PowerTransformer → SelectKBest(MI)
  → estimator. The pipeline's `.fit()` learns scaler λ and MI scores from
  train ONLY; `.predict()` applies them to test.
- **Nested cross-validation (5×3) + Optuna:** outer 5-fold measures
  generalization, inner 3-fold + 50 Optuna trials select hyperparameters.
  This is the only mathematically unbiased way to report tuned-model
  performance.
- **Statistical comparison:** paired Wilcoxon signed-rank test on the 5
  outer-fold scores — non-parametric, isolates model effect from fold variance.
- **Champion registered** in MLflow Model Registry as `@Production`.
- **Reference distributions persisted** as Delta for downstream monitoring.

### `02_batch_inference.py` — Production inference pattern

- Reads input as Delta table.
- Loads champion via `mlflow.sklearn.load_model("models:/<name>@Production")` —
  the entire `Pipeline` (preprocessing + model) comes packaged.
- Predicts and writes results back as a Delta table (append mode, multi-batch).
- Logs the run to MLflow with model version + throughput metrics.
- The notebook is ready to be wrapped as a **Databricks Job** with cron or
  file-arrival trigger.

### `03_monitoring.py` — Drift detection with temporal simulation

- **Three drift metrics per feature:** PSI (industry standard), KS test,
  Wasserstein distance.
- **Temporal simulation:** the 200-row blind set is split into 4 sub-batches
  representing 4 days of production, with progressive synthetic drift injected
  to demonstrate the system's ability to escalate severity.
- **Severity classification:** ok / warning / alert based on PSI + KS thresholds.
- **Persisted as Delta** (`regression_monitoring`) for ongoing dashboards.
- **Visual dashboard** (PSI evolution over time, severity timeline, prediction
  distribution shift) + SQL queries ready for Databricks Alerts.

---

## Quick start

### In Databricks

1. **Repos → Add Repo →** paste this repo URL.
2. Open `notebooks/00_eda.py` and run all. Repeat for 01, 02, 03 in order.
3. Edit the `EXPERIMENT_NAME` and data paths in the first cell of each notebook
   to match your workspace.

### Required cluster

- DBR 14.x ML or higher (includes MLflow, sklearn, xgboost, Spark MLlib).
- `%pip install` cells at the top of each notebook handle remaining packages
  (catboost, optuna).

---

## Architectural decisions (and why)

| Decision | Rationale |
|---|---|
| `sklearn.Pipeline` end-to-end | Mathematical guarantee against train/test leakage |
| Yeo-Johnson scaling | All 20 features rejected normality (Shapiro p<0.05) |
| Mutual-information feature selection | Captures non-linearity (cf. Pearson, which is blind to it) |
| **CatBoost vs RandomForest** | Diverse non-linear models: sequential boosting vs parallel bagging — not redundant |
| Nested CV + Optuna | Unbiased generalization estimate; TPE converges 3-5× faster than RandomSearch |
| Wilcoxon paired test | Non-parametric, controls for fold-level variance (Demšar 2006) |
| MLflow Model Registry | Versioning + stage transitions + traceability across batches |
| Delta tables for everything | Schema enforcement, time-travel, ACID, SQL-queryable |
| PSI for drift | Industry standard (especially fintech/marketing); intuitive thresholds |
| Temporal simulation | Real production has time evolution; static drift would understate the system |

---

## How to extend to production

1. Wrap `02_batch_inference.py` as a **Databricks Workflow** with cron schedule.
2. Chain `02 → 03` in the same Job; if `03` reports `severity = alert`, fire a
   notification or auto-trigger retraining.
3. Add a **drift-triggered retraining task** that re-runs `01_training.py` and
   re-deploys to the Registry only if the new model beats Production on a
   blind validation set.
4. Use **Databricks SQL Alerts** to email/Slack-notify when drift severity
   exceeds threshold.
5. Move feature engineering into the **Databricks Feature Store** so other
   teams can consume the same definitions with strong lineage.
