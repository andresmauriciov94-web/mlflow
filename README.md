# Regression on noisy 20-feature dataset
## End-to-end ML pipeline — Databricks + Spark + MLflow

**Goal.** Predict a continuous target from 20 numerical features.
**Theoretical R² ceiling = 0.92** (target has irreducible noise).

This repository implements the full data-science lifecycle on **Databricks**:

```
EDA + justification → Training + nested CV → Batch inference → Drift monitoring → Canary deployment
   (00_eda.py)          (01_training.py)      (02_batch_inference.py) (03_monitoring.py)  (04_canary_deployment.py)
```

Designed to satisfy the must-have skills of a Senior Data Scientist role:
**Python + SQL + Spark + Databricks + MLflow + linear/clustering understanding +
A/B-testing-style validation methods + basic MLOps practices**.

---
---

## Quick start

### In Databricks

1. **Repos → Add Repo →** paste this repo https://github.com/andresmauriciov94-web/mlflow.git.
2. Open `notebooks/00_eda.py` and run all. Repeat for 01, 02, 03, 04 in order.
3. Edit the `EXPERIMENT_NAME` and data paths in the first cell of each notebook
   to match your workspace.

  
# ExamplePaths 
-   INPUT_TABLE      = "xxx.blind_test_data".
-   PREDICTIONS_TBL  = "xxx.regression_predictions".
-   REFERENCE_TBL    = "xxx.regression_training_reference"
-   REGISTERED_NAME  = "regression_20feat_champion"

### Required cluster

- DBR 14.x ML or higher (includes MLflow, sklearn, xgboost, Spark MLlib).
- `%pip install` cells at the top of each notebook handle remaining packages
  (catboost, optuna).

---


## Repository layout

```
mlflow_regresion/
├── README.md
├── notebooks/
│   ├── 00_eda.py                  # EDA + clustering + quantitative justification
│   ├── 01_training.py             # Nested CV + Optuna + 2 non-redundant models
│   ├── 02_batch_inference.py      # Load champion from Registry → predict → Delta
│   ├── 03_monitoring.py           # Drift detection with temporal simulation
│   └── 04_canary_deployment.py    # A/B testing + canary promotion + MLOps lifecycle

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

### `01_training.py` — ML pipeline

- **Two non-redundant non-linear models:** CatBoost (sequential boosting) vs
  RandomForest (parallel bagging) — opposite philosophies.
- **Leakage-free `sklearn.Pipeline`:** FE → PowerTransformer → SelectKBest(MI)
  → estimator. The pipeline's `.fit()` learns scaler λ and MI scores from
  train ONLY; `.predict()` applies them to test.
- **Nested cross-validation (5×3) + Optuna:** outer 5-fold measures
  generalization, inner 3-fold + 50 Optuna trials select hyperparameters.
  This is the mathematically unbiased way to report tuned-model
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

- **Three drift metrics per feature:** PSI, KS test,
  Wasserstein distance.
- **Temporal simulation:** the 200-row blind set is split into 4 sub-batches
  representing 4 days of production, with progressive synthetic drift injected
  to demonstrate the system's ability to escalate severity.
- **Severity classification:** ok / warning / alert based on PSI + KS thresholds.
- **Persisted as Delta** (`regression_monitoring`) for ongoing dashboards.
- **Visual dashboard** (PSI evolution over time, severity timeline, prediction
  distribution shift) + SQL queries ready for Databricks Alerts.

### `04_canary_deployment.py` — Champion vs Challenger validation + canary promotion

Implements a complete **model validation and promotion pipeline** before
deploying a Challenger model to production:

#### A/A Test (sanity check)
- Splits the Champion's predictions into two random halves and runs
  Welch t-test, Mann-Whitney U, and KS test.
- Validates the framework does **not produce false positives** (all p-values > 0.05).
- Logs 6-panel diagnostic plots to MLflow (distribution overlay, boxplot,
  Q-Q plot, bootstrap CI, ECDF, summary stats).

#### A/B Test — Champion vs Challenger (50/50 split)
- Assigns 50% of observations to Champion, 50% to Challenger via random Bernoulli.
- Statistical comparison: Welch t-test + Mann-Whitney + KS test + bootstrap CI.
- Latency comparison: ms/row for each model.
- Logs 6-panel diagnostic plots as MLflow artifacts.
- Automated decision flags: `is_statistically_significant`, `challenger_is_faster`.

#### Paired Analysis — Every row predicted by both models
- Each observation receives predictions from **both** models.
- Tests: Wilcoxon signed-rank (paired), paired t-test, Shapiro-Wilk (normality of diffs).
- Correlation: Pearson r + Spearman ρ between the two models' predictions.
- 6-panel diagnostic plots logged to MLflow:
  - **Scatter agreement** (Champion ŷ vs Challenger ŷ)
  - **Bland-Altman** (bias + limits of agreement ±1.96σ)
  - **Histogram of per-row differences**
  - **Q-Q plot of differences** (normality validation for paired t-test)
  - **ECDF of |differences|** (P50 and P90 absolute error)
  - **Summary panel** with all metrics consolidated

#### Canary Deployment Simulation (progressive rollout)
- 4-stage scaling: 5% → 25% → 50% → 95% Challenger traffic.
- **Operational thresholds** (not statistical):

| Metric | Threshold | Purpose |
|---|---|---|
| Mean shift | ±1.5 | Prediction level stability |
| Prediction PSI | ≤ 0.25 | Distribution stability vs Champion baseline |
| Latency ratio | ≤ 2.0× | Performance SLA compliance |
| Correlation | ≥ 0.85 | Ranking consistency with Champion |

- Each stage logged as an independent MLflow run with metrics + tags.
- Go/no-go decision per stage; final promotion requires ALL stages healthy.

#### Automated Promotion Logic
- If all canary stages pass → promote Challenger to `@Production` alias.
- If any stage fails → keep Champion, tag as `rollback_keep_champion`.
- Decisions persisted as Delta table (`regression_canary_decisions`) for auditing.

#### Databricks

| Databricks |
|---|
| Model Registry aliases + Workflow conditional task |
| Modify `traffic_split` config in Workflow |
| Inference Tables / Delta append per batch |
| Lakehouse Monitoring + SQL Alerts + auto-rollback |


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
| A/A test before A/B | Validates framework integrity — prevents false positive deployments |
| Paired analysis + Bland-Altman | Medical-grade agreement analysis; detects systematic bias the A/B test misses |
| Canary progressive rollout | Limits blast radius; 5→25→50→95% mirrors production-grade deployment |
| Operational thresholds (PSI, latency, correlation) | Statistical significance alone is insufficient — SLAs and user experience matter |
| `mlflow.log_figure()` for plots | Diagnostic visualizations persisted as artifacts for post-mortem review |
| Decisions as Delta | Auditable promotion/rollback history queryable via SQL |

---

## MLflow Experiment Structure

All runs for the canary pipeline are tracked under a single experiment:

```
/Users/<user>/regression_canary
├── aa_test_sanity_check           # A/A validation (6 diagnostic plots)
├── ab_test_champion_vs_challenger # A/B comparison (6 diagnostic plots)
├── paired_analysis                # Full paired comparison (6 diagnostic plots)
├── canary_day_1_share_5pct        # Canary stage 1
├── canary_day_2_share_25pct       # Canary stage 2
├── canary_day_3_share_50pct       # Canary stage 3
├── canary_day_4_share_95pct       # Canary stage 4
└── promotion_decision             # Final promote/rollback action
```

Each run includes: metrics, tags, and diagnostic plot artifacts under `artifacts/plots/`.

---

## How to extend to production

1. Wrap `02_batch_inference.py` as a **Databricks Job** with cron schedule.
2. Chain `02 → 03 → 04` in the same Job:
   - If `03` reports `severity = alert`, fire a notification or auto-trigger retraining.
   - If `04` promotes the Challenger, update the Registry alias automatically.
3. Add a **drift-triggered retraining task** that re-runs `01_training.py` and
   re-deploys to the Registry only if the new model beats Production on a
   blind validation set.
4. Use **Databricks SQL Alerts** to email/Slack-notify when drift severity
   exceeds threshold or canary rollback occurs.
5. Move feature engineering into the **Databricks Feature Store** so other
   teams can consume the same definitions with strong lineage.
6. Add **Inference Tables** for real-time monitoring of prediction distributions
   and latency in serving endpoints.

---

## Model Registry Lifecycle

```
┌─────────────┐     canary passes      ┌─────────────┐
│  Challenger │ ──────────────────────► │  Production │
│  (alias)    │     all stages healthy  │  (alias)    │
└─────────────┘                         └─────────────┘
       │                                       │
       │  canary fails                         │  new challenger arrives
       ▼                                       ▼
┌─────────────┐                         ┌─────────────┐
│  Archived   │                         │  Archived   │
│  (dropped)  │                         │  (previous) │
└─────────────┘                         └─────────────┘
```

Aliases used: `@Production` (current champion), `@Challenger` (candidate under test).
Unity Catalog model: `colombina_prod.default.regression_20feat_champion`.
