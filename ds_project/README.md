# Regression on 20-feature noisy dataset

End-to-end ML pipeline: **PySpark + MLflow + Databricks**, implementing the
hybrid PyCharm-for-engineering / Databricks-for-experimentation workflow.

## Problem
Predict a continuous target from 20 numerical features. The target has
irreducible noise such that the **theoretical R² ceiling is 0.92**.

## Result
Champion model: **CatBoost** (after benchmarking 14 architectures, narrowed to 4 + stacking).

| Metric | Value |
|---|---|
| Holdout R² | ≈ 0.88 |
| Bootstrap 95% CI | [0.84, 0.91] |
| Efficiency vs ceiling | ~95% |

---

## Repository layout

```
ds_project/
├── README.md                   ← this file
├── requirements.txt
├── setup.py                    ← makes src/ importable
├── src/                        ← reusable library (developed in PyCharm)
│   ├── __init__.py
│   ├── feature_engineering.py  ← deterministic FE pipeline
│   ├── modeling.py             ← model factory (4 champions + stacking)
│   └── metrics.py              ← regression metrics + bootstrap CI + diagnostics
├── notebooks/                  ← Databricks notebooks (sources)
│   └── 01_full_pipeline.py     ← export → .ipynb in the workspace
├── tests/                      ← pytest unit tests for the library
│   └── test_feature_engineering.py
└── configs/
```

## Workflow PyCharm ↔ Databricks

This project is designed for the canonical hybrid workflow:

1. **PyCharm**: write and unit-test the library code (`src/`, `tests/`).
   - Run `pytest` locally before every push.
   - The library is pure Python — no Spark needed for unit tests.
2. **GitHub**: single source of truth.
3. **Databricks Repos** (Repos → Add Repo → paste GitHub URL):
   - Pulls automatically; the notebook does `from src.feature_engineering import ...`
   - Edit the notebook in Databricks (rich UI, MLflow, cluster) and commit back to Git.

This separation keeps engineering rigour where it belongs (IDE, tests, linters)
and lets the notebook focus on storytelling + experimentation + tracking.

---

## Running locally

```bash
# 1. clone + install
git clone <repo>.git && cd ds_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # install src/ as editable package

# 2. unit tests
pytest tests/ -v

# 3. start MLflow tracking UI (separate terminal)
mlflow ui --port 5000        # → http://localhost:5000

# 4. run the pipeline as a script (or open in Jupyter)
python notebooks/01_full_pipeline.py
```

## Running on Databricks

1. **Repos → Add Repo** → paste the GitHub URL (HTTPS or SSH).
2. Open `notebooks/01_full_pipeline.py` (Databricks renders it as a notebook).
3. Attach to a cluster with **DBR 14.x ML** (includes Spark 3.5, MLflow, sklearn, xgboost; LightGBM and CatBoost installed via `%pip` cell at the top).
4. Adjust `EXPERIMENT_NAME` and `TRAIN_PATH` / `BLIND_PATH` to your DBFS / Volume locations.
5. Run all. MLflow runs appear automatically in the Experiments tab.

---

## Sharing the work with reviewers

Three options, in order of impact:

1. **GitHub repo** — they read the code (`src/`, `tests/`) and the notebook source
   on GitHub directly. Databricks notebooks export as readable `.py` with cell markers.
2. **HTML export of the notebook** — `File → Export → HTML/.dbc`. They see all
   outputs and plots without needing a cluster.
3. **Databricks Community Edition** (free, separate from your corporate workspace) —
   import the repo there and share the public notebook link.

> **Important.** Do not develop this on your employer's Databricks workspace.
> Use either Community Edition or local pyspark + mlflow.

---

## Key design decisions (and why)

| Decision | Rationale |
|---|---|
| Yeo-Johnson over StandardScaler | All 20 features rejected normality (Shapiro p<0.05); +1.8 R² on Ridge baseline |
| No PCA | Variance is uniformly spread (3.6%–6.5% per PC). Tree models are rotation-sensitive — PCA actively hurts them. |
| Hand-crafted feature engineering | `feature_2 × feature_13` reached MI = 0.30, larger than any raw feature. Lift = +6 R² on RF, +10 on Ridge. |
| Top-50 by MI feature selection | CV-validated optimum; further pruning hurts |
| Champions: CatBoost / LGB / XGB / GBM | All boosting; broader benchmark of 14 models showed boosting dominated |
| Stacking with BayesianRidge meta | Reduces variance; BayesianRidge gives stable weights with small data |
| Bootstrap CI on R² | Point estimate alone is misleading on n=160 holdout |
| Per-instance disagreement | Cheap uncertainty proxy — std across 4 base learners |

## What this project demonstrates

- **Spark / Delta** for IO + EDA + descriptive statistics (scales when data grows)
- **MLflow** end-to-end: experiment tracking, model logging, signature inference, model registry
- **Causal-inference adjacent rigour**: bootstrap CIs, residual diagnostics (Shapiro, heteroscedasticity), honest holdout
- **Software engineering**: testable library, type hints, deterministic pipeline, no leakage
