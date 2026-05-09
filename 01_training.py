# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Training: CatBoost vs RandomForest
# MAGIC ## Nested CV + Optuna + MLflow + leakage-free Pipeline
# MAGIC
# MAGIC **Modelos.** Dos no-lineales NO redundantes:
# MAGIC - **CatBoost** — boosting secuencial, ordered boosting (regularización built-in)
# MAGIC - **RandomForest** — bagging paralelo, filosofía opuesta
# MAGIC
# MAGIC **Validación.** Nested CV (5 outer × 3 inner) + Optuna 50 trials por outer fold.
# MAGIC La métrica reportada es la media de los 5 outer-fold scores — el único
# MAGIC estimate matemáticamente insesgado de la performance esperada.
# MAGIC
# MAGIC **Anti-leakage.** Todo va dentro de un `sklearn.Pipeline` (FE → scaler →
# MAGIC selector → modelo). Los parámetros aprendidos por scaler/selector vienen
# MAGIC SOLO de los datos de train de cada fold.

# COMMAND ----------

# MAGIC %pip install -q catboost optuna
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils.validation import check_is_fitted

from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import pearsonr, spearmanr, shapiro, wilcoxon

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams["figure.dpi"] = 110

EXPERIMENT_NAME = "/Users/your.email@company.com/regression_training"  # EDIT
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load training data from Delta

# COMMAND ----------

train_sdf = spark.read.table("main.default.regression_training")
df_train = train_sdf.toPandas()
y_full = df_train["target"].values
X_full = df_train.drop(columns="target").values
print(f"Training: {X_full.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature engineering as sklearn Transformer
# MAGIC
# MAGIC Implementado como `BaseEstimator + TransformerMixin` para encajar dentro del
# MAGIC `Pipeline` y heredar la garantía anti-leakage entre folds.

# COMMAND ----------

TOP_MI_FEATURES = ["feature_2","feature_13","feature_16","feature_9",
                   "feature_3","feature_18","feature_11","feature_5"]
TOP4_LINEAR     = ["feature_2","feature_13","feature_9","feature_11"]
WEIGHTED_TOP_WEIGHTS = {"feature_2":0.50,"feature_13":0.30,
                        "feature_16":0.15,"feature_9":0.05}

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic FE: log/sqrt/sq on top-MI, products on top-4, row aggregates."""
    def __init__(self, original_cols=None):
        self.original_cols = original_cols
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns.tolist())
        else:
            self.feature_names_in_ = np.array(self.original_cols
                                                or [f"feature_{i}" for i in range(X.shape[1])])
        self.n_features_in_ = len(self.feature_names_in_)
        return self
    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = X.copy()
        for f in TOP_MI_FEATURES:
            out[f"{f}_log"]  = np.log1p(np.abs(X[f]))
            out[f"{f}_sqrt"] = np.sqrt(np.abs(X[f]))
            out[f"{f}_sq"]   = X[f] ** 2
        for i, a in enumerate(TOP4_LINEAR):
            for b in TOP4_LINEAR[i+1:]:
                out[f"{a}_x_{b}"]   = X[a] * X[b]
                out[f"{a}_div_{b}"] = X[a] / (np.abs(X[b]) + 1e-6)
        base = X[list(self.feature_names_in_)]
        out["row_mean"]  = base.mean(axis=1)
        out["row_std"]   = base.std(axis=1)
        out["row_max"]   = base.max(axis=1)
        out["row_min"]   = base.min(axis=1)
        out["row_range"] = out["row_max"] - out["row_min"]
        out["weighted_top"] = sum(w * X[f] for f, w in WEIGHTED_TOP_WEIGHTS.items())
        return out.values

# Test it works
fe_test = FeatureEngineer().fit(X_full)
print(f"After FE: {fe_test.transform(X_full).shape[1]} features (20 → 62)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pipeline factory — anti-leakage

# COMMAND ----------

def _mi_score(X, y):
    return mutual_info_regression(X, y, random_state=42)

def build_pipeline(estimator, n_select=50):
    """
    FeatureEngineer → PowerTransformer(yeo-johnson) → SelectKBest(MI) → estimator.
    Pipeline.fit() learns scaler λ + MI scores from train ONLY.
    Pipeline.predict() applies them to test — zero leakage.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("scaler",           PowerTransformer(method="yeo-johnson", standardize=True)),
        ("selector",         SelectKBest(score_func=_mi_score, k=n_select)),
        ("model",            estimator),
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Optuna search spaces — TPE Bayesian optimization

# COMMAND ----------

def catboost_search_space(trial):
    return {
        "selector__k":          trial.suggest_int("selector__k", 30, 62),
        "model__iterations":    trial.suggest_int("model__iterations", 400, 1500),
        "model__depth":         trial.suggest_int("model__depth", 4, 8),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-2, 1e-1, log=True),
        "model__l2_leaf_reg":   trial.suggest_float("model__l2_leaf_reg", 0.5, 10.0, log=True),
        "model__subsample":     trial.suggest_float("model__subsample", 0.7, 1.0),
        "model__bagging_temperature": trial.suggest_float("model__bagging_temperature", 0.0, 1.0),
    }

def random_forest_search_space(trial):
    return {
        "selector__k":              trial.suggest_int("selector__k", 30, 62),
        "model__n_estimators":      trial.suggest_int("model__n_estimators", 100, 800),
        "model__max_depth":         trial.suggest_int("model__max_depth", 5, 20),
        "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
        "model__min_samples_leaf":  trial.suggest_int("model__min_samples_leaf", 1, 10),
        "model__max_features":      trial.suggest_categorical("model__max_features",
                                                                ["sqrt", "log2", 0.5, 0.7]),
    }

SEARCH_SPACES = {
    "CatBoost":     catboost_search_space,
    "RandomForest": random_forest_search_space,
}

def build_estimator(model_name):
    """Default estimator — Optuna will overwrite hyperparams via set_params."""
    if model_name == "CatBoost":
        return CatBoostRegressor(random_state=42, verbose=False, allow_writing_files=False)
    elif model_name == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=-1)
    raise ValueError(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Nested CV implementation

# COMMAND ----------

def run_optuna(X_tr, y_tr, model_name, inner_cv, n_trials, random_state):
    """Inner Optuna study — picks hyperparameters via TPE."""
    space_fn = SEARCH_SPACES[model_name]
    def objective(trial):
        params = space_fn(trial)
        pipeline = build_pipeline(build_estimator(model_name))
        pipeline.set_params(**params)
        scores = []
        for fi, (tr, va) in enumerate(inner_cv.split(X_tr, y_tr)):
            pipeline.fit(X_tr[tr], y_tr[tr])
            s = pipeline.score(X_tr[va], y_tr[va])
            scores.append(s)
            trial.report(np.mean(scores), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    sampler = TPESampler(seed=random_state, n_startup_trials=10, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def compute_metrics(y_true, y_pred):
    """Advanced metric set."""
    res = y_true - y_pred
    eps = 1e-8
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    yt_m, yp_m = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - yt_m) * (y_pred - yp_m))
    ccc = float(2*cov / (y_true.var() + y_pred.var() + (yt_m-yp_m)**2 + eps))
    nse = float(1 - np.sum(res**2) / (np.sum((y_true-yt_m)**2) + eps))
    return {
        "r2": r2, "rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred)),
        "ccc": ccc, "nse": nse, "mbe": float(res.mean()),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_rho": float(spearmanr(y_true, y_pred)[0]),
        "efficiency": r2 / 0.92,
    }


def nested_cv(X, y, model_name, n_outer=5, n_inner=3, n_trials=50, seed=42):
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed+1)
    fold_scores, fold_params, fold_preds, fold_idx_list = [], [], [], []
    for fold_idx, (otr, ote) in enumerate(outer_cv.split(X, y)):
        study = run_optuna(X[otr], y[otr], model_name, inner_cv, n_trials, seed+fold_idx)
        best = study.best_params
        pipe = build_pipeline(build_estimator(model_name))
        pipe.set_params(**best)
        pipe.fit(X[otr], y[otr])
        yp = pipe.predict(X[ote])
        m = compute_metrics(y[ote], yp)
        fold_scores.append(m); fold_params.append(best)
        fold_preds.append(yp); fold_idx_list.append(ote)
        print(f"  outer {fold_idx}: R²={m['r2']:.4f}  RMSE={m['rmse']:.3f}  "
              f"trials={len([t for t in study.trials if t.state.is_finished()])}")
    # Pick best fold's params, refit on FULL data
    best_idx = int(np.argmax([s["r2"] for s in fold_scores]))
    best_params = fold_params[best_idx]
    final_pipe = build_pipeline(build_estimator(model_name))
    final_pipe.set_params(**best_params)
    final_pipe.fit(X, y)
    return {
        "fold_scores": fold_scores,
        "fold_params": fold_params,
        "fold_preds":  fold_preds,
        "fold_idx":    fold_idx_list,
        "best_params": best_params,
        "final_pipeline": final_pipe,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run nested CV for both models
# MAGIC
# MAGIC ⚠️ Esta celda toma 20-40 min en cluster pequeño. Reducir `N_TRIALS` para iteración rápida.

# COMMAND ----------

N_OUTER, N_INNER, N_TRIALS = 5, 3, 50

# ---- CatBoost ----
print("="*72); print("CatBoost — nested CV"); print("="*72)
mlflow.sklearn.autolog(disable=True)
t0 = time.time()
with mlflow.start_run(run_name="CatBoost_nested_cv") as cb_run:
    mlflow.set_tags({"model_family": "CatBoostRegressor", "stage": "training"})
    cb_result = nested_cv(X_full, y_full, "CatBoost",
                            N_OUTER, N_INNER, N_TRIALS)
    elapsed = time.time() - t0
    cb_r2_mean = float(np.mean([s["r2"] for s in cb_result["fold_scores"]]))
    cb_r2_std  = float(np.std([s["r2"] for s in cb_result["fold_scores"]]))
    mlflow.log_metric("cv_r2_mean", cb_r2_mean)
    mlflow.log_metric("cv_r2_std",  cb_r2_std)
    mlflow.log_metric("cv_time_min", elapsed/60)
    for fi, m in enumerate(cb_result["fold_scores"]):
        for k, v in m.items(): mlflow.log_metric(f"fold{fi}_{k}", v)
    mlflow.log_dict(cb_result["best_params"], "best_params.json")
    sig = infer_signature(X_full, cb_result["final_pipeline"].predict(X_full))
    mlflow.sklearn.log_model(cb_result["final_pipeline"], name="model",
                                signature=sig, input_example=X_full[:3])
    print(f"\nCV R² = {cb_r2_mean:.4f} ± {cb_r2_std:.4f}  ({elapsed/60:.1f} min)")
    print(f"Best params: {cb_result['best_params']}")

# COMMAND ----------

# ---- RandomForest ----
print("="*72); print("RandomForest — nested CV"); print("="*72)
t0 = time.time()
with mlflow.start_run(run_name="RandomForest_nested_cv") as rf_run:
    mlflow.set_tags({"model_family": "RandomForestRegressor", "stage": "training"})
    rf_result = nested_cv(X_full, y_full, "RandomForest",
                            N_OUTER, N_INNER, N_TRIALS)
    elapsed = time.time() - t0
    rf_r2_mean = float(np.mean([s["r2"] for s in rf_result["fold_scores"]]))
    rf_r2_std  = float(np.std([s["r2"] for s in rf_result["fold_scores"]]))
    mlflow.log_metric("cv_r2_mean", rf_r2_mean)
    mlflow.log_metric("cv_r2_std",  rf_r2_std)
    mlflow.log_metric("cv_time_min", elapsed/60)
    for fi, m in enumerate(rf_result["fold_scores"]):
        for k, v in m.items(): mlflow.log_metric(f"fold{fi}_{k}", v)
    mlflow.log_dict(rf_result["best_params"], "best_params.json")
    sig = infer_signature(X_full, rf_result["final_pipeline"].predict(X_full))
    mlflow.sklearn.log_model(rf_result["final_pipeline"], name="model",
                                signature=sig, input_example=X_full[:3])
    print(f"\nCV R² = {rf_r2_mean:.4f} ± {rf_r2_std:.4f}  ({elapsed/60:.1f} min)")
    print(f"Best params: {rf_result['best_params']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Statistical comparison — paired Wilcoxon

# COMMAND ----------

with mlflow.start_run(run_name="model_comparison"):
    mlflow.set_tag("stage", "comparison")
    cb_r2 = np.array([s["r2"] for s in cb_result["fold_scores"]])
    rf_r2 = np.array([s["r2"] for s in rf_result["fold_scores"]])
    wstat, wp = wilcoxon(cb_r2, rf_r2)
    mlflow.log_metric("catboost_r2_mean", cb_r2.mean())
    mlflow.log_metric("rf_r2_mean",        rf_r2.mean())
    mlflow.log_metric("diff_mean",         cb_r2.mean() - rf_r2.mean())
    mlflow.log_metric("wilcoxon_p_value",  float(wp))

    print(f"CatBoost outer R²s: {[round(x,4) for x in cb_r2]}")
    print(f"RF       outer R²s: {[round(x,4) for x in rf_r2]}")
    print(f"\nMean diff (CatBoost - RF): {cb_r2.mean()-rf_r2.mean():+.4f}")
    print(f"Wilcoxon p-value:          {wp:.4f}")
    if wp < 0.05:
        print(">>> Difference is STATISTICALLY SIGNIFICANT (α=0.05)")
    else:
        print(">>> No statistically significant difference detected")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(N_OUTER), cb_r2, "o-", label="CatBoost",     color="seagreen", lw=2)
    ax.plot(range(N_OUTER), rf_r2, "s-", label="RandomForest", color="steelblue", lw=2)
    ax.axhline(0.92, color="red", ls="--", alpha=0.6, label="Theoretical ceiling")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("R²")
    ax.set_title(f"Outer-fold R²: CatBoost vs RandomForest — Wilcoxon p={wp:.4f}")
    ax.legend(); ax.set_xticks(range(N_OUTER))
    plt.tight_layout()
    plt.savefig("/tmp/comparison.png", dpi=110, bbox_inches="tight"); plt.close()
    mlflow.log_artifact("/tmp/comparison.png")

# Decide champion
if cb_r2.mean() > rf_r2.mean():
    champion_name, champion_run, champion_result = "CatBoost", cb_run, cb_result
else:
    champion_name, champion_run, champion_result = "RandomForest", rf_run, rf_result
print(f"\n>>> CHAMPION: {champion_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Champion diagnostics on out-of-fold predictions

# COMMAND ----------

# Concatenate OOF predictions (honest — never trained on the data being predicted)
oof = np.empty(len(y_full))
for fi, idx in enumerate(champion_result["fold_idx"]):
    oof[idx] = champion_result["fold_preds"][fi]

with mlflow.start_run(run_name="champion_diagnostics"):
    mlflow.set_tags({"stage": "diagnostics", "champion": champion_name})

    res = y_full - oof
    sh_p = float(shapiro(res).pvalue)
    het_r, het_p = pearsonr(oof, np.abs(res))
    dw = float(np.sum(np.diff(res)**2) / np.sum(res**2))
    mlflow.log_metric("oof_r2", r2_score(y_full, oof))
    mlflow.log_metric("residual_shapiro_p", sh_p)
    mlflow.log_metric("heteroscedasticity_p", het_p)
    mlflow.log_metric("durbin_watson", dw)

    # Bootstrap CI on full-data R²
    rng = np.random.default_rng(42)
    n = len(y_full)
    boot = np.array([r2_score(y_full[rng.integers(0,n,n)], oof[rng.integers(0,n,n)])
                      for _ in range(2000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    mlflow.log_metric("oof_r2_ci95_low",  float(ci_lo))
    mlflow.log_metric("oof_r2_ci95_high", float(ci_hi))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0,0].scatter(y_full, oof, alpha=0.5, s=20, c="steelblue", edgecolor="k")
    mn, mx = y_full.min(), y_full.max()
    axes[0,0].plot([mn,mx], [mn,mx], "r--", lw=1.5)
    axes[0,0].set_xlabel("y true"); axes[0,0].set_ylabel("ŷ (OOF)")
    axes[0,0].set_title(f"OOF predictions — R²={r2_score(y_full, oof):.4f}")

    axes[0,1].scatter(oof, res, alpha=0.5, s=20, c="darkorange", edgecolor="k")
    axes[0,1].axhline(0, color="red", ls="--")
    axes[0,1].set_xlabel("ŷ"); axes[0,1].set_ylabel("residual")
    axes[0,1].set_title(f"Residuals  (BP-p={het_p:.3f}, DW={dw:.2f})")

    from scipy import stats as sst
    sst.probplot(res, dist="norm", plot=axes[1,0])
    axes[1,0].set_title(f"Q-Q plot  (Shapiro p={sh_p:.3f})")

    axes[1,1].hist(boot, bins=50, color="purple", edgecolor="k", alpha=0.75)
    axes[1,1].axvline(ci_lo, color="red", ls="--", label=f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
    axes[1,1].axvline(ci_hi, color="red", ls="--")
    axes[1,1].axvline(r2_score(y_full, oof), color="green", lw=2)
    axes[1,1].set_xlabel("Bootstrap R²"); axes[1,1].set_title("Bootstrap distribution")
    axes[1,1].legend()
    plt.tight_layout()
    plt.savefig("/tmp/diagnostics.png", dpi=110, bbox_inches="tight"); plt.close()
    mlflow.log_artifact("/tmp/diagnostics.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Register champion in Model Registry → Production stage

# COMMAND ----------

REGISTERED_NAME = "regression_20feat_champion"
champion_run_id = champion_run.info.run_id

try:
    mv = mlflow.register_model(f"runs:/{champion_run_id}/model", REGISTERED_NAME)
    print(f"Registered '{REGISTERED_NAME}' v{mv.version}")
    client = MlflowClient()
    # Tag with key metrics
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "model_family",
                                    champion_name)
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "cv_r2_mean",
                                    f"{(cb_r2_mean if champion_name=='CatBoost' else rf_r2_mean):.4f}")
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "ci95",
                                    f"[{ci_lo:.3f}, {ci_hi:.3f}]")
    # Promote to Production (Unity Catalog or classic Workspace Registry)
    try:
        client.set_registered_model_alias(REGISTERED_NAME, "Production", mv.version)
        print(f"Alias 'Production' → v{mv.version}")
    except Exception:
        # Classic Registry uses stages instead of aliases
        client.transition_model_version_stage(REGISTERED_NAME, mv.version,
                                                "Production", archive_existing_versions=True)
        print(f"Stage Production → v{mv.version}")
except Exception as e:
    print(f"(Registry call skipped: {e})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save reference distributions for monitoring
# MAGIC
# MAGIC The monitoring notebook (`03_monitoring`) will compare incoming batches
# MAGIC against this baseline to detect drift. We persist:
# MAGIC - The training feature distributions (deciles per feature)
# MAGIC - The training target distribution
# MAGIC - The OOF prediction distribution (proxy for healthy production output)

# COMMAND ----------

reference_pdf = pd.DataFrame(X_full,
                              columns=[f"feature_{i}" for i in range(20)])
reference_pdf["target"] = y_full
reference_pdf["oof_prediction"] = oof

(spark.createDataFrame(reference_pdf)
    .write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.default.regression_training_reference"))
print("Reference distributions saved → main.default.regression_training_reference")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |---|---|
# MAGIC | Champion | logged to MLflow |
# MAGIC | CV R² (mean ± std) | from nested CV outer-fold scores |
# MAGIC | 95% bootstrap CI on R² | logged |
# MAGIC | Wilcoxon p-value (CatBoost vs RF) | logged |
# MAGIC | Champion artifact | full Pipeline registered in Model Registry as Production |
# MAGIC | Reference distributions | saved as Delta for downstream monitoring |
# MAGIC
# MAGIC Next: `02_batch_inference.py` — load champion from Registry, predict the blind batch.
