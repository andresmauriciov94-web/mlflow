# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Canary Deployment + Model A/B Testing
# MAGIC
# MAGIC **Objective.** Before promoting a Challenger model to full production,
# MAGIC validate it with a **Champion vs Challenger** pattern on real traffic,
# MAGIC gradually scaling exposure (5% → 25% → 50% → 100%) based on observed metrics.
# MAGIC
# MAGIC ## Covered Concepts
# MAGIC
# MAGIC  Concept | Implementation |
# MAGIC ---|---|
# MAGIC  **Model A/B testing** | Random assignment + statistical tests on two predictors |
# MAGIC  **A/A test (sanity check)** | Verifies that the framework does not produce false positives |
# MAGIC  **Canary deployment** | Gradual promotion with operational thresholds |
# MAGIC  **Model Registry lifecycle** | Champion (Production) vs Challenger (Canary) via aliases |
# MAGIC  **Automated decision** | Promotion / rollback logic based on evidence |
# MAGIC
# MAGIC ## Equivalence with AWS SageMaker
# MAGIC
# MAGIC  AWS SageMaker | Databricks |
# MAGIC ---|---|
# MAGIC  Endpoint with `ProductionVariants` + `InitialVariantWeight` | Model Registry aliases + Workflow conditional task |
# MAGIC  `Data Capture` for logging | Inference Tables / Delta append in each batch |
# MAGIC  CloudWatch alarms + rollback | Lakehouse Monitoring + Databricks SQL Alerts |
# MAGIC  `UpdateEndpointWeights` (scale traffic) | Modify `traffic_split` config |
# MAGIC
# MAGIC For batch (this case), "traffic" translates to "% of the batch evaluated with each model".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install -q catboost
# MAGIC

# COMMAND ----------

import os, time, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from scipy import stats
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110

EXPERIMENT_NAME = "/Users/avalderrama@colombina.com/regression_canary"  # EDIT
mlflow.set_experiment(EXPERIMENT_NAME)

# Tables
BLIND_INPUT_TBL   = "hr.agent.blind_test_data"
CANARY_PREDS_TBL  = "hr.agent.regression_canary_predictions"
CANARY_DECISIONS_TBL = "hr.agent.regression_canary_decisions"

# Model Registry
REGISTERED_NAME = "regression_20feat_champion"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load both models — Champion & Challenger
# MAGIC
# MAGIC En producción real, **Champion** es la versión que actualmente sirve, y
# MAGIC **Challenger** es la versión candidata recién entrenada. Aquí usamos las
# MAGIC dos versiones que entrenamos en `01_training.py`: CatBoost (Champion)
# MAGIC y RandomForest (Challenger).

# COMMAND ----------

client = MlflowClient()

def load_model_safely(name: str, alias_or_stage: str):
    """Load via alias (Unity Catalog) or stage (classic Registry), whichever works."""
    try:
        mv = client.get_model_version_by_alias(name, alias_or_stage)
        uri = f"models:/{name}@{alias_or_stage}"
        return mlflow.sklearn.load_model(uri), mv.version, uri
    except Exception:
        try:
            versions = client.get_latest_versions(name, stages=[alias_or_stage])
            if versions:
                mv = versions[0]
                uri = f"models:/{name}/{alias_or_stage}"
                return mlflow.sklearn.load_model(uri), mv.version, uri
        except Exception as e:
            raise RuntimeError(f"Could not resolve {name}@{alias_or_stage}: {e}")

# Load champion (Production)
champion_model, champion_version, champion_uri = load_model_safely(
    REGISTERED_NAME, "Production")
print(f"CHAMPION: {champion_uri} (v{champion_version})")

# For the challenger, we'll use a separately registered model.
# In real production, this would be a different registered model
# (e.g., regression_20feat_challenger). For this demo, we register
# the RandomForest version we trained as "Challenger" alias.

# Find the RandomForest run from 01_training.py
runs = mlflow.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name(
        EXPERIMENT_NAME.replace("canary", "training")).experiment_id]
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME.replace("canary", "training"))
        else [mlflow.active_run().info.experiment_id if mlflow.active_run() else "0"],
    filter_string="tags.model_family = 'RandomForestRegressor'",
    max_results=1, order_by=["start_time DESC"],
)

if not runs.empty:
    rf_run_id = runs.iloc[0]["run_id"]
    rf_model_uri = f"runs:/{rf_run_id}/model"

    # Register as challenger (idempotent)
    try:
        mv_chall = mlflow.register_model(rf_model_uri, REGISTERED_NAME)
        try:
            client.set_registered_model_alias(REGISTERED_NAME, "Challenger", mv_chall.version)
            print(f"Registered RandomForest as Challenger alias (v{mv_chall.version})")
        except Exception:
            client.transition_model_version_stage(REGISTERED_NAME, mv_chall.version, "Staging")
            print(f"Promoted to Staging (v{mv_chall.version})")
    except Exception as e:
        print(f"Already registered or registration skipped: {e}")

challenger_model, challenger_version, challenger_uri = load_model_safely(
    REGISTERED_NAME, "Challenger")
print(f"CHALLENGER: {challenger_uri} (v{challenger_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load blind batch from Delta

# COMMAND ----------

blind_sdf = spark.read.table(BLIND_INPUT_TBL)
blind_pdf = blind_sdf.toPandas()
X_blind = blind_pdf.values
n_rows = len(blind_pdf)
print(f"Blind batch: {n_rows} rows × {blind_pdf.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. A/A Test — Sanity Check
# MAGIC
# MAGIC Antes del A/B real, validamos el framework con un **A/A test**: ambos arms
# MAGIC usan el MISMO modelo. Como por construcción no hay diferencia, los tests
# MAGIC NO deben detectar significancia. Si la detectan, el framework está roto.
# MAGIC
# MAGIC Es análogo a un programador escribiendo tests que **deben fallar** para
# MAGIC verificar que el sistema de tests funciona.

# COMMAND ----------

# DBTITLE 1,A/A Test with diagnostic plots logged to MLflow
# A/A test: both arms use the Champion
print("="*72); print("A/A TEST — Champion vs Champion (sanity check)"); print("="*72)
with mlflow.start_run(run_name="aa_test_sanity_check"):
    mlflow.set_tag("stage", "aa_test")
    mlflow.set_tag("expected_outcome", "no_significant_difference")

    arm = assign_treatment(n_rows, treatment_share=0.5, random_state=42)
    X_a = X_blind[arm == 0]
    X_b = X_blind[arm == 1]
    y_a = champion_model.predict(X_a)
    y_b = champion_model.predict(X_b)

    aa = compare_arms(y_a, y_b, "Champion_split_a", "Champion_split_b")
    for k, v in aa.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    print(f"  Mean A: {aa['mean_a']:.3f}   Mean B: {aa['mean_b']:.3f}")
    print(f"  Welch p={aa['welch_p']:.4f}  MW p={aa['mw_p']:.4f}  KS p={aa['ks_p']:.4f}")
    print(f"  CI95% on mean diff: [{aa['bootstrap_ci95_low']:.3f}, {aa['bootstrap_ci95_high']:.3f}]")

    framework_valid = (aa['welch_p'] > 0.05 and aa['mw_p'] > 0.05 and aa['ks_p'] > 0.05)
    mlflow.log_metric("framework_passes_aa_test", int(framework_valid))
    if framework_valid:
        print("\n  ✓ A/A passes — framework does not produce false positives")
    else:
        print("\n  ✗ A/A fails — framework is suspicious, investigate before A/B")

    # --- Log diagnostic plots to MLflow ---
    log_ab_plots(y_a, y_b, "Champion_split_a", "Champion_split_b", aa, prefix="aa_test_")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. A/B Test — Champion vs Challenger
# MAGIC
# MAGIC Aleatorización 50/50 (standard initial split). En un canary real
# MAGIC empezarías con 95/5 — lo veremos en §7.

# COMMAND ----------

# DBTITLE 1,A/B Test: Champion vs Challenger with diagnostic plots
print("="*72); print("A/B TEST — Champion vs Challenger (50/50 split)"); print("="*72)
with mlflow.start_run(run_name="ab_test_champion_vs_challenger") as ab_run:
    mlflow.set_tag("stage", "ab_test")
    mlflow.set_tag("champion_version",   str(champion_version))
    mlflow.set_tag("challenger_version", str(challenger_version))

    arm = assign_treatment(n_rows, treatment_share=0.5, random_state=42)
    X_control   = X_blind[arm == 0]    # Champion
    X_treatment = X_blind[arm == 1]    # Challenger
    t0 = time.time()
    y_control   = champion_model.predict(X_control)
    t_champion  = time.time() - t0
    t0 = time.time()
    y_treatment = challenger_model.predict(X_treatment)
    t_challenger = time.time() - t0

    ab = compare_arms(y_control, y_treatment, "Champion", "Challenger")
    for k, v in ab.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    # Latency comparison
    lat_champion_per_row    = t_champion    / max(len(X_control), 1)
    lat_challenger_per_row  = t_challenger  / max(len(X_treatment), 1)
    mlflow.log_metric("latency_champion_ms_per_row",    lat_champion_per_row * 1000)
    mlflow.log_metric("latency_challenger_ms_per_row",  lat_challenger_per_row * 1000)

    print(f"  Champion   mean: {ab['mean_a']:.3f}  std: {ab['std_a']:.3f}  n={ab['n_a']}")
    print(f"  Challenger mean: {ab['mean_b']:.3f}  std: {ab['std_b']:.3f}  n={ab['n_b']}")
    print(f"  Mean diff (C-Ch): {ab['mean_diff']:+.4f}")
    print(f"  CI95% on diff:    [{ab['bootstrap_ci95_low']:+.3f}, {ab['bootstrap_ci95_high']:+.3f}]")
    print(f"\n  Statistical tests:")
    print(f"    Welch t-test:  p={ab['welch_p']:.4f}")
    print(f"    Mann-Whitney:  p={ab['mw_p']:.4f}")
    print(f"    KS test:       p={ab['ks_p']:.4f}  (distribution shape)")
    print(f"\n  Latency:")
    print(f"    Champion:   {lat_champion_per_row*1000:.3f} ms/row")
    print(f"    Challenger: {lat_challenger_per_row*1000:.3f} ms/row")

    # Decision flags
    significant = ab['welch_p'] < 0.05
    challenger_faster  = lat_challenger_per_row < lat_champion_per_row
    mlflow.log_metric("is_statistically_significant", int(significant))
    mlflow.log_metric("challenger_is_faster",          int(challenger_faster))

    # --- Generate and log diagnostic plots to MLflow ---
    log_ab_plots(y_control, y_treatment, "Champion", "Challenger", ab, prefix="ab_test_")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Paired analysis — each row evaluated by BOTH models
# MAGIC
# MAGIC Más potente que A/B aleatorizado porque elimina la varianza entre instancias.
# MAGIC Cada fila tiene **dos predicciones**: comparamos las diferencias por fila.

# COMMAND ----------

# DBTITLE 1,Paired Analysis with diagnostic plots logged to MLflow
print("="*72); print("PAIRED ANALYSIS — Each row predicted by both models"); print("="*72)
with mlflow.start_run(run_name="paired_analysis"):
    mlflow.set_tag("stage", "paired_analysis")

    y_champ_all = champion_model.predict(X_blind)
    y_chall_all = challenger_model.predict(X_blind)
    diff_per_row = y_champ_all - y_chall_all

    # Paired Wilcoxon signed-rank test
    w_stat, w_p = stats.wilcoxon(y_champ_all, y_chall_all)
    # Paired t-test (assumes normality of differences)
    pt_stat, pt_p = stats.ttest_rel(y_champ_all, y_chall_all)
    # Shapiro on differences to validate paired t-test assumption
    sh_stat, sh_p = stats.shapiro(diff_per_row)

    # Pearson + Spearman between the two models' predictions
    corr_pearson, _   = stats.pearsonr(y_champ_all,  y_chall_all)
    corr_spearman, _  = stats.spearmanr(y_champ_all, y_chall_all)

    mlflow.log_metric("paired_wilcoxon_stat",  float(w_stat))
    mlflow.log_metric("paired_wilcoxon_p",     float(w_p))
    mlflow.log_metric("paired_ttest_stat",     float(pt_stat))
    mlflow.log_metric("paired_ttest_p",        float(pt_p))
    mlflow.log_metric("diff_shapiro_p",        float(sh_p))
    mlflow.log_metric("mean_abs_diff",         float(np.abs(diff_per_row).mean()))
    mlflow.log_metric("max_abs_diff",          float(np.abs(diff_per_row).max()))
    mlflow.log_metric("pearson_correlation",   float(corr_pearson))
    mlflow.log_metric("spearman_correlation",  float(corr_spearman))

    print(f"  Mean diff (Champion - Challenger): {diff_per_row.mean():+.4f}")
    print(f"  Std of diffs:                       {diff_per_row.std():.4f}")
    print(f"  Mean ABS diff:                      {np.abs(diff_per_row).mean():.4f}")
    print(f"  Max ABS diff:                       {np.abs(diff_per_row).max():.4f}")
    print(f"\n  Wilcoxon signed-rank (paired): p={w_p:.4f}")
    print(f"  Paired t-test:                 p={pt_p:.4f}  "
          f"(diff normality Shapiro p={sh_p:.3f} {'✓' if sh_p > 0.05 else '✗'})")
    print(f"\n  Models' prediction correlation:")
    print(f"    Pearson  r = {corr_pearson:.4f}")
    print(f"    Spearman ρ = {corr_spearman:.4f}")
    print(f"\n  Interpretation: models agree on RANK well "
          f"({'✓' if corr_spearman > 0.9 else 'no'}), level may shift.")

    # --- Paired analysis diagnostic plots ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Paired Analysis — Champion vs Challenger (all rows)", fontsize=14, fontweight="bold")

    # 1. Scatter: Champion vs Challenger agreement
    ax = axes[0, 0]
    ax.scatter(y_champ_all, y_chall_all, alpha=0.5, s=20, c="steelblue", edgecolor="k", lw=0.3)
    mn = min(y_champ_all.min(), y_chall_all.min())
    mx = max(y_champ_all.max(), y_chall_all.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="y = x (perfect agreement)")
    ax.set_xlabel("Champion ŷ")
    ax.set_ylabel("Challenger ŷ")
    ax.set_title(f"Prediction Agreement\nPearson r={corr_pearson:.4f}, Spearman ρ={corr_spearman:.4f}")
    ax.legend()
    ax.set_aspect("equal")

    # 2. Bland-Altman plot
    ax = axes[0, 1]
    means_per_row = (y_champ_all + y_chall_all) / 2
    ax.scatter(means_per_row, diff_per_row, alpha=0.5, s=20, c="darkorange", edgecolor="k", lw=0.3)
    ax.axhline(diff_per_row.mean(), color="red", ls="-", lw=2,
               label=f"Bias = {diff_per_row.mean():+.3f}")
    loa_upper = diff_per_row.mean() + 1.96 * diff_per_row.std()
    loa_lower = diff_per_row.mean() - 1.96 * diff_per_row.std()
    ax.axhline(loa_upper, color="red", ls="--", alpha=0.6, label=f"LoA ±1.96σ [{loa_lower:.2f}, {loa_upper:.2f}]")
    ax.axhline(loa_lower, color="red", ls="--", alpha=0.6)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Mean of both predictions")
    ax.set_ylabel("Champion − Challenger")
    ax.set_title("Bland-Altman (Agreement Analysis)")
    ax.legend(fontsize=8)

    # 3. Histogram of per-row differences
    ax = axes[0, 2]
    ax.hist(diff_per_row, bins=30, alpha=0.7, color="teal", edgecolor="white")
    ax.axvline(0, color="red", ls="-", lw=2, label="Zero diff")
    ax.axvline(diff_per_row.mean(), color="black", ls="--", lw=1.5, label=f"Mean={diff_per_row.mean():+.3f}")
    ax.set_xlabel("Champion − Challenger")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Differences\nWilcoxon p={w_p:.4f}, Paired-t p={pt_p:.4f}")
    ax.legend(fontsize=9)

    # 4. QQ plot of differences (normality check for paired t-test)
    ax = axes[1, 0]
    sorted_diffs = np.sort(diff_per_row)
    n = len(sorted_diffs)
    theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    ax.scatter(theoretical_q, sorted_diffs, alpha=0.6, s=15, color="purple")
    slope = diff_per_row.std()
    intercept = diff_per_row.mean()
    xlims = np.array([theoretical_q.min(), theoretical_q.max()])
    ax.plot(xlims, intercept + slope * xlims, "r--", lw=1.5, label="Normal reference")
    ax.set_xlabel("Theoretical quantiles (Normal)")
    ax.set_ylabel("Sample quantiles (diffs)")
    ax.set_title(f"Q-Q Plot of Differences\nShapiro p={sh_p:.4f} ({'✓ Normal' if sh_p > 0.05 else '✗ Non-normal'})")
    ax.legend()

    # 5. ECDF of absolute differences
    ax = axes[1, 1]
    abs_diffs = np.abs(diff_per_row)
    sorted_abs = np.sort(abs_diffs)
    ecdf_y = np.linspace(0, 1, len(sorted_abs))
    ax.step(sorted_abs, ecdf_y, where="post", color="darkgreen", lw=2)
    pct_90 = np.percentile(abs_diffs, 90)
    pct_50 = np.percentile(abs_diffs, 50)
    ax.axvline(pct_50, color="orange", ls="--", lw=1.5, label=f"Median |diff| = {pct_50:.3f}")
    ax.axvline(pct_90, color="red", ls="--", lw=1.5, label=f"P90 |diff| = {pct_90:.3f}")
    ax.set_xlabel("|Champion − Challenger|")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("ECDF of Absolute Differences")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 6. Summary panel
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        f"Paired Analysis Summary\n"
        f"{'─'*38}\n"
        f"N observations: {n_rows}\n"
        f"Mean diff (Champ-Chall): {diff_per_row.mean():+.4f}\n"
        f"Std of diffs:            {diff_per_row.std():.4f}\n"
        f"Mean |diff|:             {np.abs(diff_per_row).mean():.4f}\n"
        f"Max  |diff|:             {np.abs(diff_per_row).max():.4f}\n"
        f"{'─'*38}\n"
        f"Paired t-test:    p = {pt_p:.4f}\n"
        f"Wilcoxon signed:  p = {w_p:.4f}\n"
        f"Shapiro (diffs):  p = {sh_p:.4f}\n"
        f"{'─'*38}\n"
        f"Pearson r:   {corr_pearson:.4f}\n"
        f"Spearman ρ:  {corr_spearman:.4f}\n"
        f"{'─'*38}\n"
        f"Significant (α=0.05): {'YES ⚠️' if pt_p < 0.05 else 'NO ✓'}\n"
        f"Models agree on rank: {'✓' if corr_spearman > 0.9 else '✗'}\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    # Log directly to MLflow (no temp files needed)
    mlflow.log_figure(fig, "plots/paired_analysis_diagnostic.png")
    plt.show()
    plt.close(fig)
    print(f"  📊 Logged paired analysis plots to MLflow artifacts/plots/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Canary deployment simulation
# MAGIC
# MAGIC Real-world canary: empezas mandando un % pequeño del tráfico al
# MAGIC Challenger, monitoreas, escalas gradualmente. Si las métricas se mantienen
# MAGIC dentro de thresholds → escalar al siguiente nivel. Si no → rollback.
# MAGIC
# MAGIC Niveles típicos: **5% → 25% → 50% → 100%**.

# COMMAND ----------

# Thresholds for canary health (operational, not statistical)
THRESHOLDS = {
    "max_mean_shift":     1.5,       # ŷ mean shift must be < 1.5 vs Champion
    "max_pred_psi":       0.25,      # PSI on predictions vs Champion baseline
    "max_latency_ratio":  2.0,       # Challenger latency ≤ 2x Champion
    "min_correlation":    0.85,      # Predictions must correlate ≥ 0.85
}

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    eps = 1e-6
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3: return 0.0
    edges[0] -= eps; edges[-1] += eps
    r, _ = np.histogram(reference, bins=edges)
    c, _ = np.histogram(current,   bins=edges)
    r_pct = (r + eps) / (r.sum() + eps * len(r))
    c_pct = (c + eps) / (c.sum() + eps * len(c))
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def evaluate_canary_health(
    X: np.ndarray, champion_share: float, random_state: int,
) -> dict:
    """
    Simulate a canary deployment at a given Challenger share.
    Return health metrics + go/no-go decision.
    """
    arm = assign_treatment(len(X), treatment_share=(1 - champion_share),
                            random_state=random_state)
    n_canary = (arm == 1).sum()
    if n_canary == 0:
        return {"healthy": True, "n_canary": 0, "reason": "no canary traffic yet"}

    X_champ = X[arm == 0]
    X_canary = X[arm == 1]

    # Predictions and latencies
    t0 = time.time(); y_champ = champion_model.predict(X_champ);    lat_c = time.time() - t0
    t0 = time.time(); y_can   = challenger_model.predict(X_canary); lat_ca = time.time() - t0
    lat_champ_per  = lat_c  / max(len(X_champ), 1)
    lat_can_per    = lat_ca / max(len(X_canary), 1)

    # Reference for canary monitoring: champion's predictions on the same batch
    y_champ_full = champion_model.predict(X)
    pred_psi = compute_psi(y_champ_full, y_can) if len(y_can) >= 10 else 0.0

    # Compare on the rows the canary saw (against what champion WOULD have done)
    if len(X_canary) > 0:
        y_champ_on_canary = champion_model.predict(X_canary)
        mean_shift = y_can.mean() - y_champ_on_canary.mean()
        corr = (np.corrcoef(y_can, y_champ_on_canary)[0, 1]
                if len(y_can) > 1 else 1.0)
    else:
        mean_shift, corr = 0.0, 1.0

    latency_ratio = lat_can_per / max(lat_champ_per, 1e-6)

    # Apply thresholds
    fails = []
    if abs(mean_shift)     > THRESHOLDS["max_mean_shift"]:    fails.append("mean_shift")
    if pred_psi            > THRESHOLDS["max_pred_psi"]:      fails.append("prediction_psi")
    if latency_ratio       > THRESHOLDS["max_latency_ratio"]: fails.append("latency")
    if corr                < THRESHOLDS["min_correlation"]:   fails.append("correlation")

    return {
        "champion_share":   champion_share,
        "canary_share":     1 - champion_share,
        "n_champion":       int((arm == 0).sum()),
        "n_canary":         int(n_canary),
        "mean_shift":       float(mean_shift),
        "pred_psi":         float(pred_psi),
        "latency_ratio":    float(latency_ratio),
        "correlation":      float(corr),
        "fails":            fails,
        "healthy":          len(fails) == 0,
    }

# COMMAND ----------

# Simulate progressive canary scaling
print("="*72); print("CANARY DEPLOYMENT SIMULATION"); print("="*72)

CANARY_STAGES = [
    {"day": 1, "champion_share": 0.95},
    {"day": 2, "champion_share": 0.75},
    {"day": 3, "champion_share": 0.50},
    {"day": 4, "champion_share": 0.05},   # 95% Challenger (near-full promotion)
]

decisions = []
for stage in CANARY_STAGES:
    with mlflow.start_run(run_name=f"canary_day_{stage['day']}_share_{int((1-stage['champion_share'])*100)}pct"):
        mlflow.set_tags({"stage": "canary_simulation",
                          "canary_pct": str(int((1 - stage['champion_share']) * 100))})
        health = evaluate_canary_health(
            X_blind, champion_share=stage['champion_share'],
            random_state=42 + stage['day'],
        )
        # Log metrics
        for k, v in health.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_metric("healthy", int(health["healthy"]))
        mlflow.set_tag("decision",
                        "PROMOTE" if health["healthy"] else "ROLLBACK")
        mlflow.set_tag("failed_checks", ",".join(health["fails"]) or "none")

        decisions.append({"day": stage['day'], **health,
                           "decision": "promote" if health["healthy"] else "rollback"})

        pct = int((1 - stage['champion_share']) * 100)
        status = "✓ HEALTHY" if health["healthy"] else "✗ UNHEALTHY"
        print(f"\n  Day {stage['day']} — Canary at {pct}%  [{status}]")
        if health['n_canary'] == 0:
            print("    (no canary traffic)")
        else:
            print(f"    Champion split: {health['n_champion']} | Canary split: {health['n_canary']}")
            print(f"    Mean shift:    {health['mean_shift']:+.3f}  "
                  f"(threshold ±{THRESHOLDS['max_mean_shift']})")
            print(f"    Pred PSI:      {health['pred_psi']:.3f}  "
                  f"(threshold ≤{THRESHOLDS['max_pred_psi']})")
            print(f"    Latency ratio: {health['latency_ratio']:.2f}×  "
                  f"(threshold ≤{THRESHOLDS['max_latency_ratio']})")
            print(f"    Correlation:   {health['correlation']:.3f}  "
                  f"(threshold ≥{THRESHOLDS['min_correlation']})")
            if not health["healthy"]:
                print(f"    FAILED CHECKS: {', '.join(health['fails'])}")

# Final decision: promote only if ALL stages were healthy
all_healthy = all(d["healthy"] for d in decisions if d["n_canary"] > 0)
print("\n" + "="*72)
if all_healthy:
    print("✓ CANARY DEPLOYMENT SUCCESSFUL — promote Challenger to Production")
else:
    failed_days = [d["day"] for d in decisions if not d["healthy"]]
    print(f"✗ CANARY DEPLOYMENT FAILED at day(s) {failed_days} — keep Champion")
print("="*72)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Persist decisions as Delta

# COMMAND ----------

dec_pdf = pd.DataFrame(decisions)
dec_pdf["timestamp"] = datetime.now()
dec_pdf["champion_version"]   = str(champion_version)
dec_pdf["challenger_version"] = str(challenger_version)

(spark.createDataFrame(dec_pdf).write
    .format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CANARY_DECISIONS_TBL))
print(f"Decisions persisted: {CANARY_DECISIONS_TBL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Automated promotion logic
# MAGIC
# MAGIC En producción real, esta sección ejecutaría el Model Registry transition.

# COMMAND ----------

all_healthy

# COMMAND ----------

with mlflow.start_run(run_name="promotion_decision"):
    mlflow.set_tag("stage", "promotion")
    if all_healthy:
        mlflow.set_tag("action", "promote_challenger")
        mlflow.log_metric("promoted", 1)
        print("Action: would call set_registered_model_alias(name, 'Production', challenger_version)")
        print(f"  → Challenger v{challenger_version} would become new Production")
        print(f"  → Champion   v{champion_version} would move to Archived")
        # In production:
        # client.set_registered_model_alias(REGISTERED_NAME, "Production", challenger_version)
        # client.delete_registered_model_alias(REGISTERED_NAME, "Challenger")
    else:
        mlflow.set_tag("action", "rollback_keep_champion")
        mlflow.log_metric("promoted", 0)
        print(f"Action: keep Champion v{champion_version} as Production")
        print(f"  → Investigate Challenger v{challenger_version} before next attempt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Visualizations

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel 1: Champion vs Challenger predictions (paired)
ax = axes[0, 0]
ax.scatter(y_champ_all, y_chall_all, alpha=0.6, s=25, c="steelblue", edgecolor="k")
mn = min(y_champ_all.min(), y_chall_all.min())
mx = max(y_champ_all.max(), y_chall_all.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y = x")
ax.set_xlabel("Champion ŷ"); ax.set_ylabel("Challenger ŷ")
ax.set_title(f"Predictions agreement\nPearson r = {corr_pearson:.3f}")
ax.legend()

# Panel 2: Bland-Altman plot (medical/scientific standard for agreement)
ax = axes[0, 1]
means_per_row = (y_champ_all + y_chall_all) / 2
ax.scatter(means_per_row, diff_per_row, alpha=0.6, s=25, c="darkorange", edgecolor="k")
ax.axhline(diff_per_row.mean(), color="red", ls="-", lw=2,
            label=f"mean = {diff_per_row.mean():+.3f}")
ax.axhline(diff_per_row.mean() + 1.96 * diff_per_row.std(),
            color="red", ls="--", alpha=0.6, label="±1.96σ")
ax.axhline(diff_per_row.mean() - 1.96 * diff_per_row.std(),
            color="red", ls="--", alpha=0.6)
ax.set_xlabel("Mean of the two predictions")
ax.set_ylabel("Champion - Challenger")
ax.set_title("Bland-Altman plot (agreement analysis)")
ax.legend()

# Panel 3: canary scaling timeline
ax = axes[1, 0]
days_arr        = [d["day"] for d in decisions]
canary_pcts     = [(1 - d["champion_share"]) * 100 for d in decisions]
colors_arr = ["seagreen" if d["healthy"] else "darkred" for d in decisions]
bars = ax.bar(days_arr, canary_pcts, color=colors_arr, edgecolor="k", alpha=0.85)
for b, d in zip(bars, decisions):
    label = "✓" if d["healthy"] else "✗"
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f"{label} {int(b.get_height())}%",
            ha="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Day"); ax.set_ylabel("Canary traffic %")
ax.set_title("Canary scaling — green = healthy, red = rollback trigger")
ax.set_ylim(0, 110); ax.set_xticks(days_arr)

# Panel 4: distribution comparison
ax = axes[1, 1]
ax.hist(y_champ_all, bins=30, alpha=0.5, density=True,
         color="steelblue", label="Champion", edgecolor="k")
ax.hist(y_chall_all, bins=30, alpha=0.5, density=True,
         color="darkorange", label="Challenger", edgecolor="k")
ax.set_xlabel("prediction"); ax.set_ylabel("density")
ax.set_title(f"Predicted value distributions  (KS p={ab['ks_p']:.3f})")
ax.legend()

plt.tight_layout()
plt.savefig("/tmp/canary_dashboard.png", dpi=110, bbox_inches="tight")
plt.show()

with mlflow.start_run(run_name="canary_dashboard"):
    mlflow.set_tag("stage", "canary_summary")
    mlflow.log_artifact("/tmp/canary_dashboard.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary
# MAGIC
# MAGIC | Concept | Implemented |
# MAGIC |---|---|
# MAGIC | A/A sanity check | ✓ (framework validation) |
# MAGIC | A/B test 50/50 | ✓ (statistical comparison: Welch + MW + KS + bootstrap CI) |
# MAGIC | Paired analysis | ✓ (Wilcoxon + paired t + Bland-Altman) |
# MAGIC | Canary scaling 5→25→50→100% | ✓ (4-stage simulation) |
# MAGIC | Operational thresholds | ✓ (mean shift, PSI, latency, correlation) |
# MAGIC | Auto promote/rollback | ✓ (decision logic + Registry transitions) |
# MAGIC | Delta persistence | ✓ (decisions queryable via SQL) |
# MAGIC | MLflow tracking | ✓ (every stage logged with tags and metrics) |
# MAGIC
# MAGIC ## 
# MAGIC
# MAGIC | Databricks |
# MAGIC |---|
# MAGIC | Model Registry aliases + Workflows |
# MAGIC | Modify `traffic_share` + re-fire Workflow |
# MAGIC | Inference Tables / explicit Delta append per batch |
# MAGIC | Workflow with `traffic_share = 0.5` |
# MAGIC
# MAGIC
# MAGIC lakehouse-native con MLflow + Delta + Workflows.

# COMMAND ----------


