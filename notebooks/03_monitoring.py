# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Monitoring with Temporal Simulation
# MAGIC
# MAGIC **Model monitoring pattern in production.**
# MAGIC
# MAGIC Once the model is in production, we care about three things:
# MAGIC 1. **Data drift** — Do today's features look like those in training?
# MAGIC 2. **Prediction drift** — Has the prediction distribution changed?
# MAGIC 3. **Performance drift** — Is the model still accurate?
# MAGIC    (only measurable when ground truth arrives)
# MAGIC
# MAGIC **Temporal simulation.** Since the blind set contains 200 static rows, we split
# MAGIC the batch into 4 sub-batches simulating 4 days of production. Each day,
# MAGIC we inject increasing synthetic drift (mean shifts, variance changes)
# MAGIC to demonstrate that the system detects it.
# MAGIC
# MAGIC **Metrics:**
# MAGIC - **PSI (Population Stability Index)** — industry standard metric
# MAGIC - **KS test (Kolmogorov-Smirnov)** — based on CDFs
# MAGIC - **Wasserstein distance** — sensitive to tails
# MAGIC
# MAGIC PSI rule of thumb: < 0.10 stable, 0.10-0.25 warning, > 0.25 alert.

# COMMAND ----------

import os, time, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
from pyspark.sql import functions as F
from scipy.stats import ks_2samp, wasserstein_distance

EXPERIMENT_NAME = "/Users/xxx@xxx.com/regression_monitoring"  # EDIT
mlflow.set_experiment(EXPERIMENT_NAME)
plt.rcParams["figure.dpi"] = 110

REFERENCE_TBL    = "xxx.regression_training_reference"
PREDICTIONS_TBL  = "xxx.regression_predictions"
MONITORING_TBL   = "xxx.regression_monitoring"

# Drift thresholds
PSI_WARNING = 0.10
PSI_ALERT   = 0.25
KS_WARNING  = 0.10
KS_ALERT    = 0.20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Drift detection functions

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
        PSI = Σ (current_pct - reference_pct) × ln(current_pct / reference_pct)
    Bins are defined from reference quantiles. Smoothed with epsilon to avoid log(0).
    """
    eps = 1e-6
    # Quantile bins from reference
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))
    if len(bin_edges) < 3:
        return 0.0
    bin_edges[0]  -= eps
    bin_edges[-1] += eps
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def drift_metrics(ref: np.ndarray, cur: np.ndarray) -> dict:
    """All-in-one drift summary for a single feature."""
    psi = compute_psi(ref, cur)
    ks_stat, ks_p = ks_2samp(ref, cur)
    w_dist = wasserstein_distance(ref, cur)
    # Severity classification
    if psi >= PSI_ALERT or ks_stat >= KS_ALERT:
        severity = "alert"
    elif psi >= PSI_WARNING or ks_stat >= KS_WARNING:
        severity = "warning"
    else:
        severity = "ok"
    return {
        "psi":         psi,
        "ks_stat":     float(ks_stat),
        "ks_pvalue":   float(ks_p),
        "wasserstein": float(w_dist),
        "ref_mean":    float(ref.mean()),
        "cur_mean":    float(cur.mean()),
        "ref_std":     float(ref.std()),
        "cur_std":     float(cur.std()),
        "mean_shift":  float(cur.mean() - ref.mean()),
        "severity":    severity,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load reference (training) distributions

# COMMAND ----------

ref_pdf = spark.read.table(REFERENCE_TBL).toPandas() #(predicciones fuera del fold).
print(f"Reference: {ref_pdf.shape}")

reference_features    = ref_pdf[[c for c in ref_pdf.columns
                                  if c.startswith("feature_")]].values
reference_predictions = ref_pdf["oof_prediction"].values
reference_target      = ref_pdf["target"].values
print(f"Reference features: {reference_features.shape}")
print(f"Reference predictions: μ={reference_predictions.mean():.3f}  σ={reference_predictions.std():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load current production batch + load model

# COMMAND ----------

# MAGIC %pip install -q catboost

# COMMAND ----------

# Load all blind predictions made so far
preds_sdf = spark.read.table(PREDICTIONS_TBL)
preds_pdf = preds_sdf.toPandas()
print(f"Production predictions to date: {len(preds_pdf)}")

# Reload model for re-prediction during simulation
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/regression_20feat_champion@Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Temporal simulation
# MAGIC
# MAGIC **Strategy.** We split the blind set (200 rows) into 4 sub-batches (50 rows each)
# MAGIC simulating 4 days of production. Each day we inject increasing synthetic drift
# MAGIC on key features. The system should detect it and escalate severity.
# MAGIC
# MAGIC  Day | Injected drift |
# MAGIC ---|---|
# MAGIC  1 | No drift (baseline) |
# MAGIC  2 | Shift +0.5σ in feature_2 |
# MAGIC  3 | Shift +1.0σ in feature_2 + variance ×1.5 in feature_13 |
# MAGIC  4 | Severe drift: shift +2σ + variance ×2 + new correlation |

# COMMAND ----------

# Read the blind input for re-simulation
blind_pdf = spark.read.table("xxx.blind_test_data").toPandas()
n_total = len(blind_pdf)
batch_size = n_total // 4  # 50 each

# Use known std from reference
ref_means = reference_features.mean(axis=0)
ref_stds  = reference_features.std(axis=0)

simulated_batches = []
for day in range(1, 5):
    start, end = (day - 1) * batch_size, day * batch_size
    batch = blind_pdf.iloc[start:end].copy()

    if day == 1:
        pass  # no drift
    elif day == 2:
        batch["feature_2"]  += 0.5 * ref_stds[2]
    elif day == 3:
        batch["feature_2"]  += 1.0 * ref_stds[2]
        center_13 = batch["feature_13"].mean()
        batch["feature_13"] = center_13 + (batch["feature_13"] - center_13) * 1.5
    elif day == 4:
        batch["feature_2"]  += 2.0 * ref_stds[2]
        center_13 = batch["feature_13"].mean()
        batch["feature_13"] = center_13 + (batch["feature_13"] - center_13) * 2.0
        # Inject artificial correlation
        batch["feature_5"] += 0.3 * batch["feature_2"]

    batch["simulated_day"] = day
    batch["batch_timestamp"] = datetime.now() - timedelta(days=4-day)
    simulated_batches.append(batch)

simulated_full = pd.concat(simulated_batches, ignore_index=True)
print(f"Simulated 4 days × {batch_size} rows = {len(simulated_full)} total")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run monitoring per simulated day

# COMMAND ----------

monitoring_records = []
feature_cols = [f"feature_{i}" for i in range(20)]

for day in range(1, 5):
    day_batch = simulated_full[simulated_full["simulated_day"] == day]
    X_day = day_batch[feature_cols].values

    with mlflow.start_run(run_name=f"monitoring_day_{day}"):
        mlflow.set_tags({
            "stage":        "monitoring",
            "simulated_day": str(day),
            "batch_size":    str(len(day_batch)),
        })

        # ---- Per-feature drift ----
        per_feature_drift = []
        for i, col in enumerate(feature_cols):
            metrics = drift_metrics(reference_features[:, i], X_day[:, i])
            metrics["feature"]        = col
            metrics["simulated_day"]  = day
            per_feature_drift.append(metrics)
            mlflow.log_metric(f"psi_{col}",   metrics["psi"])
            mlflow.log_metric(f"ks_{col}",    metrics["ks_stat"])

        feat_drift_df = pd.DataFrame(per_feature_drift)
        max_psi  = feat_drift_df["psi"].max()
        mean_psi = feat_drift_df["psi"].mean()
        n_alerts   = (feat_drift_df["severity"] == "alert").sum()
        n_warnings = (feat_drift_df["severity"] == "warning").sum()

        mlflow.log_metric("feature_psi_max",  max_psi)
        mlflow.log_metric("feature_psi_mean", mean_psi)
        mlflow.log_metric("feature_n_alerts", int(n_alerts))
        mlflow.log_metric("feature_n_warnings", int(n_warnings))

        # ---- Prediction drift ----
        y_pred_day = model.predict(X_day)
        pred_drift = drift_metrics(reference_predictions, y_pred_day)
        for k, v in pred_drift.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"pred_{k}", float(v))

        # ---- Overall severity ----
        if pred_drift["severity"] == "alert" or n_alerts > 2:
            overall = "alert"
        elif pred_drift["severity"] == "warning" or n_warnings > 3:
            overall = "warning"
        else:
            overall = "ok"
        mlflow.set_tag("severity", overall)

        # ---- Print summary ----
        print(f"\n=== Day {day} ===")
        print(f"  Feature PSI: max={max_psi:.3f}, mean={mean_psi:.3f}  "
              f"(alerts={n_alerts}, warnings={n_warnings})")
        print(f"  Prediction PSI: {pred_drift['psi']:.3f}  "
              f"({pred_drift['severity']})")
        print(f"  Overall severity: {overall.upper()}")
        if n_alerts > 0:
            top_alerts = feat_drift_df[feat_drift_df["severity"] == "alert"] \
                .nlargest(3, "psi")
            print("  Top features with drift:")
            for _, r in top_alerts.iterrows():
                print(f"    {r['feature']}: PSI={r['psi']:.3f}, "
                      f"mean shift={r['mean_shift']:+.2f}")

        # ---- Record for monitoring table ----
        monitoring_records.append({
            "day":               day,
            "timestamp":         day_batch["batch_timestamp"].iloc[0],
            "n_rows":            len(day_batch),
            "feature_psi_max":   max_psi,
            "feature_psi_mean":  mean_psi,
            "feature_n_alerts":  int(n_alerts),
            "feature_n_warnings": int(n_warnings),
            "prediction_psi":    pred_drift["psi"],
            "prediction_ks":     pred_drift["ks_stat"],
            "prediction_mean":   float(y_pred_day.mean()),
            "severity":          overall,
        })

# COMMAND ----------

m

# COMMAND ----------

# DBTITLE 1,Baseline 200 filas sin drift
# Baseline check: ALL 200 blind rows as 1 solo día (sin drift, sin split)
X_full_blind = blind_pdf[feature_cols].values

print("=== Baseline: 200 filas vs 800 referencia (sin drift) ===")
print(f"{'Feature':<12} {'PSI':>8} {'KS stat':>8} {'Mean shift':>12} {'Severity'}")
print("-" * 55)

alerts_baseline = 0
warnings_baseline = 0
baseline_results = []
for i, col in enumerate(feature_cols):
    m = drift_metrics(reference_features[:, i], X_full_blind[:, i])
    flag = " ⚠️" if m['severity'] == 'alert' else (" ⚡" if m['severity'] == 'warning' else "")
    if m['severity'] == 'alert': alerts_baseline += 1
    if m['severity'] == 'warning': warnings_baseline += 1
    baseline_results.append({"feature": col, **m})
    print(f"{col:<12} {m['psi']:>8.4f} {m['ks_stat']:>8.4f} {m['mean_shift']:>+12.3f}  {m['severity']}{flag}")

# Prediction drift con 200 filas
y_pred_full = model.predict(X_full_blind)
pred_m = drift_metrics(reference_predictions, y_pred_full)

print(f"\n--- Prediction drift (200 filas) ---")
print(f"  PSI = {pred_m['psi']:.4f}  KS = {pred_m['ks_stat']:.4f}  Severity: {pred_m['severity']}")
print(f"  Ref mean={pred_m['ref_mean']:.3f}  Blind mean={pred_m['cur_mean']:.3f}")
print(f"\nResumen: {alerts_baseline} alerts, {warnings_baseline} warnings de 20 features")

# Log to MLflow
with mlflow.start_run(run_name="baseline_200rows_no_drift"):
    mlflow.set_tag("stage", "monitoring_baseline")
    mlflow.set_tag("batch_size", "200")
    mlflow.set_tag("drift_injected", "none")
    mlflow.log_metric("feature_psi_max", max(r['psi'] for r in baseline_results))
    mlflow.log_metric("feature_psi_mean", np.mean([r['psi'] for r in baseline_results]))
    mlflow.log_metric("feature_n_alerts", alerts_baseline)
    mlflow.log_metric("feature_n_warnings", warnings_baseline)
    mlflow.log_metric("prediction_psi", pred_m['psi'])
    mlflow.log_metric("prediction_ks", pred_m['ks_stat'])
    mlflow.log_metric("prediction_mean", float(y_pred_full.mean()))
    for r in baseline_results:
        mlflow.log_metric(f"psi_{r['feature']}", r['psi'])
        mlflow.log_metric(f"ks_{r['feature']}", r['ks_stat'])
    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv("/tmp/baseline_200rows.csv", index=False)
    mlflow.log_artifact("/tmp/baseline_200rows.csv")
    print(f"\n✅ Logged to MLflow run 'baseline_200rows_no_drift'")

# COMMAND ----------

# DBTITLE 1,Visualizaciones baseline drift
# Visualizaciones del baseline (200 filas sin drift)
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Panel 1: PSI por feature con umbrales
ax = axes[0, 0]
colors_psi = ['darkred' if r['psi'] >= PSI_ALERT else 'orange' if r['psi'] >= PSI_WARNING else 'steelblue'
              for r in baseline_results]
ax.barh(range(20), [r['psi'] for r in baseline_results], color=colors_psi, edgecolor='k', alpha=0.8)
ax.axvline(PSI_WARNING, color='orange', ls='--', lw=1.5, label=f'Warning ({PSI_WARNING})')
ax.axvline(PSI_ALERT, color='red', ls='--', lw=1.5, label=f'Alert ({PSI_ALERT})')
ax.set_yticks(range(20))
ax.set_yticklabels(feature_cols, fontsize=8)
ax.set_xlabel('PSI')
ax.set_title('PSI por feature (baseline 200 filas)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3, axis='x')

# Panel 2: Distribución reference vs blind para top 4 features con mayor PSI
ax = axes[0, 1]
top4_idx = sorted(range(20), key=lambda i: baseline_results[i]['psi'], reverse=True)[:4]
colors_dist = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for rank, idx in enumerate(top4_idx):
    col = feature_cols[idx]
    ax.hist(reference_features[:, idx], bins=25, alpha=0.3, density=True,
            color=colors_dist[rank], label=f'{col} (ref)')
    ax.hist(X_full_blind[:, idx], bins=25, alpha=0.5, density=True,
            color=colors_dist[rank], histtype='step', lw=2,
            label=f'{col} (blind) PSI={baseline_results[idx]["psi"]:.3f}')
ax.set_xlabel('Valor')
ax.set_ylabel('Densidad')
ax.set_title('Top 4 features: Reference vs Blind')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# Panel 3: Mean shift en σ por feature
ax = axes[1, 0]
shifts = [(r['cur_mean'] - r['ref_mean']) / r['ref_std'] if r['ref_std'] > 0 else 0
          for r in baseline_results]
colors_shift = ['darkorange' if abs(s) > 0.1 else 'steelblue' for s in shifts]
ax.bar(range(20), shifts, color=colors_shift, edgecolor='k', alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.axhline(+0.2, color='red', ls=':', alpha=0.5, label='±0.2σ')
ax.axhline(-0.2, color='red', ls=':', alpha=0.5)
ax.set_xticks(range(20))
ax.set_xticklabels([f'f{i}' for i in range(20)], fontsize=8)
ax.set_ylabel('Shift (σ)')
ax.set_title('Mean shift por feature (en desviaciones estándar)')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 4: PSI vs KS stat (concordancia entre métricas)
ax = axes[1, 1]
psi_vals = [r['psi'] for r in baseline_results]
ks_vals = [r['ks_stat'] for r in baseline_results]
ax.scatter(psi_vals, ks_vals, c='steelblue', s=80, edgecolor='k', alpha=0.8)
for i in range(20):
    if psi_vals[i] > PSI_WARNING or ks_vals[i] > KS_WARNING:
        ax.annotate(f'f{i}', (psi_vals[i], ks_vals[i]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
ax.axvline(PSI_WARNING, color='orange', ls='--', alpha=0.6)
ax.axhline(KS_WARNING, color='orange', ls='--', alpha=0.6)
ax.axvline(PSI_ALERT, color='red', ls='--', alpha=0.6)
ax.axhline(KS_ALERT, color='red', ls='--', alpha=0.6)
ax.set_xlabel('PSI')
ax.set_ylabel('KS stat')
ax.set_title('Concordancia PSI vs KS (deberían correlacionar)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/baseline_visualizations.png', dpi=110, bbox_inches='tight')
plt.show()

# Log visualizaciones a MLflow
with mlflow.start_run(run_name="baseline_visualizations"):
    mlflow.set_tag("stage", "monitoring_baseline")
    mlflow.set_tag("chart_type", "drift_overview")
    mlflow.log_artifact('/tmp/baseline_visualizations.png')
    mlflow.log_metric("feature_psi_max", max(psi_vals))
    mlflow.log_metric("feature_ks_max", max(ks_vals))
    mlflow.log_metric("max_shift_sigma", max(abs(s) for s in shifts))
    print('\n✅ Visualizaciones logueadas en MLflow run "baseline_visualizations"')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist monitoring history as Delta

# COMMAND ----------

monitoring_pdf = pd.DataFrame(monitoring_records)
print(monitoring_pdf.round(4).to_string(index=False))

(spark.createDataFrame(monitoring_pdf).write
    .format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MONITORING_TBL))
print(f"\nMonitoring history → {MONITORING_TBL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visual dashboard

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: PSI evolution over time
days = monitoring_pdf["day"].values
ax = axes[0, 0]
ax.plot(days, monitoring_pdf["feature_psi_max"], "o-",
         color="darkred",   lw=2, label="max feature PSI")
ax.plot(days, monitoring_pdf["feature_psi_mean"], "s-",
         color="steelblue", lw=2, label="mean feature PSI")
ax.plot(days, monitoring_pdf["prediction_psi"], "^-",
         color="seagreen",  lw=2, label="prediction PSI")
ax.axhline(PSI_WARNING, color="orange", ls="--", alpha=0.6, label=f"warning ({PSI_WARNING})")
ax.axhline(PSI_ALERT,   color="red",    ls="--", alpha=0.6, label=f"alert ({PSI_ALERT})")
ax.set_xlabel("Simulated day"); ax.set_ylabel("PSI")
ax.set_title("PSI evolution — drift detection over time")
ax.legend(); ax.set_xticks(days); ax.grid(alpha=0.3)

# Panel 2: severity bar
ax = axes[0, 1]
sev_colors = {"ok": "seagreen", "warning": "orange", "alert": "darkred"}
colors = [sev_colors[s] for s in monitoring_pdf["severity"]]
ax.bar(days, [1]*len(days), color=colors, edgecolor="k", alpha=0.85)
for i, (d, s) in enumerate(zip(days, monitoring_pdf["severity"])):
    ax.text(d, 0.5, s.upper(), ha="center", va="center",
             color="white", fontsize=12, fontweight="bold")
ax.set_xlabel("Simulated day"); ax.set_yticks([])
ax.set_title("Daily severity classification")
ax.set_xticks(days)

# Panel 3: number of alerting features per day
ax = axes[1, 0]
width = 0.35
ax.bar(days - width/2, monitoring_pdf["feature_n_alerts"], width,
        color="darkred",  edgecolor="k", label="alerts")
ax.bar(days + width/2, monitoring_pdf["feature_n_warnings"], width,
        color="orange",   edgecolor="k", label="warnings")
ax.set_xlabel("Simulated day"); ax.set_ylabel("# features")
ax.set_title("Features above drift thresholds per day")
ax.legend(); ax.set_xticks(days)

# Panel 4: prediction distribution shift
ax = axes[1, 1]
ax.hist(reference_predictions, bins=30, alpha=0.4, density=True,
         label="reference (training OOF)", color="gray")
for day in days:
    day_batch = simulated_full[simulated_full["simulated_day"] == day]
    y_pred = model.predict(day_batch[feature_cols].values)
    ax.hist(y_pred, bins=30, alpha=0.4, density=True, label=f"day {day}")
ax.set_xlabel("prediction"); ax.set_ylabel("density")
ax.set_title("Prediction distribution shift")
ax.legend()

plt.tight_layout()
plt.savefig("/tmp/monitoring_dashboard.png", dpi=110, bbox_inches="tight")
plt.show()

# Log dashboard as artifact in a final summary run
with mlflow.start_run(run_name="monitoring_dashboard"):
    mlflow.set_tag("stage", "monitoring_summary")
    mlflow.log_artifact("/tmp/monitoring_dashboard.png")
    mlflow.log_metric("days_alert",   int((monitoring_pdf["severity"] == "alert").sum()))
    mlflow.log_metric("days_warning", int((monitoring_pdf["severity"] == "warning").sum()))
    mlflow.log_metric("days_ok",      int((monitoring_pdf["severity"] == "ok").sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. SQL views for ongoing dashboard

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily severity timeline
# MAGIC SELECT day, timestamp, severity, prediction_psi, feature_psi_max,
# MAGIC        feature_n_alerts, feature_n_warnings
# MAGIC FROM xxx.regression_monitoring
# MAGIC ORDER BY day

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Days requiring intervention (>= warning)
# MAGIC SELECT count(*) as days_with_drift
# MAGIC FROM xxx.regression_monitoring
# MAGIC WHERE severity IN ('warning', 'alert')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC | Component | Status |
# MAGIC |---|---|
# MAGIC | Reference distributions | loaded from training Delta table |
# MAGIC | Drift detection | PSI + KS + Wasserstein per feature + per prediction |
# MAGIC | Temporal simulation | 4 days with progressive drift injection |
# MAGIC | Monitoring history | Delta table with one row per day |
# MAGIC | MLflow logging | one run per day with all metrics |
# MAGIC | Visual dashboard | inline + logged as artifact |
# MAGIC | SQL queries | for ongoing dashboard / alerts |
# MAGIC

# COMMAND ----------

# Compare distributions of training vs blind + log to MLflow
train_pdf = spark.table("xxx.training_data").toPandas()
blind_pdf = spark.table("xxx.blind_test_data").toPandas()

feature_cols = [f"feature_{i}" for i in range(20)]

comparison_rows = []
for col in feature_cols:
    t_mean = train_pdf[col].mean()
    b_mean = blind_pdf[col].mean()
    t_std = train_pdf[col].std()
    b_std = blind_pdf[col].std()
    shift_sigma = (b_mean - t_mean) / t_std if t_std > 0 else 0
    comparison_rows.append({
        "feature": col,
        "train_mean": t_mean, "blind_mean": b_mean,
        "train_std": t_std, "blind_std": b_std,
        "shift_sigma": shift_sigma
    })

comparison_df = pd.DataFrame(comparison_rows)
print(f"Training: {train_pdf.shape}  |  Blind: {blind_pdf.shape}")
print(comparison_df.round(4).to_string(index=False))

# Log to MLflow
with mlflow.start_run(run_name="distribution_comparison"):
    mlflow.set_tag("stage", "distribution_analysis")
    mlflow.log_metric("max_abs_shift_sigma", float(comparison_df["shift_sigma"].abs().max()))
    mlflow.log_metric("mean_abs_shift_sigma", float(comparison_df["shift_sigma"].abs().mean()))
    mlflow.log_metric("n_features_shifted_gt_1sigma", int((comparison_df["shift_sigma"].abs() > 1).sum()))
    mlflow.log_metric("n_features_shifted_gt_2sigma", int((comparison_df["shift_sigma"].abs() > 2).sum()))
    for _, row in comparison_df.iterrows():
        mlflow.log_metric(f"shift_sigma_{row['feature']}", row["shift_sigma"])
    comparison_df.to_csv("/tmp/train_vs_blind_distributions.csv", index=False)
    mlflow.log_artifact("/tmp/train_vs_blind_distributions.csv")
    print(f"\n✅ Logged to MLflow run 'distribution_comparison'")
    print(f"   Max |shift|: {comparison_df['shift_sigma'].abs().max():.3f}σ")
    print(f"   Features > 2σ: {(comparison_df['shift_sigma'].abs() > 2).sum()}")
