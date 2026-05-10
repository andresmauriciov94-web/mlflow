# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Monitoring with temporal simulation
# MAGIC
# MAGIC **Patrón de monitoring de modelos en producción.**
# MAGIC
# MAGIC Una vez el modelo está en producción nos importan 3 cosas:
# MAGIC 1. **Data drift** — ¿las features que entran hoy se parecen a las del training?
# MAGIC 2. **Prediction drift** — ¿la distribución de predicciones cambió?
# MAGIC 3. **Performance drift** — ¿el modelo sigue siendo preciso?
# MAGIC    (solo medible cuando llega ground truth)
# MAGIC
# MAGIC **Simulación temporal.** Como el blind son 200 filas estáticas, partimos
# MAGIC el batch en 4 sub-batches simulando 4 días de producción. En cada día
# MAGIC inyectamos drift sintético creciente (shift de medias, cambio de varianzas)
# MAGIC para demostrar que el sistema lo detecta.
# MAGIC
# MAGIC **Métricas:**
# MAGIC - **PSI (Population Stability Index)** — métrica estándar industria
# MAGIC - **KS test (Kolmogorov-Smirnov)** — basada en CDFs
# MAGIC - **Wasserstein distance** — sensible a colas
# MAGIC
# MAGIC Reglas de pulgar PSI: < 0.10 estable, 0.10-0.25 warning, > 0.25 alert.

# COMMAND ----------

import os, time, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
from pyspark.sql import functions as F
from scipy.stats import ks_2samp, wasserstein_distance

EXPERIMENT_NAME = "/Users/your.email@company.com/regression_monitoring"  # EDIT
mlflow.set_experiment(EXPERIMENT_NAME)
plt.rcParams["figure.dpi"] = 110

REFERENCE_TBL    = "main.default.regression_training_reference"
PREDICTIONS_TBL  = "main.default.regression_predictions"
MONITORING_TBL   = "main.default.regression_monitoring"

# Drift thresholds
PSI_WARNING = 0.10
PSI_ALERT   = 0.25
KS_WARNING  = 0.10
KS_ALERT    = 0.20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Drift detection functions

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

ref_pdf = spark.read.table(REFERENCE_TBL).toPandas()
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
# MAGIC **Estrategia.** Partimos el blind (200 filas) en 4 sub-batches (50 filas c/u)
# MAGIC simulando 4 días de producción. En cada día inyectamos drift sintético
# MAGIC creciente sobre features clave. El sistema debe detectarlo y escalar la severidad.
# MAGIC
# MAGIC | Día | Drift inyectado |
# MAGIC |---|---|
# MAGIC | 1 | Sin drift (baseline) |
# MAGIC | 2 | Shift +0.5σ en feature_2 |
# MAGIC | 3 | Shift +1.0σ en feature_2 + variance ×1.5 en feature_13 |
# MAGIC | 4 | Drift severo: shift +2σ + variance ×2 + nueva correlación |

# COMMAND ----------

# Read the blind input for re-simulation
blind_pdf = spark.read.table("main.default.regression_blind_input").toPandas()
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
# MAGIC FROM main.default.regression_monitoring
# MAGIC ORDER BY day

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Days requiring intervention (>= warning)
# MAGIC SELECT count(*) as days_with_drift
# MAGIC FROM main.default.regression_monitoring
# MAGIC WHERE severity IN ('warning', 'alert')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Production hookup
# MAGIC
# MAGIC ### Databricks SQL Alerts
# MAGIC En Databricks UI → **SQL** → **Alerts** → New Alert:
# MAGIC ```sql
# MAGIC SELECT count(*) FROM main.default.regression_monitoring
# MAGIC WHERE severity = 'alert' AND timestamp >= current_date() - INTERVAL 1 DAY
# MAGIC ```
# MAGIC Trigger: `value > 0` → email / Slack notification.
# MAGIC
# MAGIC ### Databricks Workflow
# MAGIC Encadenar `02_batch_inference.py` → `03_monitoring.py` como dos tasks
# MAGIC del mismo Job. Si monitoring detecta drift severo, se puede gatillar
# MAGIC un re-training automático.
# MAGIC
# MAGIC ### Action thresholds
# MAGIC | Severity | Action |
# MAGIC |---|---|
# MAGIC | OK | continue serving |
# MAGIC | Warning | log + investigate within 1 week |
# MAGIC | Alert | manual review + consider retraining |
# MAGIC | Critical (3+ consecutive alerts) | auto-trigger retraining workflow |
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

