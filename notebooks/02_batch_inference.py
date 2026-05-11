# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Batch Inference
# MAGIC
# MAGIC **Standard batch inference pattern in Databricks:**
# MAGIC 1. Read input Delta table
# MAGIC 2. Load champion model from MLflow Registry (`@Production`)
# MAGIC 3. Predict
# MAGIC 4. Persist as Delta table
# MAGIC 5. Log run in MLflow for auditability
# MAGIC
# MAGIC In real production, this notebook would be a **Databricks Job** scheduled
# MAGIC with cron (e.g., daily at 2 AM) or a file-arrival trigger.
# MAGIC
# MAGIC **For this case (200 blind rows)** we execute it manually, but the
# MAGIC pattern is identical to a nightly job over millions of rows.

# COMMAND ----------

# MAGIC %pip install -q catboost
# MAGIC

# COMMAND ----------

import os, time
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

EXPERIMENT_NAME = "/Users/avalderrama@colombina.com/regression_inference"
mlflow.set_experiment(EXPERIMENT_NAME)

# Paths 
INPUT_TABLE      = "hr.agent.blind_test_data"
PREDICTIONS_TBL  = "hr.agent.regression_predictions"
REFERENCE_TBL    = "hr.agent.regression_training_reference"
REGISTERED_NAME  = "regression_20feat_champion"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Persist blind data as Delta input table
# MAGIC
# MAGIC En producción este Delta lo escribiría otro pipeline (data engineering),
# MAGIC y este notebook solo lo leería. Lo persistimos aquí para reproducibilidad.

# COMMAND ----------

blind_sdf = spark.table(INPUT_TABLE)
print(f"Blind data: {blind_sdf.count()} × {len(blind_sdf.columns)}")
print(f"Persisted as Delta: {INPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load champion from MLflow Registry
# MAGIC
# MAGIC Important: cargamos el modelo **vía Registry**, no desde un pickle. Esto
# MAGIC garantiza que el Pipeline completo (FE + scaler + selector + modelo) viene
# MAGIC empaquetado y versionado.

# COMMAND ----------

# Try alias first (Unity Catalog), fall back to stage (classic Registry)
client = MlflowClient()
model_uri = None
model_version_str = None
try:
    mv = client.get_model_version_by_alias(REGISTERED_NAME, "Production")
    model_uri = f"models:/{REGISTERED_NAME}@Production"
    model_version_str = mv.version
    print(f"Loading via alias: {model_uri}  (v{model_version_str})")
except Exception:
    try:
        versions = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])
        if versions:
            mv = versions[0]
            model_uri = f"models:/{REGISTERED_NAME}/Production"
            model_version_str = mv.version
            print(f"Loading via stage: {model_uri}  (v{model_version_str})")
    except Exception as e:
        raise RuntimeError(f"Could not resolve {REGISTERED_NAME} in Registry: {e}")

model = mlflow.sklearn.load_model(model_uri)
print(f"Model class: {type(model.named_steps['model']).__name__}")
print(f"Pipeline steps: {[s[0] for s in model.steps]}")

# Inspect what version we loaded — for traceability
mv_full = client.get_model_version(REGISTERED_NAME, model_version_str)
print(f"\nVersion {model_version_str} tags:")
for tag, val in mv_full.tags.items():
    print(f"  {tag}: {val}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Predict the batch

# COMMAND ----------

with mlflow.start_run(run_name=f"batch_inference_{datetime.now():%Y%m%d_%H%M%S}"):
    mlflow.set_tags({
        "stage": "batch_inference",
        "model_name": REGISTERED_NAME,
        "model_version": model_version_str,
        "input_table": INPUT_TABLE,
    })

    # Read input
    input_sdf = spark.read.table(INPUT_TABLE)
    input_pdf = input_sdf.toPandas()
    n_rows = len(input_pdf)
    mlflow.log_metric("input_rows", n_rows)

    # Predict
    t0 = time.time()
    X_input = input_pdf.values
    predictions = model.predict(X_input)
    inference_time = time.time() - t0
    mlflow.log_metric("inference_time_sec", inference_time)
    mlflow.log_metric("inference_throughput_rows_per_sec", n_rows / max(inference_time, 1e-6))

    # Build output
    output_pdf = input_pdf.copy()
    output_pdf["prediction"]           = predictions
    output_pdf["prediction_timestamp"] = datetime.now()
    output_pdf["model_version"]        = model_version_str

    # Statistics on predictions
    mlflow.log_metric("pred_mean",  float(predictions.mean()))
    mlflow.log_metric("pred_std",   float(predictions.std()))
    mlflow.log_metric("pred_min",   float(predictions.min()))
    mlflow.log_metric("pred_max",   float(predictions.max()))

    print(f"Predictions: n={n_rows}  range=[{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"  mean = {predictions.mean():.3f}  std = {predictions.std():.3f}")
    print(f"  inference time: {inference_time:.2f}s ({n_rows/inference_time:.0f} rows/sec)")

    # Persist as Delta — APPEND mode (every run adds a new batch)
    out_sdf = spark.createDataFrame(output_pdf)
    (out_sdf.write.format("delta").mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(PREDICTIONS_TBL))
    print(f"Predictions appended to {PREDICTIONS_TBL}")
    mlflow.log_metric("output_rows_written", n_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify output via SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   model_version,
# MAGIC   COUNT(*)                  as n_predictions,
# MAGIC   ROUND(AVG(prediction), 3)  as pred_mean,
# MAGIC   ROUND(STDDEV(prediction),3) as pred_std,
# MAGIC   ROUND(MIN(prediction), 2)   as pred_min,
# MAGIC   ROUND(MAX(prediction), 2)   as pred_max,
# MAGIC   MAX(prediction_timestamp)   as last_run
# MAGIC FROM hr.agent.regression_predictions
# MAGIC GROUP BY model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. How this becomes a production Job
# MAGIC
# MAGIC In Databricks UI:
# MAGIC 1. **Workflows** → **Create Job**
# MAGIC 2. Task type: Notebook → select `02_batch_inference.py`
# MAGIC 3. Cluster: **Job cluster** (auto-starts, saves costs)
# MAGIC 4. Schedule: cron (e.g., `0 2 * * *` = every day at 2 AM)
# MAGIC 5. Notifications: email / Slack on failure
# MAGIC 6. Retries: 3 before marking as failed
# MAGIC 7. (Optional) File arrival trigger: runs when new data appears in the Volume
# MAGIC
# MAGIC Each job run generates a new MLflow run. The next notebook
# MAGIC (`03_monitoring`) consumes these runs to detect drift.

# COMMAND ----------


