"""
Spark MLlib regressors — Databricks-native (in-house).

These models train inside the Spark cluster and scale horizontally. On n=800
they don't outperform CatBoost, but the point is to demonstrate fluency with
the native Databricks ML stack: pipelines, vector assembly, distributed
gradient boosting, and integration with MLflow autologging.

Use them when:
    - dataset > a few million rows
    - features come from existing Delta tables
    - you want lineage end-to-end inside Spark
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def build_spark_pipeline(feature_cols, model_kind: str = "gbt", **kw):
    """
    Returns (Pipeline, fit_kwargs_template) for one of:
        - "gbt"  : GBTRegressor (Spark MLlib gradient boosting)
        - "rf"   : RandomForestRegressor (Spark MLlib bagged trees)
        - "glr"  : GeneralizedLinearRegression (baseline)

    The pipeline assembles columns → VectorAssembler → StandardScaler → model.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.regression import (
        GBTRegressor, RandomForestRegressor, GeneralizedLinearRegression,
    )

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                                handleInvalid="error")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)

    if model_kind == "gbt":
        regressor = GBTRegressor(
            featuresCol="features", labelCol="target",
            maxIter=kw.get("maxIter", 200),
            maxDepth=kw.get("maxDepth", 5),
            stepSize=kw.get("stepSize", 0.05),
            subsamplingRate=kw.get("subsamplingRate", 0.85),
            minInstancesPerNode=kw.get("minInstancesPerNode", 5),
            seed=kw.get("seed", 42),
        )
    elif model_kind == "rf":
        regressor = RandomForestRegressor(
            featuresCol="features", labelCol="target",
            numTrees=kw.get("numTrees", 300),
            maxDepth=kw.get("maxDepth", 12),
            minInstancesPerNode=kw.get("minInstancesPerNode", 3),
            featureSubsetStrategy=kw.get("featureSubsetStrategy", "sqrt"),
            seed=kw.get("seed", 42),
        )
    elif model_kind == "glr":
        regressor = GeneralizedLinearRegression(
            featuresCol="features", labelCol="target",
            family=kw.get("family", "gaussian"),
            link=kw.get("link", "identity"),
            regParam=kw.get("regParam", 0.1),
            maxIter=kw.get("maxIter", 100),
        )
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")

    return Pipeline(stages=[assembler, scaler, regressor])


def evaluate_spark_model(spark_pipeline_model, sdf, label_col: str = "target"):
    """
    Evaluate a fitted Spark pipeline on a Spark DataFrame.
    Returns dict with R², RMSE, MAE.
    """
    from pyspark.ml.evaluation import RegressionEvaluator
    preds = spark_pipeline_model.transform(sdf)
    metrics = {}
    for metric in ("r2", "rmse", "mae"):
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction",
                                        metricName=metric)
        metrics[metric] = float(evaluator.evaluate(preds))
    return metrics, preds


def spark_predictions_to_numpy(preds_sdf) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (y_true, y_pred) into numpy for downstream metrics/SHAP."""
    pdf = preds_sdf.select("target", "prediction").toPandas()
    return pdf["target"].values, pdf["prediction"].values
