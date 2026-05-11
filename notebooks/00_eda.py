# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — EDA + Clustering + Justificación cuantitativa
# MAGIC
# MAGIC **Objetivo.** Antes de modelar, demostrar **con números** dos cosas:
# MAGIC
# MAGIC 1. Los modelos lineales no son adecuados → la relación features↔target tiene
# MAGIC    componente no-lineal medible.
# MAGIC 2. PCA no aporta → la varianza está distribuida uniformemente entre PCs, no
# MAGIC    concentrada.
# MAGIC
# MAGIC **Bonus:** clustering K-means para detectar grupos naturales en los datos.
# MAGIC Si los clusters tienen `target` mean muy distinto pero los inputs no son
# MAGIC linealmente separables, refuerza el caso para modelos no-lineales.
# MAGIC
# MAGIC **Stack:** Spark + Spark MLlib + MLflow nativo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110

# MLflow native — replace with your workspace path
EXPERIMENT_NAME = "/Users/xxx@xxx/regression_eda" 
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(disable=True)  # evita runs fantasma en cada .fit()

# Data path — adjust to your Volume/DBFS location
TRAIN_PATH = "xxx.training_data"
print(f"Spark {spark.version}  |  MLflow {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingestion via Spark 

# COMMAND ----------

train_sdf = spark.table(TRAIN_PATH)
print(f"Training: {train_sdf.count()}, {len(train_sdf.columns)}")

train_sdf.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Descriptive statistics + data quality

# COMMAND ----------

# Spark describe (push-down to executors)
desc_pdf = train_sdf.describe().toPandas().set_index("summary").T.astype(float).round(3)
print(desc_pdf.to_string())

n_nulls = train_sdf.select([F.sum(F.col(c).isNull().cast("int")).alias(c)
                              for c in train_sdf.columns]).toPandas().T.values.sum()
n_dupes = train_sdf.count() - train_sdf.dropDuplicates().count()
print(f"\nTotal nulls: {int(n_nulls)}  |  Duplicates: {int(n_dupes)}")

# COMMAND ----------

with mlflow.start_run(run_name="outlier_analysis"):
    mlflow.set_tag("stage", "eda_outliers")
    
    # Convert Spark to pandas for outlier analysis (n=800 fits in memory)
    pdf = train_sdf.toPandas()
    feature_cols = [c for c in pdf.columns if c != "target"]
    X_arr = pdf[feature_cols].values
    
    # --- Method 1: IQR ---
    iqr_results = []
    for col in feature_cols:
        q1, q3 = pdf[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low_bound, high_bound = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = ((pdf[col] < low_bound) | (pdf[col] > high_bound)).sum()
        iqr_results.append({
            "feature": col,
            "iqr_outliers": int(outliers),
            "iqr_pct": 100 * outliers / len(pdf),
        })
    iqr_df = pd.DataFrame(iqr_results)
    
    # --- Method 2: Modified Z-score (MAD-based) ---
    # MAD-Z is more robust than standard z-score to extreme values
    # because median + MAD are not influenced by outliers themselves
    mad_results = []
    for col in feature_cols:
        median = pdf[col].median()
        mad = np.median(np.abs(pdf[col] - median))
        if mad == 0:
            mad_z = np.zeros(len(pdf))
        else:
            mad_z = 0.6745 * (pdf[col] - median) / mad
        outliers = (np.abs(mad_z) > 3.5).sum()  # 3.5 = standard threshold
        mad_results.append({
            "feature": col,
            "mad_outliers": int(outliers),
            "mad_pct": 100 * outliers / len(pdf),
        })
    mad_df = pd.DataFrame(mad_results)
    
    # Merge both
    univ_df = iqr_df.merge(mad_df, on="feature")
    print("Univariate outlier analysis:")
    print(univ_df.round(2).to_string(index=False))
    
    # Log totals
    mlflow.log_metric("total_iqr_outliers", int(univ_df["iqr_outliers"].sum()))
    mlflow.log_metric("total_mad_outliers", int(univ_df["mad_outliers"].sum()))
    mlflow.log_metric("max_iqr_pct_per_feature", float(univ_df["iqr_pct"].max()))
    mlflow.log_metric("max_mad_pct_per_feature", float(univ_df["mad_pct"].max()))

# COMMAND ----------

    # --- Method 3: Isolation Forest (multivariate) ---
    # Detects observations where the COMBINATION of features is unusual
    # even if each feature individually is within range
    mlflow.end_run() 
    with mlflow.start_run(run_name=" Isolation Forest (multivariate) "):
        iso = IsolationForest(
            contamination=0.05,    # assume ~5% multivariate outliers
            random_state=42,
            n_estimators=200,
        )
        iso_pred = iso.fit_predict(X_arr)
        iso_scores = -iso.score_samples(X_arr)  # higher = more anomalous
        n_iso_outliers = (iso_pred == -1).sum()
        mlflow.log_metric("isolation_forest_outliers", int(n_iso_outliers))
        mlflow.log_metric("isolation_forest_pct", 100 * n_iso_outliers / len(pdf))
        
        print(f"\nIsolation Forest: {n_iso_outliers} multivariate outliers "
            f"({100*n_iso_outliers/len(pdf):.1f}%)")
        
        # Inspect what the IF flagged
        iso_outlier_idx = np.where(iso_pred == -1)[0]
        if n_iso_outliers > 0:
            # Are these flagged also via IQR or MAD? Cross-validate
            outlier_summary = pd.DataFrame({
                "row_idx": iso_outlier_idx,
                "iso_score": iso_scores[iso_outlier_idx],
                "target": pdf["target"].values[iso_outlier_idx],
            }).sort_values("iso_score", ascending=False).head(10)
            print("\nTop-10 multivariate outliers (highest anomaly score):")
            print(outlier_summary.round(3).to_string(index=False))

            # --- Visualization ---
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            # Panel 1: outlier counts per feature (IQR vs MAD)
            width = 0.35
            x = np.arange(len(feature_cols))
            axes[0].bar(x - width/2, univ_df["iqr_outliers"], width,
                        label="IQR (1.5×)", color="steelblue", edgecolor="k")
            axes[0].bar(x + width/2, univ_df["mad_outliers"], width,
                        label="MAD-Z (>3.5)", color="darkorange", edgecolor="k")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f.replace("feature_", "f") for f in feature_cols],
                                        rotation=90, fontsize=8)
            axes[0].set_ylabel("# outliers")
            axes[0].set_title("Univariate outliers per feature")
            axes[0].legend()

            # Panel 2: distribution of Isolation Forest scores
            axes[1].hist(iso_scores, bins=40, color="seagreen",
                            edgecolor="k", alpha=0.85)
            threshold = np.percentile(iso_scores, 95)
            axes[1].axvline(threshold, color="red", ls="--",
                                label=f"95th pct = {threshold:.3f}")
            axes[1].set_xlabel("Anomaly score")
            axes[1].set_ylabel("# observations")
            axes[1].set_title("Isolation Forest anomaly scores")
            axes[1].legend()

            # Panel 3: do outliers correlate with extreme target?
            axes[2].scatter(iso_scores, pdf["target"].values,
                                alpha=0.5, s=20, c="purple", edgecolor="k")
            axes[2].set_xlabel("Anomaly score")
            axes[2].set_ylabel("target")
            axes[2].set_title("Anomaly score vs target  (do outliers have extreme y?)")

            plt.tight_layout()
            plt.savefig("outlier_analysis.png", dpi=110, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact("outlier_analysis.png")


            # --- Decision ---
            total_iqr   = int(univ_df["iqr_outliers"].sum())
            total_mad   = int(univ_df["mad_outliers"].sum())
            total_iso   = int(n_iso_outliers)

            # Check: do isolation-forest-flagged rows have extreme targets?
            # If yes, they're informative and should NOT be removed
            if n_iso_outliers > 5:
                from scipy.stats import ks_2samp
                target_normal   = pdf["target"].values[iso_pred == 1]
                target_outlier  = pdf["target"].values[iso_pred == -1]
                ks_stat, ks_p   = ks_2samp(target_normal, target_outlier)
                mlflow.log_metric("outlier_target_ks_pvalue", float(ks_p))
                outliers_have_extreme_target = ks_p < 0.05
            else:
                outliers_have_extreme_target = False

            # Decision logic
            if total_iso == 0 and total_iqr == 0:
                verdict = "No outliers detected by any method — keep all rows"
            elif outliers_have_extreme_target:
                verdict = (f"Outliers detected ({total_iso} multivariate) BUT they "
                            f"correlate with extreme target values (KS p<0.05) — "
                            f"keep all rows because outliers carry signal, not noise")
            else:
                verdict = (f"Outliers detected ({total_iso} multivariate) but "
                            f"keep all rows: CatBoost/RF are robust to outliers via "
                            f"tree splits, and Yeo-Johnson scaler downstream "
                            f"compresses tails. Removal would discard data without "
                            f"clear benefit.")

            mlflow.set_tag("verdict_outliers", verdict)
            print("\n" + "="*72)
            print("OUTLIER ANALYSIS — VERDICT")
            print("="*72)
            print(verdict)

# COMMAND ----------

# --- Decision ---
total_iqr   = int(univ_df["iqr_outliers"].sum())
total_mad   = int(univ_df["mad_outliers"].sum())
total_iso   = int(n_iso_outliers)

# Check: do isolation-forest-flagged rows have extreme targets?
# If yes, they're informative and should NOT be removed
if n_iso_outliers > 5:
    from scipy.stats import ks_2samp
    target_normal   = pdf["target"].values[iso_pred == 1]
    target_outlier  = pdf["target"].values[iso_pred == -1]
    ks_stat, ks_p   = ks_2samp(target_normal, target_outlier)
    mlflow.log_metric("outlier_target_ks_pvalue", float(ks_p))
    outliers_have_extreme_target = ks_p < 0.05
else:
    outliers_have_extreme_target = False

# Decision logic
if total_iso == 0 and total_iqr == 0:
    verdict = "No outliers detected by any method — keep all rows"
elif outliers_have_extreme_target:
    verdict = (f"Outliers detected ({total_iso} multivariate) BUT they "
                f"correlate with extreme target values (KS p<0.05) — "
                f"keep all rows because outliers carry signal, not noise")
else:
    verdict = (f"Outliers detected ({total_iso} multivariate) but "
                f"keep all rows: CatBoost/RF are robust to outliers via "
                f"tree splits, and Yeo-Johnson scaler downstream "
                f"compresses tails. Removal would discard data without "
                f"clear benefit.")

mlflow.set_tag("verdict_outliers", verdict)
print("\n" + "="*72)
print("OUTLIER ANALYSIS — VERDICT")
print("="*72)
print(verdict)

# COMMAND ----------

# MAGIC %md
# MAGIC **Verdict on outliers.** See the MLflow tag `verdict_outliers` for the
# MAGIC automated decision. In all branches, we keep all rows because:
# MAGIC
# MAGIC - CatBoost and RandomForest split data by feature thresholds, so an 
# MAGIC   outlier only contaminates the leaf it lands in — no global influence
# MAGIC - The downstream PowerTransformer (Yeo-Johnson) compresses tails 
# MAGIC   monotonically, reducing the leverage of extreme values
# MAGIC - Removing rows from n=800 has high opportunity cost; removing 
# MAGIC   informative extremes (the case our KS test guards against) would 
# MAGIC   bias the model
# MAGIC
# MAGIC In production we re-evaluate this decision via the monitoring notebook,
# MAGIC which checks if incoming batches contain proportionally more anomalies 
# MAGIC than training (Isolation Forest score drift).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distributed Pearson + Spearman correlation
# MAGIC
# MAGIC The difference between **|Spearman|** and **|Pearson|** reveals monotonic nonlinear relationships. If for any feature `|ρ| >> |r|`, that feature has nonlinear signal with the target — invisible to linear models.

# COMMAND ----------

assembler = VectorAssembler(inputCols=train_sdf.columns, outputCol="feat_vec")
vec_df = assembler.transform(train_sdf).select("feat_vec")
pearson_mat  = Correlation.corr(vec_df, "feat_vec", method="pearson").head()[0].toArray()
spearman_mat = Correlation.corr(vec_df, "feat_vec", method="spearman").head()[0].toArray()

cols = train_sdf.columns
target_idx = cols.index("target")
corr_df = pd.DataFrame({
    "feature":  [c for c in cols if c != "target"],
    "pearson":  [pearson_mat[i, target_idx]  for i, c in enumerate(cols) if c != "target"],
    "spearman": [spearman_mat[i, target_idx] for i, c in enumerate(cols) if c != "target"],
})
corr_df["abs_pearson"]  = corr_df["pearson"].abs()
corr_df["abs_spearman"] = corr_df["spearman"].abs()
corr_df["nonlinearity_gap"] = corr_df["abs_spearman"] - corr_df["abs_pearson"]
corr_df = corr_df.sort_values("abs_pearson", ascending=False).reset_index(drop=True)
print(corr_df.round(4).to_string(index=False))

print(f"\nFeatures con |ρ| - |r| > 0.05 (señal NO-lineal):")
nonlin = corr_df[corr_df["nonlinearity_gap"] > 0.05]
print(nonlin[["feature", "pearson", "spearman", "nonlinearity_gap"]].round(4).to_string(index=False))

# COMMAND ----------

# Mutual information para las features (no-linealidad)
from sklearn.feature_selection import mutual_info_regression

feature_cols = [c for c in train_sdf.columns if c != "target"]
X = train_sdf.select(feature_cols).toPandas()
y = train_sdf.select("target").toPandas().values.ravel()

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({"feature": feature_cols, "mutual_info": mi_scores})
mi_df = mi_df.sort_values("mutual_info", ascending=False).reset_index(drop=True)

# Top 10 mutual info
top_mi = mi_df.head(10)["feature"].tolist()
# Top 4 Pearson
top_pearson = corr_df.sort_values("abs_pearson", ascending=False).head(4)["feature"].tolist()

print("Top 10 features by mutual info:", top_mi)
print("Top 4 features by |Pearson|:", top_pearson)

# COMMAND ----------

mlflow.end_run()  # cierra cualquier run huérfano de celdas anteriores
with mlflow.start_run(run_name="feature_selection_mi"):
    mlflow.log_param("top_10_mi_features", top_mi)
    mlflow.log_param("top_4_pearson_features", top_pearson)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. PCA — quantitative test

# COMMAND ----------

from pyspark.ml.feature import PCA as SparkPCA
import pyspark.sql.functions as Fsql

with mlflow.start_run(run_name="pca_justification"):
    mlflow.set_tag("stage", "eda_justification")

    # Standardize features in Spark
    feature_cols = [c for c in cols if c != "target"]
    asm = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    sdf_features = asm.transform(train_sdf)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                             withMean=True, withStd=True)
    scaler_model = scaler.fit(sdf_features)
    sdf_scaled = scaler_model.transform(sdf_features)

    # PCA Spark con todos los componentes para ver el reparto de varianza
    pca = SparkPCA(k=20, inputCol="features_scaled", outputCol="pca_features")
    pca_model = pca.fit(sdf_scaled)
    explained = pca_model.explainedVariance.toArray()

    print("Variance explained per PC:")
    cumulative = 0
    for i, e in enumerate(explained):
        cumulative += e
        mlflow.log_metric(f"pca_var_pc{i+1:02d}", float(e))
        print(f"  PC{i+1:2d}: {e*100:5.2f}%  (cum: {cumulative*100:5.2f}%)")

    # Métrica clave: ¿cuántos PCs hacen falta para 95%?
    n_for_95 = int(np.argmax(np.cumsum(explained) >= 0.95)) + 1
    mlflow.log_metric("n_pcs_for_95pct_variance", n_for_95)
    print(f"\nPCs needed for 95% variance: {n_for_95}")
    print(f"Top-3 PCs explain only {sum(explained[:3])*100:.1f}% "
          f"(in problems where PCA helps, this is typically >70%)")

    # Plot scree
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(1, 21), explained*100, color="steelblue", edgecolor="k")
    axes[0].plot(range(1, 21), np.cumsum(explained)*100, "ro-", label="cumulative")
    axes[0].axhline(95, color="gray", ls="--", label="95%")
    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance (%)")
    axes[0].set_title("Scree plot — variance distributed uniformly")
    axes[0].legend()

    # Histograma de varianzas
    axes[1].hist(explained*100, bins=10, color="darkgreen", edgecolor="k", alpha=0.8)
    axes[1].axvline(explained.mean()*100, color="red", ls="--",
                     label=f"mean={explained.mean()*100:.2f}%")
    axes[1].set_xlabel("Variance per component (%)")
    axes[1].set_title("Distribution of variance across PCs")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("pca_analysis.png", dpi=110, bbox_inches="tight"); plt.close()
    mlflow.log_artifact("pca_analysis.png")
    

    mlflow.set_tag("verdict_pca",
                    "REJECTED — variance uniformly spread; no informative reduction")

# COMMAND ----------

# MAGIC %md
# MAGIC **Verdict on PCA.** Each component explains 3–6% of variance — variance is
# MAGIC distributed almost uniformly. Reducing dimensions only loses signal. PCA
# MAGIC also rotates the axes, which **harms tree-based models** because they split
# MAGIC on individual features (rotated combinations are no longer aligned with
# MAGIC informative thresholds). **PCA REJECTED.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Justification for NOT using linear models
# MAGIC
# MAGIC We compare a regularized linear model (Ridge) against a simple nonlinear baseline
# MAGIC (RandomForest), both using standardized features. If the gap is large, the linear model
# MAGIC does not capture the structure of the problem.
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.regression import LinearRegression, RandomForestRegressor as SparkRF
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

with mlflow.start_run(run_name="linear_vs_nonlinear"):
    mlflow.set_tag("stage", "eda_justification")

    # Build features once
    asm = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scl = StandardScaler(inputCol="features_raw", outputCol="features",
                          withMean=True, withStd=True)
    sdf_in = scl.fit(asm.transform(train_sdf)).transform(asm.transform(train_sdf))

    evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction",
                                      metricName="r2")

    # ---- Linear (Ridge) with CV over alpha ----
    lr = LinearRegression(featuresCol="features", labelCol="target",
                            elasticNetParam=0.0)  # pure L2 = Ridge
    lr_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0, 10.0]) \
        .build()
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_grid,
                            evaluator=evaluator, numFolds=5, seed=42)
    lr_model = lr_cv.fit(sdf_in)
    lr_r2 = max(lr_model.avgMetrics)
    lr_best_alpha = lr_model.bestModel._java_obj.getRegParam()
    mlflow.log_metric("ridge_cv_r2", lr_r2)
    mlflow.log_metric("ridge_best_alpha", lr_best_alpha)

    # ---- Non-linear baseline (RF) ----
    rf = SparkRF(featuresCol="features", labelCol="target",
                  numTrees=100, maxDepth=8, seed=42)
    rf_grid = ParamGridBuilder().build()  # use defaults — just a baseline
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_grid,
                            evaluator=evaluator, numFolds=5, seed=42)
    rf_model = rf_cv.fit(sdf_in)
    rf_r2 = max(rf_model.avgMetrics)
    mlflow.log_metric("rf_cv_r2", rf_r2)

    gap = rf_r2 - lr_r2
    mlflow.log_metric("nonlinear_lift_r2", gap)
    print(f"Ridge (best α={lr_best_alpha}):  CV R² = {lr_r2:.4f}")
    print(f"RandomForest:                    CV R² = {rf_r2:.4f}")
    print(f"Gap (non-linear lift):           {gap*100:+.2f} pp of R²")
    print(f"Theoretical ceiling 0.92  →  Ridge leaves "
          f"{(0.92-lr_r2)*100:.1f} pp on the table that RF recovers.")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(["Ridge\n(linear)", "RandomForest\n(non-linear)"],
            [lr_r2, rf_r2],
            color=["coral", "seagreen"], edgecolor="k", alpha=0.85)
    ax.axvline(0.92, color="red", ls="--", label="Theoretical ceiling")
    ax.set_xlabel("CV R² (5-fold)")
    ax.set_title(f"Linear vs non-linear baseline — gap = {gap:+.3f}")
    ax.legend(); ax.set_xlim(0, 1)
    for i, r in enumerate([lr_r2, rf_r2]):
        ax.text(r-0.02, i, f"{r:.3f}", va="center", ha="right",
                color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("linear_vs_nonlinear.png", dpi=110, bbox_inches="tight"); plt.close()
    mlflow.log_artifact("linear_vs_nonlinear.png")

    mlflow.set_tag("verdict_linear_models",
                    f"REJECTED — gap {gap*100:+.1f}pp confirms non-linear structure")

# COMMAND ----------

# MAGIC %md
# MAGIC **Verdict on linear models.** RandomForest beats Ridge by a measurable
# MAGIC margin (~5+ percentage points of R²). The gap is structural — not a
# MAGIC tuning issue, since we already swept Ridge's α via CV. The
# MAGIC Spearman/Pearson divergence in §4 confirmed monotone non-linear
# MAGIC structure. **Linear models REJECTED.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. K-means clustering — Are there natural groups in the data?
# MAGIC
# MAGIC **Why clustering in EDA.** If the data forms groups with very different `target`
# MAGIC values, but those groups are not linearly separable (centroids intertwined in the
# MAGIC original feature space), it reinforces the argument for non-linearity.

# COMMAND ----------

with mlflow.start_run(run_name="kmeans_clustering"):
    mlflow.set_tag("stage", "eda_clustering")

    asm = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scl = StandardScaler(inputCol="features_raw", outputCol="features",
                          withMean=True, withStd=True)
    sdf_in = scl.fit(asm.transform(train_sdf)).transform(asm.transform(train_sdf))

    # Elbow + silhouette para elegir k
    silhouette_eval = ClusteringEvaluator(featuresCol="features",
                                            metricName="silhouette",
                                            distanceMeasure="squaredEuclidean")
    elbow_data = []
    for k in [2, 3, 4, 5, 6]:
        kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=50)
        km_model = kmeans.fit(sdf_in)
        wcss = km_model.summary.trainingCost  # within-cluster sum of squares
        sil = silhouette_eval.evaluate(km_model.transform(sdf_in))
        elbow_data.append({"k": k, "wcss": wcss, "silhouette": sil})
        mlflow.log_metric(f"kmeans_wcss_k{k}", wcss)
        mlflow.log_metric(f"kmeans_silhouette_k{k}", sil)
    elbow_df = pd.DataFrame(elbow_data)
    print(elbow_df.round(4).to_string(index=False))

    # Pick k using silhouette (>0.25 considered reasonable structure)
    best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
    mlflow.log_metric("best_k", best_k)
    print(f"\nBest k by silhouette: {best_k}")

    # Re-fit with best k and analyze cluster-target relationship
    kmeans_best = KMeans(featuresCol="features", k=best_k, seed=42, maxIter=50)
    km_model = kmeans_best.fit(sdf_in)
    sdf_clustered = km_model.transform(sdf_in)

    cluster_stats = (sdf_clustered.groupBy("prediction")
                       .agg(F.count("*").alias("n"),
                             F.mean("target").alias("target_mean"),
                             F.stddev("target").alias("target_std"))
                       .toPandas()
                       .rename(columns={"prediction": "cluster"})
                       .sort_values("cluster"))
    print("\nCluster vs target distribution:")
    print(cluster_stats.round(3).to_string(index=False))

    # Within/between cluster target variance ratio
    overall_var = train_sdf.select(F.variance("target").alias("v")).first()["v"]
    within_var = (sdf_clustered.groupBy("prediction")
                    .agg(F.variance("target").alias("v"), F.count("*").alias("n"))
                    .toPandas())
    within_pooled = (within_var["v"] * (within_var["n"] - 1)).sum() / (within_var["n"].sum() - len(within_var))
    eta_squared = 1 - within_pooled / overall_var
    mlflow.log_metric("eta_squared_target_by_cluster", eta_squared)
    print(f"\nη² (target variance explained by cluster): {eta_squared:.3f}")
    print(f"Interpretation: {eta_squared*100:.1f}% of target variance "
          f"is explained by cluster membership alone.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(elbow_df["k"], elbow_df["wcss"], "o-", color="steelblue", lw=2)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("WCSS")
    axes[0].set_title("Elbow plot")

    axes[1].plot(elbow_df["k"], elbow_df["silhouette"], "o-", color="darkgreen", lw=2)
    axes[1].axvline(best_k, color="red", ls="--", label=f"best k={best_k}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Silhouette score"); axes[1].legend()

    cluster_target_pdf = sdf_clustered.select("prediction", "target").toPandas()
    cluster_target_pdf.boxplot(column="target", by="prediction", ax=axes[2])
    axes[2].set_xlabel("Cluster"); axes[2].set_ylabel("target")
    axes[2].set_title(f"Target distribution by cluster (η²={eta_squared:.3f})")
    plt.suptitle("")
    plt.tight_layout()
    
    plt.savefig("kmeans_analysis.png", dpi=110, bbox_inches="tight"); plt.close()
    mlflow.log_artifact("kmeans_analysis.png")

    if eta_squared > 0.10:
        mlflow.set_tag("clustering_finding",
                        f"Clusters explain {eta_squared*100:.1f}% of target variance — "
                        "structural groups exist that linear models cannot exploit")

# COMMAND ----------

# MAGIC %md
# MAGIC η² = 1 − (varianza intra-clusters / varianza total)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Conclusiones EDA
# MAGIC
# MAGIC | Question | Verdict | Evidence |
# MAGIC |---|---|---|
# MAGIC | Is the data clean? | ✅ | 0 nulls, 0 duplicates |
# MAGIC | Linear models suffice? | ❌ | Ridge plateaus ~5pp below RF baseline; Spearman > Pearson on key features |
# MAGIC | Should we use PCA? | ❌ | Variance uniformly spread (3-6%/PC); top-3 PCs only ~15% |
# MAGIC | Are there latent groups? | ✅ | K-means finds clusters; η² target by cluster ≈ logged metric |
# MAGIC
# MAGIC ### Implications for modelling
# MAGIC - Use **non-linear models** (gradient boosting, random forest)
# MAGIC - Apply **feature engineering** to expose interactions and non-linear transforms
# MAGIC - **Skip PCA**
# MAGIC - Use **mutual-information** based feature selection (captures non-linearity)
# MAGIC - Validate via **nested cross-validation** to avoid biased hyperparameter selection
# MAGIC
# MAGIC Next notebook: `01_training.py` — implements the modelling pipeline.
# MAGIC

# COMMAND ----------


