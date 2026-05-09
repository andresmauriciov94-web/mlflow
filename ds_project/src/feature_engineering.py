"""
Feature engineering pipeline.

Diseñado para ser ejecutable tanto en pandas (notebook exploratorio)
como en pyspark.pandas (cluster Databricks) sin cambios.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Constantes derivadas del EDA (training_data.csv, n=800)
# Ver notebooks/01_eda.ipynb para la justificación cuantitativa.
# ----------------------------------------------------------------------
TOP_MI_FEATURES: List[str] = [
    "feature_2", "feature_13", "feature_16", "feature_9",
    "feature_3", "feature_18", "feature_11", "feature_5",
]

TOP4_LINEAR: List[str] = [
    "feature_2", "feature_13", "feature_9", "feature_11",
]

WEIGHTED_TOP_WEIGHTS = {
    "feature_2": 0.50,
    "feature_13": 0.30,
    "feature_16": 0.15,
    "feature_9": 0.05,
}


def engineer_features(df: pd.DataFrame, original_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Genera 42 features derivadas a partir de las 20 originales.

    Estrategia:
        - Transformaciones no-lineales (log/sqrt/sq) sobre top-8 por MI
        - Productos y ratios entre top-4 (feature_2 × feature_13 fue la
          interacción con MI=0.30, mayor que cualquier feature original)
        - Estadísticos por fila (mean/std/max/min/range)
        - Combinación lineal ponderada de top-4 (mini-target hand-crafted)

    Args:
        df: DataFrame con (al menos) las 20 features originales.
        original_cols: Lista de nombres de las features originales.
            Si None, se asume `feature_0` … `feature_19`.

    Returns:
        DataFrame con 62 columnas (20 originales + 42 derivadas).
    """
    if original_cols is None:
        original_cols = [f"feature_{i}" for i in range(20)]

    out = df.copy()

    # --- 1. Transformaciones no-lineales sobre top-MI ---
    for f in TOP_MI_FEATURES:
        out[f"{f}_log"]  = np.log1p(np.abs(df[f]))
        out[f"{f}_sqrt"] = np.sqrt(np.abs(df[f]))
        out[f"{f}_sq"]   = df[f] ** 2

    # --- 2. Interacciones top-4 (productos y ratios) ---
    for i, a in enumerate(TOP4_LINEAR):
        for b in TOP4_LINEAR[i + 1:]:
            out[f"{a}_x_{b}"]   = df[a] * df[b]
            out[f"{a}_div_{b}"] = df[a] / (np.abs(df[b]) + 1e-6)

    # --- 3. Estadísticos por fila (sobre originales) ---
    base = df[original_cols]
    out["row_mean"]  = base.mean(axis=1)
    out["row_std"]   = base.std(axis=1)
    out["row_max"]   = base.max(axis=1)
    out["row_min"]   = base.min(axis=1)
    out["row_range"] = out["row_max"] - out["row_min"]

    # --- 4. Suma ponderada del top-4 ---
    out["weighted_top"] = sum(w * df[f] for f, w in WEIGHTED_TOP_WEIGHTS.items())

    return out


def get_engineered_columns() -> List[str]:
    """Devuelve los nombres de las 62 columnas tras engineer_features."""
    cols = [f"feature_{i}" for i in range(20)]
    for f in TOP_MI_FEATURES:
        cols += [f"{f}_log", f"{f}_sqrt", f"{f}_sq"]
    for i, a in enumerate(TOP4_LINEAR):
        for b in TOP4_LINEAR[i + 1:]:
            cols += [f"{a}_x_{b}", f"{a}_div_{b}"]
    cols += ["row_mean", "row_std", "row_max", "row_min", "row_range", "weighted_top"]
    return cols
