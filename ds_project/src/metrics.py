"""
Advanced regression metrics + diagnostics.

Includes:
    - Standard set      : R², RMSE, MAE, MedAE, MaxAbsErr, ExplVar
    - Correlation       : Pearson, Spearman, Concordance Correlation Coef.
    - Scale-free        : MAPE, SMAPE
    - Bias              : MBE (Mean Bias Error), Theil's U2
    - Goodness-of-fit   : Nash-Sutcliffe Efficiency, Willmott Index of Agreement
    - Quantile losses   : pinball loss at P10, P50, P90
    - Calibration       : prediction interval coverage (PICP), MPIW
    - Uncertainty       : bootstrap CI, residual diagnostics (Shapiro, BP test)
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr, shapiro
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error, max_error, explained_variance_score,
)


THEORETICAL_R2_CEILING = 0.92  # given by problem (target noise floor)


# ============================================================
# Core metric set
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Comprehensive regression metrics (sane defaults for any problem)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    res = y_true - y_pred

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE / SMAPE — guard against y=0
    eps = 1e-8
    mape  = float(np.mean(np.abs(res / (np.abs(y_true) + eps))) * 100)
    smape = float(np.mean(2 * np.abs(res) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)

    # Mean Bias Error: tells us if we systematically over/under-predict
    mbe = float(res.mean())

    # Theil's U2 (1=as good as naive, <1 better than naive)
    naive_diffs_sq = np.mean(np.diff(y_true) ** 2) + eps
    theils_u2 = float(np.sqrt(np.mean(res ** 2) / naive_diffs_sq))

    # Concordance Correlation Coefficient (Lin 1989)
    yt_m, yp_m = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - yt_m) * (y_pred - yp_m))
    ccc = float(2 * cov / (y_true.var() + y_pred.var() + (yt_m - yp_m) ** 2 + eps))

    # Nash-Sutcliffe Efficiency (hydrology / ecology standard)
    nse = float(1 - np.sum(res ** 2) / (np.sum((y_true - yt_m) ** 2) + eps))

    # Willmott's Index of Agreement d
    denom = np.sum((np.abs(y_pred - yt_m) + np.abs(y_true - yt_m)) ** 2) + eps
    willmott_d = float(1 - np.sum(res ** 2) / denom)

    return {
        "r2":            r2,
        "rmse":          rmse,
        "mae":           mae,
        "medae":         float(median_absolute_error(y_true, y_pred)),
        "max_abs_err":   float(max_error(y_true, y_pred)),
        "explained_var": float(explained_variance_score(y_true, y_pred)),
        "pearson_r":     float(pearsonr(y_true, y_pred)[0]),
        "spearman_rho":  float(spearmanr(y_true, y_pred)[0]),
        "ccc":           ccc,           # Concordance Correlation Coefficient
        "mape_pct":      mape,
        "smape_pct":     smape,
        "mbe":           mbe,            # Mean Bias Error
        "theils_u2":     theils_u2,
        "nse":           nse,            # Nash-Sutcliffe Efficiency
        "willmott_d":    willmott_d,
        "efficiency":    r2 / THEORETICAL_R2_CEILING,
    }


# ============================================================
# Bootstrap CIs — for any metric
# ============================================================
def bootstrap_r2_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
    n_resamples: int = 2000, alpha: float = 0.05, random_state: int = 42,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Percentile bootstrap CI for R². Returns (samples, (low, high))."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        samples[i] = r2_score(y_true[idx], y_pred[idx])
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return samples, (float(low), float(high))


def bootstrap_metric_ci(
    y_true: np.ndarray, y_pred: np.ndarray, metric_fn,
    n_resamples: int = 1000, alpha: float = 0.05, random_state: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap CI for any metric_fn(y_true, y_pred) -> float."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(samples.mean()), (float(low), float(high))


# ============================================================
# Quantile (pinball) loss — for prediction interval evaluation
# ============================================================
def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Asymmetric absolute loss for quantile q ∈ (0, 1)."""
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ============================================================
# Prediction interval calibration
# ============================================================
def interval_metrics(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray,
    nominal_coverage: float = 0.90,
) -> Dict[str, float]:
    """
    Evaluate quality of prediction intervals.

    Returns:
        - picp        : Prediction Interval Coverage Probability (empirical)
        - mpiw        : Mean Prediction Interval Width
        - calibration : |empirical_coverage - nominal_coverage|
        - normalized_mpiw: MPIW / range(y) — width as fraction of target range
    """
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    picp = float(inside.mean())
    mpiw = float(np.mean(y_upper - y_lower))
    return {
        "picp":             picp,
        "mpiw":             mpiw,
        "calibration_err":  float(abs(picp - nominal_coverage)),
        "normalized_mpiw":  mpiw / (y_true.max() - y_true.min() + 1e-8),
        "nominal_coverage": nominal_coverage,
    }


# ============================================================
# Residual diagnostics
# ============================================================
def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tests on residuals:
        - Normality (Shapiro-Wilk)
        - Heteroscedasticity (Breusch-Pagan-style: |res| ~ ŷ correlation)
        - Bias (mean of residuals should be ~0)
        - Autocorrelation (Durbin-Watson approx)
    """
    res = y_true - y_pred
    sh_stat, sh_p = shapiro(res) if len(res) <= 5000 else (np.nan, np.nan)
    het_r, het_p = pearsonr(y_pred, np.abs(res))
    # Durbin-Watson (informal autocorrelation indicator)
    dw = float(np.sum(np.diff(res) ** 2) / np.sum(res ** 2))

    return {
        "residual_mean":         float(res.mean()),
        "residual_std":          float(res.std()),
        "residual_skew":         float(((res - res.mean()) ** 3).mean() / (res.std() ** 3 + 1e-12)),
        "residual_kurtosis":     float(((res - res.mean()) ** 4).mean() / (res.std() ** 4 + 1e-12) - 3),
        "shapiro_p":             float(sh_p),
        "is_normal_residuals":   bool(sh_p > 0.05),
        "heteroscedasticity_r":  float(het_r),
        "heteroscedasticity_p":  float(het_p),
        "is_homoscedastic":      bool(het_p > 0.05),
        "durbin_watson":         dw,
    }


# ============================================================
# Stratified residual analysis (errors by prediction quantile)
# ============================================================
def residuals_by_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10,
) -> Dict[str, list]:
    """
    Stratify residuals by prediction quantile to see where the model
    over/underperforms. Useful to detect systematic bias in extreme regions.
    """
    res = y_true - y_pred
    quantile_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bins, mae_per_bin, bias_per_bin = [], [], []
    for i in range(n_bins):
        mask = (y_pred >= quantile_edges[i]) & (y_pred <= quantile_edges[i + 1])
        if mask.sum() == 0: continue
        bins.append((float(quantile_edges[i]), float(quantile_edges[i + 1])))
        mae_per_bin.append(float(np.abs(res[mask]).mean()))
        bias_per_bin.append(float(res[mask].mean()))
    return {"bins": bins, "mae": mae_per_bin, "bias": bias_per_bin}
