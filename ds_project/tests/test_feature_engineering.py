"""
Tests unitarios para el módulo de feature engineering.
Ejecutar:  pytest tests/
"""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    engineer_features,
    get_engineered_columns,
    TOP_MI_FEATURES,
    TOP4_LINEAR,
)


@pytest.fixture
def sample_df():
    """20 features sintéticas con valores positivos (como el dataset real)."""
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1000, size=(50, 20))
    cols = [f"feature_{i}" for i in range(20)]
    return pd.DataFrame(data, columns=cols)


def test_output_has_62_columns(sample_df):
    out = engineer_features(sample_df)
    assert out.shape[1] == 62, f"Esperaba 62 cols, obtuve {out.shape[1]}"


def test_no_nan_no_inf(sample_df):
    out = engineer_features(sample_df)
    assert not out.isnull().any().any(), "Hay NaN tras FE"
    assert np.isfinite(out.values).all(), "Hay infs tras FE"


def test_columns_match_specification(sample_df):
    out = engineer_features(sample_df)
    expected = get_engineered_columns()
    assert list(out.columns) == expected


def test_log_transformation_correct(sample_df):
    out = engineer_features(sample_df)
    expected = np.log1p(np.abs(sample_df["feature_2"]))
    np.testing.assert_array_almost_equal(out["feature_2_log"].values, expected.values)


def test_interaction_correct(sample_df):
    out = engineer_features(sample_df)
    expected = sample_df["feature_2"] * sample_df["feature_13"]
    np.testing.assert_array_almost_equal(out["feature_2_x_feature_13"].values, expected.values)


def test_row_aggregates(sample_df):
    out = engineer_features(sample_df)
    orig_cols = [f"feature_{i}" for i in range(20)]
    np.testing.assert_array_almost_equal(
        out["row_mean"].values, sample_df[orig_cols].mean(axis=1).values
    )


def test_idempotent_on_same_input(sample_df):
    """Aplicar dos veces da el mismo resultado (función pura)."""
    a = engineer_features(sample_df)
    b = engineer_features(sample_df)
    pd.testing.assert_frame_equal(a, b)


def test_does_not_mutate_input(sample_df):
    original = sample_df.copy()
    _ = engineer_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)
