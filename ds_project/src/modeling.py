"""
Model factory: top-3 boosting champions + Neural Network.

Selección final tras benchmarking exhaustivo (14 modelos → top-4 → top-3):
    1. CatBoost          - holdout R² 0.8833
    2. GradientBoosting  - holdout R² 0.8749
    3. LightGBM          - holdout R² 0.8701
    + KerasMLPRegressor  - regularized MLP (4to candidato)

Hiperparámetros fijados tras RandomizedSearchCV + grid manual.
Ver notebooks/02_databricks_native.py para la experimentación completa.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ============================================================
# Boosting champions (top-3 from benchmark)
# ============================================================
def build_catboost(random_state: int = 42) -> CatBoostRegressor:
    """CatBoost champion: depth=5, lr=0.05, l2=1.0  (CV R² = 0.879)."""
    return CatBoostRegressor(
        iterations=1000,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=1.0,
        random_state=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def build_gradient_boosting(random_state: int = 42) -> GradientBoostingRegressor:
    """sklearn GBM: 400 trees, depth=4, subsample=0.85  (CV R² = 0.867)."""
    return GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        min_samples_leaf=5,
        random_state=random_state,
    )


def build_lightgbm(random_state: int = 42) -> LGBMRegressor:
    """LightGBM tuneado por RandomizedSearchCV (30 iters, CV R² = 0.866)."""
    return LGBMRegressor(
        n_estimators=1196,
        num_leaves=62,
        max_depth=3,
        learning_rate=0.0656,
        min_child_samples=6,
        subsample=0.965,
        colsample_bytree=0.898,
        reg_alpha=3.724,
        reg_lambda=0.116,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


# ============================================================
# Neural Network — sklearn-compatible wrapper around Keras
# ============================================================
class KerasMLPRegressor(BaseEstimator, RegressorMixin):
    """
    Regularized MLP for tabular regression on small data (n=800).

    Architecture rationale:
        - Small width (64-32-16): with 800 rows a wider network overfits
        - Heavy regularization: dropout(0.3) + L2(1e-4) + BatchNorm
        - Adam + early stopping on val_loss + ReduceLROnPlateau
        - Standardized targets (NN training is more stable on scaled y)
        - Internal val split: 15% of training data, no leakage to outer holdout

    Compatible with sklearn (fit/predict/get_params/set_params) so it slots
    cleanly into cross_val_score, MLflow autolog, and our pipeline.
    """
    def __init__(
        self,
        hidden_units: tuple = (64, 32, 16),
        dropout: float = 0.3,
        l2_reg: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 20,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _build_keras_model(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, regularizers, Sequential
        tf.keras.utils.set_random_seed(self.random_state)
        model = Sequential([layers.Input(shape=(n_features,))])
        for units in self.hidden_units:
            model.add(layers.Dense(
                units, activation="relu",
                kernel_regularizer=regularizers.l2(self.l2_reg),
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse", metrics=["mae"],
        )
        return model

    def fit(self, X, y):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_ = float(y.mean())
        self.y_std_  = float(y.std()) or 1.0
        y_scaled = (y - self.y_mean_) / self.y_std_

        self.model_ = self._build_keras_model(X.shape[1])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.patience,
                          restore_best_weights=True, verbose=self.verbose),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=self.patience // 2, min_lr=1e-6,
                              verbose=self.verbose),
        ]
        self.history_ = self.model_.fit(
            X, y_scaled,
            validation_split=0.15, epochs=self.epochs,
            batch_size=self.batch_size, callbacks=callbacks,
            verbose=self.verbose, shuffle=True,
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        y_scaled = self.model_.predict(X, verbose=0).flatten()
        return y_scaled * self.y_std_ + self.y_mean_

    def get_params(self, deep=True):
        return {"hidden_units": self.hidden_units, "dropout": self.dropout,
                "l2_reg": self.l2_reg, "learning_rate": self.learning_rate,
                "batch_size": self.batch_size, "epochs": self.epochs,
                "patience": self.patience, "random_state": self.random_state,
                "verbose": self.verbose}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_neural_net(random_state: int = 42) -> KerasMLPRegressor:
    """Champion NN config (validated by CV)."""
    return KerasMLPRegressor(
        hidden_units=(64, 32, 16), dropout=0.3, l2_reg=1e-4,
        learning_rate=1e-3, batch_size=32, epochs=200, patience=20,
        random_state=random_state, verbose=0,
    )


def get_all_champions(random_state: int = 42) -> Dict[str, Any]:
    """Top-3 boosting + Neural Net = the 4 final candidates."""
    return {
        "CatBoost":         build_catboost(random_state),
        "GradientBoosting": build_gradient_boosting(random_state),
        "LightGBM":         build_lightgbm(random_state),
        "NeuralNet":        build_neural_net(random_state),
    }
