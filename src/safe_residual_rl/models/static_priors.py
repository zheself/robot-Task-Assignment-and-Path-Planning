"""Supervised REAL_STATIC priors; all share q_rad + nominal_TCP_m -> error_m."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from .error_prior import MeanErrorPrior, RidgeErrorPrior


def prior_features(q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(np.asarray(q_rad, dtype=float))
    x = np.atleast_2d(np.asarray(x_m, dtype=float))
    if q.shape[0] != x.shape[0] or q.shape[1] != 6 or x.shape[1] != 3:
        raise ValueError("expected q_rad (N,6) and nominal_TCP_m (N,3)")
    return np.column_stack((q, x))


class ZeroErrorPrior:
    def fit(self, q_rad: np.ndarray, x_m: np.ndarray, error_m: np.ndarray) -> "ZeroErrorPrior":
        prior_features(q_rad, x_m)
        return self

    def predict(self, q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        features = prior_features(q_rad, x_m)
        prediction = np.zeros((len(features), 3))
        return prediction[0] if np.asarray(q_rad).ndim == 1 else prediction


class MeanBiasPrior(MeanErrorPrior):
    def fit(self, q_rad: np.ndarray, x_m: np.ndarray, error_m: np.ndarray) -> "MeanBiasPrior":
        prior_features(q_rad, x_m)
        super().fit(error_m)
        return self


class SklearnStaticPrior:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, q_rad: np.ndarray, x_m: np.ndarray, error_m: np.ndarray) -> "SklearnStaticPrior":
        self.estimator.fit(prior_features(q_rad, x_m), np.asarray(error_m, dtype=float))
        return self

    def predict(self, q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        prediction = np.asarray(self.estimator.predict(prior_features(q_rad, x_m)), dtype=float)
        return prediction[0] if np.asarray(q_rad).ndim == 1 else prediction


def make_static_priors(seed: int = 2026) -> dict:
    """Fixed pre-advisor model suite; no test-domain hyperparameter fitting."""
    return {
        "zero_error": ZeroErrorPrior(),
        "mean_bias": MeanBiasPrior(),
        "ridge": RidgeErrorPrior(regularization=1e-3),
        "extra_trees": SklearnStaticPrior(
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=-1)
        ),
        "random_forest": SklearnStaticPrior(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=-1)
        ),
        "rbf_kernel": SklearnStaticPrior(
            Pipeline([("scale", StandardScaler()), ("model", KernelRidge(alpha=2e-4, kernel="rbf", gamma=0.12))])
        ),
        "mlp": SklearnStaticPrior(
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=(64, 32), activation="tanh", alpha=1e-3,
                                learning_rate_init=5e-4, max_iter=1800, random_state=seed,
                            ),
                        ),
                    ]
                ),
                transformer=StandardScaler(),
            )
        ),
    }


def fit_prior(prior, q_rad: np.ndarray, x_m: np.ndarray, error_m: np.ndarray):
    """Handle the legacy transparent ridge while keeping a common public interface."""
    return prior.fit(q_rad, x_m, error_m)
