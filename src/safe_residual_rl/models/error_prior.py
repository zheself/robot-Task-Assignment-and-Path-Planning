"""Transparent supervised error prior used as a baseline/simulator prior."""

from __future__ import annotations

import numpy as np


class RidgeErrorPrior:
    def __init__(self, regularization: float = 1e-3) -> None:
        self.regularization = regularization
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    @staticmethod
    def _raw_features(q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q_rad, dtype=float))
        x = np.atleast_2d(np.asarray(x_m, dtype=float))
        if q.shape[0] != x.shape[0] or q.shape[1] != 6 or x.shape[1] != 3:
            raise ValueError("expected q (N,6) and x (N,3)")
        return np.column_stack((np.sin(q), np.cos(q), x))

    def fit(self, q_rad: np.ndarray, x_m: np.ndarray, error_m: np.ndarray) -> "RidgeErrorPrior":
        raw = self._raw_features(q_rad, x_m)
        target = np.asarray(error_m, dtype=float)
        self.feature_mean = raw.mean(axis=0)
        self.feature_scale = raw.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-8] = 1.0
        standardized = (raw - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(raw)), standardized))
        penalty = np.eye(design.shape[1]) * self.regularization
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return self

    def predict(self, q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("fit must be called before predict")
        raw = self._raw_features(q_rad, x_m)
        design = np.column_stack((np.ones(len(raw)), (raw - self.feature_mean) / self.feature_scale))
        prediction = design @ self.weights
        return prediction[0] if np.asarray(q_rad).ndim == 1 else prediction


class MeanErrorPrior:
    """Strong simple baseline: training-set mean error, with no test fitting."""

    def __init__(self) -> None:
        self.mean_error_m: np.ndarray | None = None

    def fit(self, error_m: np.ndarray) -> "MeanErrorPrior":
        values = np.asarray(error_m, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("error_m must have shape (N,3)")
        self.mean_error_m = values.mean(axis=0)
        return self

    def predict(self, q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        if self.mean_error_m is None:
            raise RuntimeError("fit must be called before predict")
        count = 1 if np.asarray(q_rad).ndim == 1 else len(q_rad)
        result = np.repeat(self.mean_error_m[None, :], count, axis=0)
        return result[0] if np.asarray(q_rad).ndim == 1 else result
