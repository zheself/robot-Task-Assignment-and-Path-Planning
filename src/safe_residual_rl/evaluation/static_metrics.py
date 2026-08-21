from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from safe_residual_rl.models.static_priors import prior_features


def error_metrics(target_m: np.ndarray, prediction_m: np.ndarray) -> dict:
    residual_mm = (np.asarray(prediction_m) - np.asarray(target_m)) * 1000.0
    norm_mm = np.linalg.norm(residual_mm, axis=1)
    return {
        "count": len(norm_mm),
        "rmse_mm": float(np.sqrt(np.mean(np.sum(residual_mm**2, axis=1)))),
        "mae_mm": float(np.mean(norm_mm)),
        "p95_mm": float(np.percentile(norm_mm, 95)),
        "max_mm": float(np.max(norm_mm)),
        "xyz_rmse_mm": np.sqrt(np.mean(residual_mm**2, axis=0)).tolist(),
        "xyz_mae_mm": np.mean(np.abs(residual_mm), axis=0).tolist(),
        "xyz_bias_mm": np.mean(residual_mm, axis=0).tolist(),
    }


class TrainingSupport:
    def __init__(self) -> None:
        self.mean = None
        self.scale = None
        self.minimum = None
        self.maximum = None
        self.distance_threshold = None
        self.neighbors = None

    def fit(self, q_rad: np.ndarray, x_m: np.ndarray) -> "TrainingSupport":
        features = prior_features(q_rad, x_m)
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        standardized = (features - self.mean) / self.scale
        self.minimum = features.min(axis=0)
        self.maximum = features.max(axis=0)
        self.neighbors = NearestNeighbors(n_neighbors=2).fit(standardized)
        train_distances = self.neighbors.kneighbors(standardized, return_distance=True)[0][:, 1]
        self.distance_threshold = float(np.percentile(train_distances, 95))
        return self

    def describe(self, q_rad: np.ndarray, x_m: np.ndarray) -> dict:
        features = prior_features(q_rad, x_m)
        standardized = (features - self.mean) / self.scale
        distance = self.neighbors.kneighbors(standardized, n_neighbors=1, return_distance=True)[0][:, 0]
        inside_bounds = np.all((features >= self.minimum) & (features <= self.maximum), axis=1)
        close = distance <= self.distance_threshold
        labels = np.where(~inside_bounds, "outside_axis_bounds", np.where(close, "near_training_support", "sparse_in_bounds"))
        return {
            "distance": distance,
            "labels": labels,
            "distance_summary": {
                "mean": float(distance.mean()), "median": float(np.median(distance)),
                "p95": float(np.percentile(distance, 95)), "max": float(distance.max()),
                "train_p95_threshold": self.distance_threshold,
            },
            "outside_axis_fraction": float(np.mean(~inside_bounds)),
            "far_fraction": float(np.mean(~close)),
        }


def grouped_metrics(target, prediction, groups: np.ndarray) -> dict:
    return {
        str(group): error_metrics(target[groups == group], prediction[groups == group])
        for group in sorted(set(groups))
    }


def evaluate_static_prior(prior, dataset, mask: np.ndarray, support: TrainingSupport) -> dict:
    q = dataset.q_rad[mask]
    x = dataset.x_nominal_m[mask]
    target = dataset.error_m[mask]
    prediction = prior.predict(q, x)
    support_result = support.describe(q, x)
    by_support = {}
    for label in sorted(set(support_result["labels"])):
        local = support_result["labels"] == label
        by_support[label] = error_metrics(target[local], prediction[local])
    return {
        "overall": error_metrics(target, prediction),
        "by_session": grouped_metrics(target, prediction, dataset.session_id[mask]),
        "by_date": grouped_metrics(target, prediction, dataset.date_id[mask]),
        "by_workspace_support": by_support,
        "support": {key: value for key, value in support_result.items() if key not in ("distance", "labels")},
    }
