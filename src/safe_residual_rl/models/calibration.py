"""Training-only calibration summaries for SIM_CALIBRATED environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationProfile:
    prior_name: str
    unexplained_residual_m: np.ndarray
    session_shift_proxy_m: np.ndarray
    session_ids: tuple[str, ...]
    summary: dict

    def sample_episode_error(self, prior, seed: int):
        rng = np.random.default_rng(seed)
        shift = self.session_shift_proxy_m[int(rng.integers(0, len(self.session_shift_proxy_m)))].copy()
        residuals = self.unexplained_residual_m

        class EpisodeError:
            def __call__(self, q_rad, x_m):
                prediction = np.asarray(prior.predict(q_rad, x_m), dtype=float)
                unexplained = residuals[int(rng.integers(0, len(residuals)))]
                return prediction + shift + unexplained

        return EpisodeError()


def fit_calibration_profile(prior_name: str, prior, dataset, train_mask: np.ndarray) -> CalibrationProfile:
    q = dataset.q_rad[train_mask]
    x = dataset.x_nominal_m[train_mask]
    target = dataset.error_m[train_mask]
    session = dataset.session_id[train_mask]
    residual = target - prior.predict(q, x)
    session_ids = tuple(sorted(set(session)))
    session_means = np.stack([residual[session == value].mean(axis=0) for value in session_ids])
    global_mean = residual.mean(axis=0)
    shift_proxy = session_means - global_mean
    centered = residual.copy()
    for value, mean in zip(session_ids, session_means):
        centered[session == value] -= mean
    norm_mm = np.linalg.norm(centered, axis=1) * 1000.0
    shift_norm_mm = np.linalg.norm(shift_proxy, axis=1) * 1000.0
    summary = {
        "terminology": {
            "centered_model_residual": "unexplained_residual_not_measurement_noise",
            "file_mean_difference": "session_shift_proxy_not_confirmed_physical_drift",
        },
        "training_rows": len(q),
        "training_sessions": len(session_ids),
        "unexplained_residual_norm_mm": {
            "mean": float(norm_mm.mean()), "median": float(np.median(norm_mm)),
            "p95": float(np.percentile(norm_mm, 95)), "max": float(norm_mm.max()),
        },
        "unexplained_residual_component_std_mm": (centered.std(axis=0) * 1000.0).tolist(),
        "session_shift_proxy_norm_mm": {
            "mean": float(shift_norm_mm.mean()), "p95": float(np.percentile(shift_norm_mm, 95)),
            "max": float(shift_norm_mm.max()),
        },
        "session_shift_proxy_components_mm": {
            value: (shift * 1000.0).tolist() for value, shift in zip(session_ids, shift_proxy)
        },
    }
    return CalibrationProfile(prior_name, centered, shift_proxy, session_ids, summary)
