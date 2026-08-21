"""Synthetic data for pipeline validation, never for final scientific claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safe_residual_rl.kinematics import UR5Kinematics


@dataclass(frozen=True)
class MeasurementDataset:
    q_rad: np.ndarray
    x_nominal_m: np.ndarray
    x_measured_m: np.ndarray
    session_id: np.ndarray
    path_id: np.ndarray
    date_id: np.ndarray
    evidence_level: str = "synthetic_for_pipeline_validation_only"

    @property
    def error_m(self) -> np.ndarray:
        return self.x_measured_m - self.x_nominal_m

    def validate(self) -> None:
        count = len(self.q_rad)
        if self.q_rad.shape != (count, 6):
            raise ValueError("q_rad must have shape (N, 6)")
        if self.x_nominal_m.shape != (count, 3) or self.x_measured_m.shape != (count, 3):
            raise ValueError("TCP arrays must have shape (N, 3)")
        if not all(len(values) == count for values in (self.session_id, self.path_id, self.date_id)):
            raise ValueError("metadata columns must have N entries")
        if not np.isfinite(self.q_rad).all() or not np.isfinite(self.x_measured_m).all():
            raise ValueError("dataset contains non-finite values")


class SyntheticErrorField:
    """Smooth, millimetre-scale position error field with hidden parameters."""

    def __init__(self, seed: int = 17, amplitude_m: float = 0.0025) -> None:
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(16, 3))
        weights /= np.maximum(np.linalg.norm(weights, axis=0, keepdims=True), 1e-12)
        self.weights = amplitude_m * weights

    @staticmethod
    def features(q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q_rad, dtype=float))
        x = np.atleast_2d(np.asarray(x_m, dtype=float))
        if q.shape[0] != x.shape[0] or q.shape[1] != 6 or x.shape[1] != 3:
            raise ValueError("expected q (N,6) and x (N,3)")
        return np.column_stack((np.ones(len(q)), np.sin(q), np.cos(q), x))

    def __call__(self, q_rad: np.ndarray, x_m: np.ndarray) -> np.ndarray:
        result = self.features(q_rad, x_m) @ self.weights
        scalar_input = np.asarray(q_rad).ndim == 1
        return result[0] if scalar_input else result


def generate_reference_path(
    kinematics: UR5Kinematics,
    seed: int,
    length: int = 80,
    workspace_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    center = np.array([0.1, -1.20, 1.35, -1.15, -1.35, 0.15])
    center += workspace_shift * np.array([0.6, -0.3, 0.25, 0.2, -0.2, 0.3])
    phase = rng.uniform(0.0, 2.0 * np.pi, size=6)
    amplitude = np.array([0.22, 0.16, 0.18, 0.12, 0.10, 0.16]) * rng.uniform(0.75, 1.1, size=6)
    progress = np.linspace(0.0, 2.0 * np.pi, length)
    q_ref = center + amplitude * np.sin(progress[:, None] + phase)
    q_ref += 0.035 * np.sin(2.0 * progress[:, None] + phase / 2.0)
    return q_ref, kinematics.position_batch(q_ref)


def generate_measurement_dataset(
    kinematics: UR5Kinematics,
    error_field: SyntheticErrorField,
    seed: int,
    session_prefix: str,
    date_id: str,
    path_count: int = 5,
    path_length: int = 80,
    drift_m: np.ndarray | None = None,
    noise_std_m: float = 0.00015,
    workspace_shift: float = 0.0,
) -> MeasurementDataset:
    rng = np.random.default_rng(seed)
    drift = np.zeros(3) if drift_m is None else np.asarray(drift_m, dtype=float)
    q_parts, x_parts, measured_parts = [], [], []
    session_parts, path_parts, date_parts = [], [], []
    for path_index in range(path_count):
        q_ref, x_ref = generate_reference_path(
            kinematics,
            seed=seed * 100 + path_index,
            length=path_length,
            workspace_shift=workspace_shift,
        )
        session_drift = drift + rng.normal(0.0, 0.00012, size=3)
        noise = rng.normal(0.0, noise_std_m, size=x_ref.shape)
        measured = x_ref + error_field(q_ref, x_ref) + session_drift + noise
        q_parts.append(q_ref)
        x_parts.append(x_ref)
        measured_parts.append(measured)
        session_parts.extend([f"{session_prefix}_s{path_index:02d}"] * path_length)
        path_parts.extend([f"{session_prefix}_p{path_index:02d}"] * path_length)
        date_parts.extend([date_id] * path_length)
    dataset = MeasurementDataset(
        q_rad=np.vstack(q_parts),
        x_nominal_m=np.vstack(x_parts),
        x_measured_m=np.vstack(measured_parts),
        session_id=np.asarray(session_parts),
        path_id=np.asarray(path_parts),
        date_id=np.asarray(date_parts),
    )
    dataset.validate()
    return dataset
