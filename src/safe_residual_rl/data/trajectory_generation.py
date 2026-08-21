"""Continuous trajectories generated from REAL_STATIC training support, never CSV row order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.neighbors import NearestNeighbors

from safe_residual_rl.kinematics import UR5Kinematics


@dataclass(frozen=True)
class GeneratedTrajectory:
    q_ref_rad: np.ndarray
    x_ref_m: np.ndarray
    generator: str
    evidence_level: str
    ood: dict


class RealSupportTrajectoryGenerator:
    def __init__(self, kinematics: UR5Kinematics) -> None:
        self.kinematics = kinematics
        self.q_min = None
        self.q_max = None
        self.q_low = None
        self.q_high = None
        self.mean = None
        self.scale = None
        self.neighbors = None
        self.training_q = None

    def fit(self, q_rad: np.ndarray, nominal_tcp_m: np.ndarray) -> "RealSupportTrajectoryGenerator":
        q = np.asarray(q_rad, dtype=float)
        x = np.asarray(nominal_tcp_m, dtype=float)
        self.training_q = q.copy()
        self.q_min, self.q_max = q.min(axis=0), q.max(axis=0)
        self.q_low, self.q_high = np.percentile(q, [5, 95], axis=0)
        features = np.column_stack((q, x))
        self.mean, self.scale = features.mean(axis=0), features.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        self.neighbors = NearestNeighbors(n_neighbors=1).fit((features - self.mean) / self.scale)
        return self

    def _ood(self, q: np.ndarray, x: np.ndarray) -> dict:
        features = np.column_stack((q, x))
        distance = self.neighbors.kneighbors((features - self.mean) / self.scale, return_distance=True)[0][:, 0]
        outside_q = np.any((q < self.q_min) | (q > self.q_max), axis=1)
        return {
            "nearest_standardized_q_tcp_distance": {
                "mean": float(distance.mean()), "median": float(np.median(distance)),
                "p95": float(np.percentile(distance, 95)), "max": float(distance.max()),
            },
            "outside_training_joint_bounds_fraction": float(outside_q.mean()),
        }

    def generate(self, kind: str, seed: int, length: int = 80) -> GeneratedTrajectory:
        rng = np.random.default_rng(seed)
        span = np.maximum(self.q_high - self.q_low, 1e-3)
        center = self.training_q[int(rng.integers(0, len(self.training_q)))].copy()
        progress = np.linspace(0.0, 1.0, length)
        evidence = "SIM_CALIBRATED"
        if kind == "sine":
            amplitude = rng.uniform(0.025, 0.075, size=6) * span
            phase = rng.uniform(0.0, 2.0 * np.pi, size=6)
            q = center + amplitude * np.sin(2.0 * np.pi * progress[:, None] + phase)
            q = np.clip(q, self.q_low, self.q_high)
        elif kind == "smooth_random":
            knot_t = np.linspace(0.0, 1.0, 7)
            knots = center + rng.normal(0.0, 0.045, size=(7, 6)) * span
            knots = np.clip(knots, self.q_low, self.q_high)
            q = make_interp_spline(knot_t, knots, k=3)(progress)
            q = np.clip(q, self.q_low, self.q_high)
        elif kind == "workspace_holdout":
            evidence = "SIM_STRESS"
            direction = np.where(rng.random(6) > 0.5, 1.0, -1.0)
            center = np.where(direction > 0, self.q_high, self.q_low) + direction * 0.08 * span
            amplitude = rng.uniform(0.015, 0.04, size=6) * span
            q = center + amplitude * np.sin(2.0 * np.pi * progress[:, None] + rng.uniform(0, 2 * np.pi, 6))
        else:
            raise ValueError(f"unknown trajectory kind: {kind}")
        x = self.kinematics.position_batch(q)
        return GeneratedTrajectory(q, x, kind, evidence, self._ood(q, x))
