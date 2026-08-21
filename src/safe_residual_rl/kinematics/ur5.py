"""Minimal UR5 position kinematics in internal SI units."""

from __future__ import annotations

import numpy as np


UR5_DH = np.array(
    [
        [0.089159, 0.0, 0.0, np.pi / 2.0],
        [0.0, 0.0, -0.425, 0.0],
        [0.0, 0.0, -0.39225, 0.0],
        [0.10915, 0.0, 0.0, np.pi / 2.0],
        [0.09465, 0.0, 0.0, -np.pi / 2.0],
        [0.0823, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _dh_transform(d: float, theta: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class UR5Kinematics:
    """Standard-DH UR5 model copied from the available group resource.

    The convention still needs to be checked against the real controller and
    measurement coordinate system before real-data claims are made.
    """

    def __init__(self, dh_params: np.ndarray | None = None) -> None:
        self.dh = np.array(UR5_DH if dh_params is None else dh_params, dtype=float)
        if self.dh.shape != (6, 4):
            raise ValueError("dh_params must have shape (6, 4)")

    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,):
            raise ValueError("q_rad must have shape (6,)")
        transform = np.eye(4)
        for joint, (d, theta_offset, a, alpha) in zip(q, self.dh):
            transform = transform @ _dh_transform(d, joint + theta_offset, a, alpha)
        return transform

    def position(self, q_rad: np.ndarray) -> np.ndarray:
        return self.forward(q_rad)[:3, 3]

    def position_batch(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.ndim != 2 or q.shape[1] != 6:
            raise ValueError("q_rad must have shape (N, 6)")
        return np.stack([self.position(row) for row in q])

    def position_jacobian(self, q_rad: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        jacobian = np.empty((3, 6), dtype=float)
        for index in range(6):
            step = np.zeros(6)
            step[index] = eps
            jacobian[:, index] = (self.position(q + step) - self.position(q - step)) / (2.0 * eps)
        return jacobian


def path_frames(points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return right-handed local [tangent, normal, binormal] frames and curvature."""
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points_m must have shape (T, 3), T >= 3")
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    dt = np.gradient(tangent, axis=0)
    curvature = np.linalg.norm(dt, axis=1)
    frames = np.empty((len(points), 3, 3), dtype=float)
    previous_normal = np.array([0.0, 0.0, 1.0])
    for index, direction in enumerate(tangent):
        normal_raw = dt[index] - np.dot(dt[index], direction) * direction
        if np.linalg.norm(normal_raw) < 1e-8:
            normal_raw = previous_normal - np.dot(previous_normal, direction) * direction
        if np.linalg.norm(normal_raw) < 1e-8:
            fallback = np.array([1.0, 0.0, 0.0])
            normal_raw = fallback - np.dot(fallback, direction) * direction
        normal = normal_raw / np.linalg.norm(normal_raw)
        binormal = np.cross(direction, normal)
        binormal /= np.linalg.norm(binormal)
        normal = np.cross(binormal, direction)
        frames[index] = np.column_stack((direction, normal, binormal))
        previous_normal = normal
    return frames, curvature
