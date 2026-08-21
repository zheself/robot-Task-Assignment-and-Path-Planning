from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safe_residual_rl.kinematics import UR5Kinematics


@dataclass(frozen=True)
class ProjectionResult:
    delta_q_rad: np.ndarray
    requested_world_m: np.ndarray
    executed_world_m: np.ndarray
    clipped: bool
    reasons: tuple[str, ...]
    condition_number: float


class SafetyProjector:
    def __init__(
        self,
        kinematics: UR5Kinematics,
        max_cartesian_step_m: float = 0.006,
        max_joint_step_rad: float = 0.04,
        damping: float = 0.02,
        max_condition_number: float = 250.0,
        joint_limits_rad: np.ndarray | None = None,
    ) -> None:
        self.kinematics = kinematics
        self.max_cartesian_step_m = max_cartesian_step_m
        self.max_joint_step_rad = max_joint_step_rad
        self.damping = damping
        self.max_condition_number = max_condition_number
        self.joint_limits = np.asarray(
            joint_limits_rad
            if joint_limits_rad is not None
            else np.column_stack((-2.0 * np.pi * np.ones(6), 2.0 * np.pi * np.ones(6))),
            dtype=float,
        )

    def project_world(self, q_rad: np.ndarray, requested_world_m: np.ndarray) -> ProjectionResult:
        q = np.asarray(q_rad, dtype=float)
        requested = np.asarray(requested_world_m, dtype=float)
        if requested.shape != (3,):
            raise ValueError("requested_world_m must have shape (3,)")
        reasons: list[str] = []
        bounded = requested.copy()
        norm = np.linalg.norm(bounded)
        if norm > self.max_cartesian_step_m:
            bounded *= self.max_cartesian_step_m / norm
            reasons.append("cartesian_step")

        jacobian = self.kinematics.position_jacobian(q)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
        damping = self.damping
        if condition > self.max_condition_number:
            damping *= min(condition / self.max_condition_number, 20.0)
            reasons.append("near_singularity")
        inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping**2 * np.eye(3))
        delta_q = inverse @ bounded
        largest_joint_step = np.max(np.abs(delta_q))
        if largest_joint_step > self.max_joint_step_rad:
            delta_q *= self.max_joint_step_rad / largest_joint_step
            reasons.append("joint_step")

        proposed_q = q + delta_q
        limited_q = np.clip(proposed_q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        if not np.allclose(proposed_q, limited_q):
            delta_q = limited_q - q
            reasons.append("joint_limit")
        executed = jacobian @ delta_q
        return ProjectionResult(
            delta_q_rad=delta_q,
            requested_world_m=requested,
            executed_world_m=executed,
            clipped=bool(reasons),
            reasons=tuple(reasons),
            condition_number=condition,
        )
