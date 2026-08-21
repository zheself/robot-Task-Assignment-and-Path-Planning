"""Dependency-light core environment for sequential trajectory compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from safe_residual_rl.kinematics import UR5Kinematics, path_frames
from safe_residual_rl.safety import SafetyProjector


ErrorFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
PriorFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Trajectory:
    q_ref_rad: np.ndarray
    x_ref_m: np.ndarray
    path_id: str

    def validate(self) -> None:
        if self.q_ref_rad.ndim != 2 or self.q_ref_rad.shape[1] != 6:
            raise ValueError("q_ref_rad must have shape (T,6)")
        if self.x_ref_m.shape != (len(self.q_ref_rad), 3) or len(self.q_ref_rad) < 3:
            raise ValueError("x_ref_m must have shape (T,3), T >= 3")


class ResidualTrajectoryEnv:
    """One episode is one ordered reference path.

    The action at time t is a local TNB residual for reference point t+1. The
    observation includes the measured error at t, avoiding oracle access to the
    simulator's hidden error field or episode drift.
    """

    def __init__(
        self,
        kinematics: UR5Kinematics,
        trajectory: Trajectory,
        true_error: ErrorFunction,
        projector: SafetyProjector,
        base_prior: PriorFunction | None = None,
        plant_kinematics: UR5Kinematics | None = None,
        joint_zero_offset_rad: np.ndarray | None = None,
        tcp_offset_m: np.ndarray | None = None,
        episode_drift_m: np.ndarray | None = None,
        noise_std_m: float = 0.00015,
        action_delay_steps: int = 0,
        residual_action_bound_m: float = 0.002,
        history_length: int = 4,
        seed: int = 0,
    ) -> None:
        trajectory.validate()
        self.kinematics = kinematics
        self.trajectory = trajectory
        self.true_error = true_error
        self.projector = projector
        self.base_prior = base_prior
        self.plant_kinematics = kinematics if plant_kinematics is None else plant_kinematics
        self.joint_zero_offset_rad = (
            np.zeros(6) if joint_zero_offset_rad is None else np.asarray(joint_zero_offset_rad, dtype=float)
        )
        self.tcp_offset_m = np.zeros(3) if tcp_offset_m is None else np.asarray(tcp_offset_m, dtype=float)
        if self.joint_zero_offset_rad.shape != (6,) or self.tcp_offset_m.shape != (3,):
            raise ValueError("joint_zero_offset_rad and tcp_offset_m must have shape (6,) and (3,)")
        self.episode_drift_m = np.zeros(3) if episode_drift_m is None else np.asarray(episode_drift_m)
        self.noise_std_m = float(noise_std_m)
        self.action_delay_steps = int(action_delay_steps)
        if not 0 <= self.action_delay_steps <= 3:
            raise ValueError("action_delay_steps must be in [0,3] for the explicit delay state")
        self.history_length = int(history_length)
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        self.initial_seed = int(seed)
        self.frames, self.curvature = path_frames(trajectory.x_ref_m)
        self.action_bound_m = float(residual_action_bound_m)
        if not 0.0 < self.action_bound_m <= projector.max_cartesian_step_m:
            raise ValueError("residual_action_bound_m must be positive and no larger than projector bound")
        self.index = 0
        self.current_error_m = np.zeros(3)
        self.previous_action_local_m = np.zeros(3)
        self.current_condition_number = 0.0
        self._rng = np.random.default_rng(seed)
        self._noise = np.zeros_like(trajectory.x_ref_m)
        self._action_queue: list[np.ndarray] = []
        self._error_history = np.zeros((self.history_length, 3))
        self._action_history = np.zeros((self.history_length, 3))
        self._history_validity = np.zeros(self.history_length)

    @property
    def observation_layout(self) -> dict[str, slice]:
        cursor = 0
        layout = {}
        for name, width in (
            ("q_rad", 6), ("x_ref_m", 3), ("error_history_m", 3 * self.history_length),
            ("action_history_m", 3 * self.history_length), ("history_validity", self.history_length),
            ("path_delta_m", 3), ("path_tangent", 3), ("scalars", 3), ("delay_one_hot", 4),
        ):
            layout[name] = slice(cursor, cursor + width); cursor += width
        return layout

    @property
    def next_frame(self) -> np.ndarray:
        return self.frames[min(self.index + 1, len(self.frames) - 1)]

    @property
    def observation(self) -> np.ndarray:
        """Current deployable observation; hidden plant parameters are excluded."""
        return self._observation().copy()

    def _base_world(self, index: int) -> np.ndarray:
        if self.base_prior is None:
            return np.zeros(3)
        predicted_error = np.asarray(
            self.base_prior(self.trajectory.q_ref_rad[index], self.trajectory.x_ref_m[index]), dtype=float
        )
        return -predicted_error

    def _evaluate_point(self, index: int, residual_local_m: np.ndarray) -> tuple[np.ndarray, dict]:
        q_ref = self.trajectory.q_ref_rad[index]
        requested_world = self._base_world(index) + self.frames[index] @ residual_local_m
        projection = self.projector.project_world(q_ref, requested_world)
        q_command = q_ref + projection.delta_q_rad
        physical_q = q_command + self.joint_zero_offset_rad
        kinematic_position = self.plant_kinematics.position(physical_q) + self.tcp_offset_m
        actual = (
            kinematic_position
            + self.true_error(physical_q, kinematic_position)
            + self.episode_drift_m
            + self._noise[index]
        )
        error = actual - self.trajectory.x_ref_m[index]
        info = {
            "index": index,
            "q_command_rad": q_command,
            "q_physical_rad": physical_q,
            "x_actual_m": actual,
            "error_m": error,
            "requested_world_m": projection.requested_world_m,
            "executed_world_m": projection.executed_world_m,
            "safety_clipped": projection.clipped,
            "safety_reasons": projection.reasons,
            "condition_number": projection.condition_number,
        }
        return error, info

    def _observation(self) -> np.ndarray:
        next_index = min(self.index + 1, len(self.trajectory.q_ref_rad) - 1)
        path_delta = self.trajectory.x_ref_m[next_index] - self.trajectory.x_ref_m[self.index]
        return np.concatenate(
            (
                self.trajectory.q_ref_rad[self.index],
                self.trajectory.x_ref_m[self.index],
                self._error_history.reshape(-1),
                self._action_history.reshape(-1),
                self._history_validity,
                path_delta,
                self.frames[self.index, :, 0],
                np.array(
                    [
                        self.curvature[self.index],
                        self.index / (len(self.frames) - 1),
                        np.log1p(self.current_condition_number) / 10.0,
                    ]
                ),
                np.eye(4, dtype=float)[self.action_delay_steps],
            )
        ).astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        used_seed = self.initial_seed if seed is None else int(seed)
        self._rng = np.random.default_rng(used_seed)
        self._noise = self._rng.normal(0.0, self.noise_std_m, size=self.trajectory.x_ref_m.shape)
        self.index = 0
        self.previous_action_local_m = np.zeros(3)
        self._error_history.fill(0.0)
        self._action_history.fill(0.0)
        self._history_validity.fill(0.0)
        self._action_queue = [np.zeros(3) for _ in range(self.action_delay_steps)]
        self.current_error_m, info = self._evaluate_point(0, np.zeros(3))
        self._error_history[-1] = self.current_error_m
        self._history_validity[-1] = 1.0
        self.current_condition_number = info["condition_number"]
        info["seed"] = used_seed
        return self._observation(), info

    def step(self, action_local_m: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_action = np.asarray(action_local_m, dtype=float)
        if requested_action.shape != (3,) or not np.isfinite(requested_action).all():
            raise ValueError("action must be a finite vector with shape (3,)")
        norm = np.linalg.norm(requested_action)
        input_clipped = norm > self.action_bound_m
        if norm > self.action_bound_m:
            requested_action = requested_action * self.action_bound_m / norm
        self._action_queue.append(requested_action)
        applied_action = self._action_queue.pop(0)
        next_index = self.index + 1
        error, info = self._evaluate_point(next_index, applied_action)
        if input_clipped:
            info["safety_clipped"] = True
            info["safety_reasons"] = ("residual_action_bound",) + info["safety_reasons"]
        error_mm = error * 1000.0
        action_scale = max(self.action_bound_m, 1e-9)
        action_cost = np.sum((applied_action / action_scale) ** 2)
        smoothness_cost = np.sum(((applied_action - self.previous_action_local_m) / action_scale) ** 2)
        safety_cost = 1.0 if info["safety_clipped"] else 0.0
        reward = -float(np.sum(error_mm**2) + 0.03 * action_cost + 0.02 * smoothness_cost + 0.2 * safety_cost)
        self.index = next_index
        self.current_error_m = error
        self.current_condition_number = info["condition_number"]
        self.previous_action_local_m = applied_action
        self._error_history[:-1] = self._error_history[1:]
        self._error_history[-1] = error
        self._action_history[:-1] = self._action_history[1:]
        self._action_history[-1] = applied_action
        self._history_validity[:-1] = self._history_validity[1:]
        self._history_validity[-1] = 1.0
        terminated = self.index == len(self.trajectory.q_ref_rad) - 1
        info.update(
            {
                "requested_action_local_m": requested_action,
                "applied_action_local_m": applied_action,
                "action_input_clipped": input_clipped,
                "reward_terms": {
                    "tracking": -float(np.sum(error_mm**2)),
                    "action": -float(0.03 * action_cost),
                    "smoothness": -float(0.02 * smoothness_cost),
                    "safety": -float(0.2 * safety_cost),
                },
            }
        )
        return self._observation(), reward, terminated, False, info
