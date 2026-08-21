"""Trajectory-specific iterative learning control baseline."""

from __future__ import annotations

from typing import Callable

import numpy as np

from safe_residual_rl.envs import ResidualTrajectoryEnv
from safe_residual_rl.evaluation.rollout import rollout


def _smooth(actions: np.ndarray, weight: float) -> np.ndarray:
    if len(actions) < 3 or weight <= 0.0:
        return actions
    result = actions.copy()
    result[1:-1] = (1.0 - weight) * actions[1:-1] + 0.5 * weight * (actions[:-2] + actions[2:])
    return result


def train_ilc_actions(
    env_factory: Callable[[int], ResidualTrajectoryEnv],
    seed: int,
    iterations: int = 6,
    learning_gain: float = 0.45,
    smoothing_weight: float = 0.25,
) -> np.ndarray:
    probe = env_factory(seed)
    actions = np.zeros((len(probe.trajectory.q_ref_rad) - 1, 3), dtype=float)
    for iteration in range(iterations):
        env = env_factory(seed)
        # Same path and episode drift, but a different measurement-noise draw
        # on every physical repeat.
        env.reset(seed=100_000 + seed + iteration)
        errors_local = []
        for step in range(len(actions)):
            _, _, terminated, _, info = env.step(actions[step])
            errors_local.append(env.frames[env.index].T @ info["error_m"])
            if terminated:
                break
        actions -= learning_gain * np.asarray(errors_local)
        actions = _smooth(actions, smoothing_weight)
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        actions *= np.minimum(1.0, env.action_bound_m / np.maximum(norms, 1e-12))
    return actions


def evaluate_repeated_path_ilc(
    env_factory: Callable[[int], ResidualTrajectoryEnv], seeds: tuple[int, ...], iterations: int = 6
) -> dict:
    runs = []
    for seed in seeds:
        actions = train_ilc_actions(env_factory, seed, iterations=iterations)
        runs.append(rollout(env_factory(seed), lambda env, values=actions: values[env.index], 200_000 + seed))
    metrics = {key: [run[key] for run in runs] for key in runs[0] if key != "steps"}
    return {
        "mean": {key: float(np.mean(values)) for key, values in metrics.items()},
        "std": {key: float(np.std(values)) for key, values in metrics.items()},
        "runs": runs,
        "protocol": "trajectory_specific_repeated_execution_with_independent_noise_draws",
        "iterations": iterations,
    }
