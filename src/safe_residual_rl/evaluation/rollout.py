from __future__ import annotations

from typing import Callable

import numpy as np

from safe_residual_rl.envs import ResidualTrajectoryEnv


Policy = Callable[[ResidualTrajectoryEnv], np.ndarray]


def rollout(env: ResidualTrajectoryEnv, policy: Policy, seed: int) -> dict:
    env.reset(seed=seed)
    errors, actions, rewards, clipped = [env.current_error_m.copy()], [], [], []
    terminated = False
    while not terminated:
        action = np.asarray(policy(env), dtype=float)
        _, reward, terminated, _, info = env.step(action)
        errors.append(info["error_m"])
        actions.append(info["applied_action_local_m"])
        rewards.append(reward)
        clipped.append(info["safety_clipped"])
    errors_m = np.asarray(errors)
    actions_m = np.asarray(actions)
    norms_mm = np.linalg.norm(errors_m, axis=1) * 1000.0
    action_variation_mm = (
        np.sum(np.linalg.norm(np.diff(actions_m, axis=0), axis=1)) * 1000.0 if len(actions_m) > 1 else 0.0
    )
    return {
        "rmse_mm": float(np.sqrt(np.mean(np.sum((errors_m * 1000.0) ** 2, axis=1)))),
        "mae_mm": float(np.mean(norms_mm)),
        "p95_mm": float(np.percentile(norms_mm, 95)),
        "max_mm": float(np.max(norms_mm)),
        "action_total_variation_mm": float(action_variation_mm),
        "safety_clip_rate": float(np.mean(clipped)) if clipped else 0.0,
        "return": float(np.sum(rewards)),
        "steps": int(len(errors_m)),
    }


def evaluate_policy(
    env_factory: Callable[[int], ResidualTrajectoryEnv], policy: Policy, seeds: tuple[int, ...]
) -> dict:
    runs = [rollout(env_factory(seed), policy, seed) for seed in seeds]
    metrics = {key: [run[key] for run in runs] for key in runs[0] if key != "steps"}
    return {
        "mean": {key: float(np.mean(values)) for key, values in metrics.items()},
        "std": {key: float(np.std(values)) for key, values in metrics.items()},
        "runs": runs,
    }
