"""Small policy-search smoke test; not the proposed paper algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from safe_residual_rl.envs import ResidualTrajectoryEnv


EnvFactory = Callable[[int], ResidualTrajectoryEnv]


@dataclass(frozen=True)
class CEMResult:
    gains: np.ndarray
    history: tuple[float, ...]


def feedback_action(env: ResidualTrajectoryEnv, gains: np.ndarray) -> np.ndarray:
    local_error = env.next_frame.T @ env.current_error_m
    return -np.asarray(gains) * local_error


def _score(gains: np.ndarray, env_factory: EnvFactory, seeds: tuple[int, ...]) -> float:
    returns = []
    for seed in seeds:
        env = env_factory(seed)
        env.reset(seed=seed)
        total = 0.0
        terminated = False
        while not terminated:
            _, reward, terminated, _, _ = env.step(feedback_action(env, gains))
            total += reward
        returns.append(total)
    return float(np.mean(returns))


def train_diagonal_feedback_cem(
    env_factory: EnvFactory,
    training_seeds: tuple[int, ...],
    seed: int = 0,
    iterations: int = 8,
    population: int = 24,
    elite_fraction: float = 0.25,
) -> CEMResult:
    rng = np.random.default_rng(seed)
    mean = np.ones(3)
    std = np.full(3, 0.65)
    history: list[float] = []
    elite_count = max(2, int(population * elite_fraction))
    for _ in range(iterations):
        candidates = np.clip(rng.normal(mean, std, size=(population, 3)), 0.0, 2.5)
        scores = np.array([_score(candidate, env_factory, training_seeds) for candidate in candidates])
        elite = candidates[np.argsort(scores)[-elite_count:]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.05)
        history.append(float(scores.max()))
    return CEMResult(gains=mean, history=tuple(history))
