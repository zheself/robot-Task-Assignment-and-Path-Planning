"""Optional Gymnasium adapter. The core environment remains NumPy-only."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - exercised only without optional dependency
    raise ImportError("Install the project dependencies to use GymnasiumResidualEnv") from error

from .core import ResidualTrajectoryEnv


def normalize_observation(observation: np.ndarray, action_bound_m: float, layout: dict[str, slice]) -> np.ndarray:
    """Scale known physical fields without fitting any test-domain statistics."""
    scaled = np.asarray(observation, dtype=np.float32).copy()
    scaled[layout["q_rad"]] /= np.pi
    scaled[layout["error_history_m"]] *= 1000.0
    scaled[layout["action_history_m"]] /= action_bound_m
    scaled[layout["path_delta_m"]] *= 100.0
    return scaled


class GymnasiumResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, core_env: ResidualTrajectoryEnv) -> None:
        super().__init__()
        self.core = core_env
        observation, _ = self.core.reset()
        observation = normalize_observation(observation, self.core.action_bound_m, self.core.observation_layout)
        limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-limit, limit, shape=observation.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        actual_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        observation, info = self.core.reset(seed=actual_seed)
        return normalize_observation(observation, self.core.action_bound_m, self.core.observation_layout), info

    def step(self, action: np.ndarray):
        physical_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.core.action_bound_m
        observation, reward, terminated, truncated, info = self.core.step(physical_action)
        return normalize_observation(observation, self.core.action_bound_m, self.core.observation_layout), reward, terminated, truncated, info


class DomainRandomizedGymnasiumResidualEnv(gym.Env):
    """Build a new hidden plant/path from a training-only factory every reset."""

    metadata = {"render_modes": []}

    def __init__(self, core_factory, initial_seed: int = 0) -> None:
        super().__init__()
        self.core_factory = core_factory
        self.initial_seed = int(initial_seed)
        self.core = core_factory(self.initial_seed)
        observation, _ = self.core.reset(seed=self.initial_seed)
        observation = normalize_observation(observation, self.core.action_bound_m, self.core.observation_layout)
        limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-limit, limit, shape=observation.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        actual_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.core = self.core_factory(actual_seed)
        observation, info = self.core.reset(seed=actual_seed)
        return normalize_observation(observation, self.core.action_bound_m, self.core.observation_layout), info

    def step(self, action: np.ndarray):
        physical_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.core.action_bound_m
        observation, reward, terminated, truncated, info = self.core.step(physical_action)
        return normalize_observation(observation, self.core.action_bound_m, self.core.observation_layout), reward, terminated, truncated, info
