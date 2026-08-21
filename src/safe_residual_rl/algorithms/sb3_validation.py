from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from safe_residual_rl.envs.gymnasium_env import normalize_observation
from safe_residual_rl.evaluation import evaluate_policy


def model_policy(model):
    def policy(core):
        normalized = normalize_observation(core.observation, core.action_bound_m, core.observation_layout)
        action = model.predict(normalized, deterministic=True)[0]
        return action * core.action_bound_m
    return policy


class ValidationRMSECallback(BaseCallback):
    """Checkpoint selection uses validation factories only; test is inaccessible."""

    def __init__(self, validation_factory, validation_seeds, eval_freq: int, output_dir: Path) -> None:
        super().__init__(verbose=0)
        if getattr(validation_factory, "role", None) != "validation":
            raise ValueError("callback requires a validation-role factory")
        self.validation_factory = validation_factory
        self.validation_seeds = tuple(validation_seeds)
        self.eval_freq = int(eval_freq)
        self.output_dir = Path(output_dir)
        self.best_rmse = float("inf")
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        evaluation = evaluate_policy(self.validation_factory, model_policy(self.model), self.validation_seeds)
        rmse = evaluation["mean"]["rmse_mm"]
        record = {
            "timesteps": int(self.num_timesteps), "validation_rmse_mm": float(rmse),
            "validation_p95_mm": float(evaluation["mean"]["p95_mm"]),
            "validation_safety_clip_rate": float(evaluation["mean"]["safety_clip_rate"]),
        }
        self.history.append(record)
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.model.save(self.output_dir / "best_model")
        return True
