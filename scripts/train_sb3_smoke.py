#!/usr/bin/env python3
"""CPU-only SAC/TD3 integration smoke test on training-domain synthetic paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
import stable_baselines3

from safe_residual_rl.data import SyntheticErrorField, generate_measurement_dataset, generate_reference_path, load_manifest
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.envs.gymnasium_env import DomainRandomizedGymnasiumResidualEnv, normalize_observation
from safe_residual_rl.evaluation import evaluate_policy
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.models import RidgeErrorPrior
from safe_residual_rl.safety import SafetyProjector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sb3_smoke")
    return parser.parse_args()


def build_components():
    kinematics = UR5Kinematics()
    error_field = SyntheticErrorField(seed=17)
    projector = SafetyProjector(kinematics)
    training_measurements = generate_measurement_dataset(
        kinematics, error_field, 11, "train", "date_A", path_count=6
    )
    prior = RidgeErrorPrior().fit(
        training_measurements.q_rad, training_measurements.x_nominal_m, training_measurements.error_m
    )
    return kinematics, error_field, projector, prior


def training_factory(kinematics, error_field, projector, prior):
    def factory(seed: int):
        rng = np.random.default_rng(seed)
        q_ref, x_ref = generate_reference_path(
            kinematics, seed=100_000 + seed, length=48, workspace_shift=rng.uniform(-0.12, 0.12)
        )
        return ResidualTrajectoryEnv(
            kinematics, Trajectory(q_ref, x_ref, f"rl_train_{seed}"), error_field, projector,
            base_prior=prior.predict,
            episode_drift_m=rng.normal(0.0, 0.00035, size=3),
            noise_std_m=float(rng.uniform(0.00010, 0.00022)),
            action_delay_steps=int(rng.integers(0, 2)), seed=seed,
        )
    return factory


def test_factory(kinematics, error_field, projector, prior):
    def factory(seed: int):
        q_ref, x_ref = generate_reference_path(kinematics, 200_000 + seed, length=64, workspace_shift=0.2)
        return ResidualTrajectoryEnv(
            kinematics, Trajectory(q_ref, x_ref, f"rl_test_{seed}"), error_field, projector,
            base_prior=prior.predict,
            episode_drift_m=np.array([0.85, -0.55, 0.45]) * 1e-3,
            noise_std_m=0.00022, action_delay_steps=1, seed=seed,
        )
    return factory


def train_one(name, env, timesteps, seed, output_dir):
    common = dict(
        policy="MlpPolicy", env=env, learning_rate=3e-4, buffer_size=max(5000, timesteps * 2),
        learning_starts=min(250, max(50, timesteps // 5)), batch_size=64, gamma=0.98,
        train_freq=1, gradient_steps=1, policy_kwargs={"net_arch": [64, 64]},
        seed=seed, verbose=0, device="cpu",
    )
    if name == "sac":
        model = SAC(**common)
    elif name == "td3":
        model = TD3(
            **common,
            action_noise=NormalActionNoise(mean=np.zeros(3), sigma=np.full(3, 0.12)),
            policy_delay=2,
        )
    else:
        raise ValueError(name)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(output_dir / f"{name}_smoke")
    return model


def main():
    args = parse_args()
    timesteps = 500 if args.quick else args.timesteps
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(ROOT / "data" / "manifests" / "synthetic_ur5_pre_advisor_v1.json")
    kinematics, error_field, projector, prior = build_components()
    core_train_factory = training_factory(kinematics, error_field, projector, prior)
    gym_env = DomainRandomizedGymnasiumResidualEnv(core_train_factory, initial_seed=123)
    check_env(gym_env, warn=True, skip_render_check=True)
    held_out_factory = test_factory(kinematics, error_field, projector, prior)
    test_seeds = (901, 902, 903)
    algorithm_seeds = {"sac": 310, "td3": 311}
    common_hyperparameters = {
        "learning_rate": 3e-4,
        "buffer_size": max(5000, timesteps * 2),
        "learning_starts": min(250, max(50, timesteps // 5)),
        "batch_size": 64,
        "gamma": 0.98,
        "train_freq": 1,
        "gradient_steps": 1,
        "net_arch": [64, 64],
        "residual_action_bound_m": 0.002,
    }
    results = {
        "evidence_level": "SYNTHETIC",
        "purpose": "SB3_API_AND_LEARNING_LOOP_SMOKE_ONLY",
        "manifest_sha256": manifest.sha256,
        "training_timesteps_per_algorithm": timesteps,
        "git_commit": "unavailable_repository_not_initialized",
        "algorithm_seeds": algorithm_seeds,
        "test_seeds": list(test_seeds),
        "training_domain": {
            "workspace_shift_range": [-0.12, 0.12],
            "episode_drift_std_m": 0.00035,
            "measurement_noise_std_m_range": [0.00010, 0.00022],
            "action_delay_steps": [0, 1],
        },
        "held_out_domain": {
            "workspace_shift": 0.2,
            "episode_drift_m": [0.00085, -0.00055, 0.00045],
            "measurement_noise_std_m": 0.00022,
            "action_delay_steps": 1,
        },
        "common_hyperparameters": common_hyperparameters,
        "algorithm_specific": {
            "sac": {},
            "td3": {"normal_action_noise_sigma": 0.12, "policy_delay": 2},
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": importlib.metadata.version("torch"),
        },
        "evaluation": {
            "projected_supervised_prior": evaluate_policy(
                held_out_factory, lambda core: np.zeros(3), test_seeds
            )
        },
    }
    for name in ("sac", "td3"):
        model = train_one(name, gym_env, timesteps, seed=algorithm_seeds[name], output_dir=args.output_dir)
        policy = lambda core, trained=model: (
            trained.predict(
                normalize_observation(core.observation, core.action_bound_m, core.observation_layout), deterministic=True
            )[0]
            * core.action_bound_m
        )
        results["evaluation"][name] = evaluate_policy(held_out_factory, policy, test_seeds)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        method: values["mean"]["rmse_mm"] for method, values in results["evaluation"].items()
    }, indent=2))
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
