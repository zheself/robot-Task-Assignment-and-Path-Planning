#!/usr/bin/env python3
"""Train multi-prior SAC/TD3; validation chooses checkpoints, test is never loaded."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.algorithms import ValidationRMSECallback
from safe_residual_rl.envs import RealCalibratedSuite
from safe_residual_rl.envs.gymnasium_env import DomainRandomizedGymnasiumResidualEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=6000)
    parser.add_argument("--seeds", default="401,402,403")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sequence_rl")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) < 3:
        raise ValueError("at least three training seeds are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite = RealCalibratedSuite(args.root, tree_estimators=40)
    train_factory = suite.core_factory("train")
    validation_factory = suite.core_factory("validation")
    validation_seeds = (501, 502, 503)
    summary = {
        "evidence_level": "SIM_CALIBRATED",
        "real_source_evidence": "REAL_STATIC_UNVERIFIED_METADATA",
        "purpose": "multi_prior_rl_training_and_validation_checkpoint_selection",
        "candidate_split_hash": suite.candidate_split["sha256"],
        "candidate_split_status": suite.candidate_split["status"],
        "git_commit": "unavailable_repository_not_initialized",
        "train_prior_names": list(train_factory.prior_names),
        "validation_prior_names": list(validation_factory.prior_names),
        "frozen_test_prior_accessed": False,
        "training_seeds": list(seeds), "validation_seeds": list(validation_seeds),
        "timesteps": args.timesteps, "history_length": 4,
        "residual_action_bound_m": 0.002, "cartesian_projection_bound_m": 0.006,
        "tree_estimators_inner_loop": suite.tree_estimators,
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"),
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "torch": importlib.metadata.version("torch"),
        },
        "runs": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for algorithm_index, algorithm in enumerate(("sac", "td3")):
        summary["runs"][algorithm] = {}
        for seed in seeds:
            run_dir = args.output_dir / algorithm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            env = DomainRandomizedGymnasiumResidualEnv(train_factory, initial_seed=seed)
            common = dict(
                policy="MlpPolicy", env=env, learning_rate=3e-4,
                buffer_size=max(8000, args.timesteps * 2), learning_starts=min(500, args.timesteps // 5),
                batch_size=64, gamma=0.98, train_freq=1, gradient_steps=1,
                policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=0, device="cpu",
            )
            if algorithm == "sac":
                model = SAC(**common)
            else:
                model = TD3(
                    **common, action_noise=NormalActionNoise(np.zeros(3), np.full(3, 0.12)), policy_delay=2
                )
            callback = ValidationRMSECallback(
                validation_factory, validation_seeds, eval_freq=max(250, args.timesteps // 8), output_dir=run_dir
            )
            model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
            model.save(run_dir / "final_model")
            history = callback.history
            (run_dir / "validation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            config = {
                "algorithm": algorithm, "seed": seed, "timesteps": args.timesteps,
                "candidate_split_hash": suite.candidate_split["sha256"],
                "train_prior_names": list(train_factory.prior_names), "validation_seeds": list(validation_seeds),
                "best_validation_rmse_mm": callback.best_rmse,
                "git_commit": "unavailable_repository_not_initialized",
                "versions": summary["versions"],
                "failure_if_no_checkpoint": not (run_dir / "best_model.zip").exists(),
            }
            (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            summary["runs"][algorithm][str(seed)] = {"best_validation_rmse_mm": callback.best_rmse, "history": history}
            axes[algorithm_index].plot(
                [item["timesteps"] for item in history], [item["validation_rmse_mm"] for item in history],
                marker="o", label=f"seed {seed}",
            )
        axes[algorithm_index].set_title(algorithm.upper()); axes[algorithm_index].set_xlabel("Training timesteps")
        axes[algorithm_index].grid(alpha=0.25); axes[algorithm_index].legend()
    axes[0].set_ylabel("Validation trajectory RMSE (mm)")
    fig.suptitle("SIM_CALIBRATED validation learning curves (test prior not accessed)")
    fig.savefig(args.output_dir / "validation_learning_curves.png", dpi=180); plt.close(fig)
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({a: {s: v["best_validation_rmse_mm"] for s, v in runs.items()} for a, runs in summary["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
