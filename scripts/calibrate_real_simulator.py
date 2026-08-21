#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.algorithms.cem import feedback_action, train_diagonal_feedback_cem
from safe_residual_rl.data import (
    RealSupportTrajectoryGenerator, candidate_split_document, load_ur5_static_csvs, masks_from_candidate_split,
)
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.evaluation import evaluate_policy
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.models import SklearnStaticPrior, fit_calibration_profile, fit_prior, make_static_priors
from safe_residual_rl.safety import SafetyProjector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "real_calibrated_simulator")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_ur5_static_csvs(args.root, joint_unit="degree", length_unit="mm")
    dataset = loaded.dataset
    candidate = candidate_split_document(); masks = masks_from_candidate_split(dataset, candidate)
    train = masks["train"]
    suite = make_static_priors(seed=2026)
    # The benchmark uses 300 trees. A declared compact prior is used inside the
    # sequential simulator to keep single-step prediction tractable.
    suite["extra_trees"] = SklearnStaticPrior(
        ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2, max_features=1.0, random_state=2026, n_jobs=1)
    )
    selected_names = ("ridge", "extra_trees", "rbf_kernel")
    models, profiles = {}, {}
    for name in selected_names:
        model = suite[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_prior(model, dataset.q_rad[train], dataset.x_nominal_m[train], dataset.error_m[train])
        models[name] = model
        profiles[name] = fit_calibration_profile(name, model, dataset, train)

    kinematics = UR5Kinematics(); projector = SafetyProjector(kinematics)
    generator = RealSupportTrajectoryGenerator(kinematics).fit(dataset.q_rad[train], dataset.x_nominal_m[train])
    trajectory_audit = {
        kind: [generator.generate(kind, seed=700 + index, length=64).ood for index in range(5)]
        for kind in ("sine", "smooth_random", "workspace_holdout")
    }
    base_prior = models["ridge"]

    def env_factory(prior_names, test_prior=None):
        def factory(seed: int):
            name = test_prior if test_prior is not None else prior_names[seed % len(prior_names)]
            kind = "sine" if seed % 2 == 0 else "smooth_random"
            generated = generator.generate(kind, seed=10_000 + seed, length=64)
            field = profiles[name].sample_episode_error(models[name], seed=20_000 + seed)
            return ResidualTrajectoryEnv(
                kinematics, Trajectory(generated.q_ref_rad, generated.x_ref_m, f"{kind}_{seed}"),
                field, projector, base_prior=base_prior.predict, noise_std_m=0.0,
                action_delay_steps=1, residual_action_bound_m=0.002, seed=seed,
            )
        return factory

    iterations = 2 if args.quick else 5
    population = 6 if args.quick else 12
    training_seeds = (101, 102) if args.quick else (101, 102, 103, 104, 105, 106)
    single = train_diagonal_feedback_cem(
        env_factory(("ridge",)), training_seeds, seed=41, iterations=iterations, population=population
    )
    multi = train_diagonal_feedback_cem(
        env_factory(("ridge", "extra_trees")), training_seeds, seed=42, iterations=iterations, population=population
    )
    unseen_factory = env_factory(("rbf_kernel",), test_prior="rbf_kernel")
    test_seeds = (801, 802, 803) if args.quick else (801, 802, 803, 804, 805)
    evaluation = {
        "base_projected_ridge": evaluate_policy(unseen_factory, lambda env: np.zeros(3), test_seeds),
        "single_prior_cem": evaluate_policy(
            unseen_factory, lambda env, gains=single.gains: feedback_action(env, gains), test_seeds
        ),
        "multi_prior_cem": evaluate_policy(
            unseen_factory, lambda env, gains=multi.gains: feedback_action(env, gains), test_seeds
        ),
    }
    result = {
        "evidence_level": "SIM_CALIBRATED",
        "real_source_evidence": "REAL_STATIC_UNVERIFIED_METADATA",
        "candidate_split_hash": candidate["sha256"],
        "candidate_split_status": candidate["status"],
        "git_commit": "unavailable_repository_not_initialized",
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "run_mode": "quick" if args.quick else "standard",
        "terminology_boundary": {
            "model_residual": "unexplained_residual",
            "file_mean_difference": "session_shift_proxy",
            "prohibited_unverified_terms": ["measurement_noise", "physical_calibration_drift"],
        },
        "training_prior_names": ["ridge", "extra_trees"],
        "unseen_prior_name": "rbf_kernel",
        "simulator_prior_capacity": {"extra_trees_n_estimators": 60, "benchmark_extra_trees_n_estimators": 300},
        "calibration": {name: profile.summary for name, profile in profiles.items()},
        "trajectory_generators": trajectory_audit,
        "policy_search": {
            "algorithm": "CEM_smoke_not_paper_method", "training_seeds": list(training_seeds),
            "test_seeds": list(test_seeds), "single_prior_gains": single.gains.tolist(),
            "multi_prior_gains": multi.gains.tolist(), "iterations": iterations, "population": population,
        },
        "unseen_prior_evaluation": evaluation,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# SIM_CALIBRATED multi-prior check", "",
        "> Calibrated from REAL_STATIC data under unverified degree/mm/frame/TCP assumptions. Unexplained residual is not called measurement noise; session-shift proxy is not called physical drift.", "",
        "| Policy | Unseen RBF-prior RMSE mm | P95 mm | Action TV mm | Safety clip rate |", "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in evaluation.items():
        mean = metrics["mean"]
        lines.append(f"| {name} | {mean['rmse_mm']:.3f} | {mean['p95_mm']:.3f} | {mean['action_total_variation_mm']:.3f} | {mean['safety_clip_rate']:.3f} |")
    lines += ["", "Single-prior and multi-prior CEM are diagnostic feedback policies, not the proposed RL method.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
