#!/usr/bin/env python3
"""Run the pre-advisor synthetic end-to-end smoke pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safe_residual_rl.algorithms.cem import feedback_action, train_diagonal_feedback_cem
from safe_residual_rl.algorithms.ilc import evaluate_repeated_path_ilc
from safe_residual_rl.data import SyntheticErrorField, generate_measurement_dataset, generate_reference_path, load_manifest
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.evaluation import evaluate_policy
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.models import MeanErrorPrior, RidgeErrorPrior
from safe_residual_rl.safety import SafetyProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pre_advisor_smoke")
    parser.add_argument("--quick", action="store_true", help="Use a smaller CEM run for CI/local checks")
    return parser.parse_args()


def prior_metrics(prior: RidgeErrorPrior, dataset) -> dict:
    target = dataset.error_m
    prediction = prior.predict(dataset.q_rad, dataset.x_nominal_m)
    zero_rmse = np.sqrt(np.mean(np.sum((target * 1000.0) ** 2, axis=1)))
    model_rmse = np.sqrt(np.mean(np.sum(((prediction - target) * 1000.0) ** 2, axis=1)))
    return {
        "zero_prediction_rmse_mm": float(zero_rmse),
        "prior_rmse_mm": float(model_rmse),
        "relative_improvement_pct": float(100.0 * (zero_rmse - model_rmse) / zero_rmse),
    }


def make_env_factory(kinematics, error_field, projector, prior, scenario):
    def factory(seed: int) -> ResidualTrajectoryEnv:
        plant_kinematics = kinematics
        joint_zero = np.zeros(6)
        tcp_offset = np.zeros(3)
        if scenario == "train":
            workspace_shift, drift, noise, delay, path_seed = (
                0.0, np.array([0.2, -0.1, 0.15]) * 1e-3, 0.00015, 0, seed
            )
        elif scenario == "unseen_path":
            workspace_shift, drift, noise, delay, path_seed = (
                0.0, np.array([0.25, -0.05, 0.10]) * 1e-3, 0.00015, 0, 10_000 + seed
            )
        elif scenario == "cross_date_drift":
            workspace_shift, drift, noise, delay, path_seed = (
                0.0, np.array([1.0, -0.65, 0.55]) * 1e-3, 0.00018, 0, 20_000 + seed
            )
        elif scenario == "workspace_holdout":
            workspace_shift, drift, noise, delay, path_seed = (
                0.45, np.array([0.4, -0.2, 0.3]) * 1e-3, 0.00018, 0, 30_000 + seed
            )
        elif scenario == "noise_delay":
            workspace_shift, drift, noise, delay, path_seed = (
                0.0, np.array([0.7, -0.4, 0.4]) * 1e-3, 0.00035, 2, 40_000 + seed
            )
        elif scenario in ("kinematic_perturbation", "combined_shift"):
            rng = np.random.default_rng(50_000 + seed)
            workspace_shift = 0.25 if scenario == "combined_shift" else 0.0
            drift = np.array([0.8, -0.5, 0.45]) * 1e-3
            noise = 0.00030 if scenario == "combined_shift" else 0.00018
            delay = 2 if scenario == "combined_shift" else 0
            path_seed = (60_000 if scenario == "combined_shift" else 50_000) + seed
            perturbed_dh = kinematics.dh.copy()
            perturbed_dh[:, 0] += rng.normal(0.0, 0.00008, size=6)
            perturbed_dh[:, 2] += rng.normal(0.0, 0.00008, size=6)
            plant_kinematics = UR5Kinematics(perturbed_dh)
            joint_zero = rng.normal(0.0, 0.0007, size=6)
            tcp_offset = rng.normal(0.0, 0.00025, size=3)
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        q_ref, x_ref = generate_reference_path(kinematics, path_seed, length=64, workspace_shift=workspace_shift)
        return ResidualTrajectoryEnv(
            kinematics=kinematics,
            trajectory=Trajectory(q_ref, x_ref, path_id=f"{scenario}_{path_seed}"),
            true_error=error_field,
            projector=projector,
            base_prior=None if prior is None else prior.predict,
            plant_kinematics=plant_kinematics,
            joint_zero_offset_rad=joint_zero,
            tcp_offset_m=tcp_offset,
            episode_drift_m=drift,
            noise_std_m=noise,
            action_delay_steps=delay,
            seed=seed,
        )
    return factory


def render_markdown(report: dict) -> str:
    lines = [
        "# Pre-advisor synthetic pipeline smoke report", "",
        "> Evidence level: **synthetic_for_pipeline_validation_only**. These results verify software and experimental interfaces; they are not real-robot, factory, hemming, or sim-to-real evidence.", "",
        "## Data and error-prior check", "",
        f"- Training samples: {report['data']['train_samples']}",
        f"- Validation samples: {report['data']['validation_samples']}",
        f"- Held-out cross-date samples: {report['data']['cross_date_samples']}",
        f"- Validation prior RMSE: {report['prior']['validation']['prior_rmse_mm']:.3f} mm",
        f"- Cross-date prior RMSE: {report['prior']['cross_date']['prior_rmse_mm']:.3f} mm", "",
        "## Learned smoke policy", "",
        f"CEM diagonal feedback gains: `{report['policy_search']['gains']}`. CEM is only a learning-loop check; SAC/TD3 remain planned paper baselines.", "",
        "## Paired trajectory results", "",
        "| Scenario | Method | RMSE mm | P95 mm | Action TV mm | Safety clip rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario, methods in report["evaluation"].items():
        for method, summary in methods.items():
            mean = summary["mean"]
            lines.append(
                f"| {scenario} | {method} | {mean['rmse_mm']:.3f} | {mean['p95_mm']:.3f} | "
                f"{mean['action_total_variation_mm']:.3f} | {mean['safety_clip_rate']:.3f} |"
            )
    lines += ["", "## Interpretation boundary", "",
        "A lower synthetic error only shows that the sequential interface, safety layer and evaluation code can express a compensation problem. The next evidence upgrade is to fit error/noise/drift from verified group measurements and rerun the unchanged protocol.", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PROJECT_ROOT / "data" / "manifests" / "synthetic_ur5_pre_advisor_v1.json")
    kinematics = UR5Kinematics()
    error_field = SyntheticErrorField(seed=17)
    projector = SafetyProjector(kinematics)
    train = generate_measurement_dataset(kinematics, error_field, 11, "train", "date_A", path_count=6)
    validation = generate_measurement_dataset(kinematics, error_field, 12, "validation", "date_A", path_count=3)
    cross_date = generate_measurement_dataset(
        kinematics, error_field, 13, "cross_date", "date_B", path_count=3,
        drift_m=np.array([1.0, -0.65, 0.55]) * 1e-3,
    )
    train_groups = set(train.session_id) | set(train.path_id)
    held_out_groups = set(validation.session_id) | set(validation.path_id) | set(cross_date.session_id) | set(cross_date.path_id)
    if train_groups & held_out_groups:
        raise RuntimeError("group leakage detected between training and held-out data")
    prior = RidgeErrorPrior(regularization=1e-3).fit(train.q_rad, train.x_nominal_m, train.error_m)
    cem = train_diagonal_feedback_cem(
        make_env_factory(kinematics, error_field, projector, prior, "train"),
        training_seeds=(101, 102) if args.quick else (101, 102, 103, 104), seed=29,
        iterations=3 if args.quick else 8, population=10 if args.quick else 24,
    )
    test_seeds = (701, 702, 703) if args.quick else (701, 702, 703, 704, 705)
    evaluation = {}
    mean_prior = MeanErrorPrior().fit(train.error_m)
    for scenario in (
        "unseen_path", "cross_date_drift", "workspace_holdout", "noise_delay",
        "kinematic_perturbation", "combined_shift",
    ):
        no_comp = make_env_factory(kinematics, error_field, projector, None, scenario)
        with_mean_bias = make_env_factory(kinematics, error_field, projector, mean_prior, scenario)
        with_prior = make_env_factory(kinematics, error_field, projector, prior, scenario)
        policies = {
            "no_compensation": (no_comp, lambda env: np.zeros(3)),
            "mean_bias": (with_mean_bias, lambda env: np.zeros(3)),
            "projected_supervised_prior": (with_prior, lambda env: np.zeros(3)),
            "fixed_feedback": (with_prior, lambda env: feedback_action(env, np.ones(3))),
            "cem_smoke_policy": (with_prior, lambda env, gains=cem.gains: feedback_action(env, gains)),
        }
        evaluation[scenario] = {
            method: evaluate_policy(factory, policy, test_seeds)
            for method, (factory, policy) in policies.items()
        }
        evaluation[scenario]["ilc_repeated_path"] = evaluate_repeated_path_ilc(
            with_prior, test_seeds, iterations=3 if args.quick else 6
        )
    report = {
        "evidence_level": "synthetic_for_pipeline_validation_only",
        "split_policy": "grouped_by_session_and_path_before_model_fit",
        "manifest_sha256": manifest.sha256,
        "run_configuration": {
            "cem_training_seeds": [101, 102] if args.quick else [101, 102, 103, 104],
            "cem_seed": 29,
            "test_seeds": list(test_seeds),
            "residual_action_bound_m": 0.002,
            "total_cartesian_projection_bound_m": 0.006,
            "ilc_iterations": 3 if args.quick else 6,
            "ilc_protocol": "trajectory_specific_repeated_execution_with_independent_noise_draws",
        },
        "data": {"train_samples": len(train.q_rad), "validation_samples": len(validation.q_rad),
                 "cross_date_samples": len(cross_date.q_rad), "train_dates": sorted(set(train.date_id)),
                 "test_dates": sorted(set(cross_date.date_id))},
        "prior": {"validation": prior_metrics(prior, validation), "cross_date": prior_metrics(prior, cross_date)},
        "policy_search": {"algorithm": "CEM_smoke_only", "gains": cem.gains.tolist(), "history": list(cem.history)},
        "evaluation": evaluation,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output_dir / "synthetic_data_card.md").write_text(
        "# Synthetic data card\n\nPurpose: pipeline validation only.\n\nGenerated from the unverified standard-DH UR5 model with a hidden smooth millimetre-scale error field, session drift and Gaussian measurement noise. Splits use disjoint session/path identifiers. This dataset must not appear as real evidence in the paper.\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
