#!/usr/bin/env python3
"""Evaluate validation-selected checkpoints once on frozen unseen-prior tests."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, TD3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.algorithms import model_policy
from safe_residual_rl.envs import RealCalibratedSuite
from safe_residual_rl.evaluation import evaluate_policy


def aggregate(evaluations):
    runs = [run for evaluation in evaluations for run in evaluation["runs"]]
    keys = [key for key in runs[0] if key != "steps"]
    return {"mean": {key: float(np.mean([run[key] for run in runs])) for key in keys}, "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, default=ROOT / "outputs" / "sequence_rl")
    parser.add_argument("--seeds", default="401,402,403")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "frozen_rl_test")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",")); test_seeds = (901, 902, 903, 904, 905)
    suite = RealCalibratedSuite(args.root, tree_estimators=40)
    scenarios = {
        "unseen_rbf_prior_unseen_path": suite.core_factory("test", "nominal"),
        "unseen_rbf_prior_workspace_holdout": suite.core_factory("test", "workspace_holdout"),
    }
    result = {
        "evidence_level": "SIM_CALIBRATED_AND_SIM_STRESS",
        "candidate_split_hash": suite.candidate_split["sha256"],
        "candidate_split_status": suite.candidate_split["status"],
        "frozen_unseen_prior": suite.UNSEEN_TEST_PRIOR, "test_seeds": list(test_seeds),
        "checkpoint_selection": "validation_only", "results": {}, "failures": [],
        "git_commit": "unavailable_repository_not_initialized",
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"),
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "torch": importlib.metadata.version("torch"),
        },
    }
    rows = []
    for scenario, factory in scenarios.items():
        baseline = evaluate_policy(factory, lambda env: np.zeros(3), test_seeds)
        result["results"][scenario] = {"projected_ridge_base": baseline, "sac": {}, "td3": {}}
        rows.append({"scenario": scenario, "method": "projected_ridge_base", "training_seed": "none", **baseline["mean"]})
        for algorithm, model_class in (("sac", SAC), ("td3", TD3)):
            evaluations = []
            for seed in seeds:
                checkpoint = args.training_dir / algorithm / f"seed_{seed}" / "best_model.zip"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                model = model_class.load(checkpoint, device="cpu")
                evaluation = evaluate_policy(factory, model_policy(model), test_seeds)
                result["results"][scenario][algorithm][str(seed)] = evaluation
                evaluations.append(evaluation)
                rows.append({"scenario": scenario, "method": algorithm, "training_seed": seed, **evaluation["mean"]})
            result["results"][scenario][f"{algorithm}_aggregate"] = aggregate(evaluations)
            aggregate_rmse = result["results"][scenario][f"{algorithm}_aggregate"]["mean"]["rmse_mm"]
            if aggregate_rmse >= baseline["mean"]["rmse_mm"]:
                result["failures"].append(
                    f"{algorithm} did not beat projected base in {scenario}: {aggregate_rmse:.3f} >= {baseline['mean']['rmse_mm']:.3f} mm"
                )
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scenario", "method", "training_seed", "rmse_mm", "mae_mm", "p95_mm", "max_mm", "action_total_variation_mm", "safety_clip_rate", "return"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (scenario, values) in zip(axes, result["results"].items()):
        labels = ["base", "SAC", "TD3"]
        rmse = [
            values["projected_ridge_base"]["mean"]["rmse_mm"],
            values["sac_aggregate"]["mean"]["rmse_mm"], values["td3_aggregate"]["mean"]["rmse_mm"],
        ]
        axis.bar(labels, rmse); axis.set_title(scenario.replace("_", "\n")); axis.set_ylabel("RMSE mm"); axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Frozen unseen-prior test (SIM_CALIBRATED / SIM_STRESS)")
    fig.savefig(args.output_dir / "frozen_test_comparison.png", dpi=180); plt.close(fig)
    lines = [
        "# Frozen multi-seed RL test", "",
        "> Validation selected checkpoints. Frozen tests use unseen RBF simulator prior. These are SIM_CALIBRATED/SIM_STRESS results, not real trajectories.", "",
        "| Scenario | Base RMSE | SAC aggregate RMSE | TD3 aggregate RMSE |", "|---|---:|---:|---:|",
    ]
    for scenario, values in result["results"].items():
        lines.append(
            f"| {scenario} | {values['projected_ridge_base']['mean']['rmse_mm']:.3f} | "
            f"{values['sac_aggregate']['mean']['rmse_mm']:.3f} | {values['td3_aggregate']['mean']['rmse_mm']:.3f} |"
        )
    lines += ["", "Failures retained:", ""] + [f"- {failure}" for failure in result["failures"]]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
