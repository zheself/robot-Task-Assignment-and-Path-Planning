#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports" / "pre_advisor"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((OUTPUTS / "ur5_static_analysis" / "analysis.json").read_text())
    priors = json.loads((OUTPUTS / "real_static_priors" / "summary.json").read_text())
    calibrated = json.loads((OUTPUTS / "real_calibrated_simulator" / "summary.json").read_text())
    frozen = json.loads((OUTPUTS / "frozen_rl_test" / "summary.json").read_text())

    # Per-file quality and FK compatibility.
    files = [item["file"] for item in analysis["per_file"]]
    short = [name.replace("建模数据", "model").replace("验证数据", "validation") for name in files]
    error_mean = [item["position_error_norm_mm"]["mean"] for item in analysis["per_file"]]
    fk_median = [item["fk_nominal_delta_mm"]["median"] for item in analysis["per_file"]]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].bar(np.arange(len(files)), error_mean); axes[0].set_ylabel("Mean position error norm (mm)")
    axes[0].set_title("REAL_STATIC per-file quality (units/frames/TCP unverified)"); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(np.arange(len(files)), fk_median, color="tab:orange"); axes[1].set_ylabel("Median FK–nominal delta (mm)")
    axes[1].axhline(3.0, color="red", linestyle="--", label="diagnostic threshold 3 mm")
    axes[1].set_xticks(np.arange(len(files)), short, rotation=35, ha="right", fontsize=8); axes[1].legend(); axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(REPORTS / "real_static_per_file_quality.png", dpi=180); plt.close(fig)

    # Static prior model comparison.
    model_names = list(priors["results"])
    roles = ("validation", "test_cross_date")
    x = np.arange(len(model_names)); width = 0.38
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for index, role in enumerate(roles):
        values = [priors["results"][name][role]["overall"]["rmse_mm"] for name in model_names]
        axis.bar(x + (index - 0.5) * width, values, width, label=role)
    axis.set_xticks(x, model_names, rotation=25, ha="right"); axis.set_ylabel("Position-error prediction RMSE (mm)")
    axis.set_title("REAL_STATIC supervised priors (simulator priors/baselines only)"); axis.legend(); axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(0, min(10, max(5, axis.get_ylim()[1])))
    fig.savefig(REPORTS / "real_static_prior_comparison.png", dpi=180); plt.close(fig)

    # Calibration terminology-safe summary and trajectory OOD.
    names = list(calibrated["calibration"])
    residual_p95 = [calibrated["calibration"][name]["unexplained_residual_norm_mm"]["p95"] for name in names]
    shift_p95 = [calibrated["calibration"][name]["session_shift_proxy_norm_mm"]["p95"] for name in names]
    kinds = list(calibrated["trajectory_generators"])
    ood_p95 = [np.mean([run["nearest_standardized_q_tcp_distance"]["p95"] for run in calibrated["trajectory_generators"][kind]]) for kind in kinds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(np.arange(len(names)) - 0.18, residual_p95, 0.36, label="unexplained residual P95")
    axes[0].bar(np.arange(len(names)) + 0.18, shift_p95, 0.36, label="session-shift proxy P95")
    axes[0].set_xticks(np.arange(len(names)), names); axes[0].set_ylabel("mm"); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(kinds, ood_p95, color="tab:green"); axes[1].set_ylabel("Mean trajectory P95 support distance")
    axes[1].set_title("Training-support OOD diagnostic"); axes[1].tick_params(axis="x", rotation=20); axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("SIM_CALIBRATED residual/shift proxies and generated trajectories")
    fig.savefig(REPORTS / "sim_calibration_and_trajectory_ood.png", dpi=180); plt.close(fig)

    # Compact CSV tables.
    with (REPORTS / "real_static_prior_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["model", "validation_rmse_mm", "cross_date_test_rmse_mm", "test_p95_mm"])
        for name in model_names:
            writer.writerow([
                name, priors["results"][name]["validation"]["overall"]["rmse_mm"],
                priors["results"][name]["test_cross_date"]["overall"]["rmse_mm"],
                priors["results"][name]["test_cross_date"]["overall"]["p95_mm"],
            ])
    with (REPORTS / "frozen_rl_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["scenario", "method", "aggregate_rmse_mm"])
        for scenario, values in frozen["results"].items():
            writer.writerow([scenario, "projected_ridge_base", values["projected_ridge_base"]["mean"]["rmse_mm"]])
            writer.writerow([scenario, "sac", values["sac_aggregate"]["mean"]["rmse_mm"]])
            writer.writerow([scenario, "td3", values["td3_aggregate"]["mean"]["rmse_mm"]])
    shutil.copyfile(OUTPUTS / "sequence_rl" / "validation_learning_curves.png", REPORTS / "sequence_rl_validation_curves.png")
    shutil.copyfile(OUTPUTS / "frozen_rl_test" / "frozen_test_comparison.png", REPORTS / "frozen_rl_comparison.png")
    print("Wrote compact REAL_STATIC/SIM_CALIBRATED figures and CSVs to", REPORTS)


if __name__ == "__main__":
    main()
