#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "pre_advisor_smoke" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "pre_advisor")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("evidence_level") != "synthetic_for_pipeline_validation_only":
        raise ValueError("this plotting entry point accepts only the synthetic smoke schema")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(report["evaluation"])
    methods = list(next(iter(report["evaluation"].values())))
    rows = []
    for scenario in scenarios:
        for method in methods:
            mean = report["evaluation"][scenario][method]["mean"]
            rows.append({"scenario": scenario, "method": method, **mean})
    csv_path = args.output_dir / "synthetic_method_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(scenarios))
    width = 0.82 / len(methods)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(methods)))
    for method_index, (method, color) in enumerate(zip(methods, colors)):
        offset = (method_index - (len(methods) - 1) / 2.0) * width
        rmse = [report["evaluation"][scenario][method]["mean"]["rmse_mm"] for scenario in scenarios]
        clips = [report["evaluation"][scenario][method]["mean"]["safety_clip_rate"] for scenario in scenarios]
        axes[0].bar(x + offset, rmse, width, label=method, color=color)
        axes[1].bar(x + offset, clips, width, color=color)
    axes[0].set_ylabel("Trajectory RMSE (mm)")
    axes[0].set_title("SYNTHETIC pipeline-validation comparison (not real-robot evidence)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("Safety clip rate")
    axes[1].set_xticks(x, [name.replace("_", "\n") for name in scenarios])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(bottom=0.0)
    fig.text(0.5, 0.002, "ILC uses trajectory-specific repeated execution; other learned policies are zero-shot on each test path.", ha="center", fontsize=9)
    figure_path = args.output_dir / "synthetic_method_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {csv_path}\nWrote {figure_path}")


if __name__ == "__main__":
    main()
