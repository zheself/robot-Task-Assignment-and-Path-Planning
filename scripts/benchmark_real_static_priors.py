#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.data import candidate_split_document, load_ur5_static_csvs, masks_from_candidate_split
from safe_residual_rl.evaluation import TrainingSupport, evaluate_static_prior
from safe_residual_rl.models import fit_prior, make_static_priors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "real_static_priors")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    loaded = load_ur5_static_csvs(args.root, joint_unit="degree", length_unit="mm")
    dataset = loaded.dataset
    candidate = candidate_split_document()
    masks = masks_from_candidate_split(dataset, candidate)
    train = masks["train"]
    support = TrainingSupport().fit(dataset.q_rad[train], dataset.x_nominal_m[train])
    models = make_static_priors(seed=2026)
    results = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_prior(model, dataset.q_rad[train], dataset.x_nominal_m[train], dataset.error_m[train])
        results[name] = {
            role: evaluate_static_prior(model, dataset, masks[role], support)
            for role in ("train", "validation", "test_cross_date", "external_frame_shift_diagnostic")
        }
        with (args.output_dir / "models" / f"{name}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    best_model = min(models, key=lambda name: results[name]["validation"]["overall"]["rmse_mm"])
    summary = {
        "evidence_level": "REAL_STATIC",
        "purpose": "simulator_prior_and_supervised_baseline_benchmark_not_paper_innovation",
        "metadata_status": "degree_mm_frames_tcp_and_date_semantics_unverified",
        "candidate_split": candidate,
        "split_row_counts": {role: int(mask.sum()) for role, mask in masks.items()},
        "feature_interface": "q_rad[6] + nominal_TCP_m[3] -> position_error_m[3]",
        "normalization_policy": "fit_on_candidate_training_groups_only",
        "model_seed": 2026,
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "validation_selected_model": best_model,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for model, model_result in results.items():
        for role, role_result in model_result.items():
            rows.append({"model": model, "role": role, **role_result["overall"]})
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        flat_fields = ["model", "role", "count", "rmse_mm", "mae_mm", "p95_mm", "max_mm"]
        writer = csv.DictWriter(handle, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# REAL_STATIC prior benchmark", "",
        "> Models are simulator priors/supervised baselines, not the paper's main innovation. Units, frames, TCP and date semantics remain unverified.", "",
        f"Candidate split hash: `{candidate['sha256']}`; validation-selected model: `{best_model}`.", "",
        "| Model | Validation RMSE mm | Cross-date test RMSE mm | Test P95 mm | External 20250714 RMSE mm |", "|---|---:|---:|---:|---:|",
    ]
    for name in models:
        val = results[name]["validation"]["overall"]
        test = results[name]["test_cross_date"]["overall"]
        external = results[name]["external_frame_shift_diagnostic"]["overall"]
        lines.append(f"| {name} | {val['rmse_mm']:.3f} | {test['rmse_mm']:.3f} | {test['p95_mm']:.3f} | {external['rmse_mm']:.3f} |")
    lines += ["", "The 20250714 column is an external frame/TCP-shift diagnostic and must not be pooled with the primary cross-date result.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
