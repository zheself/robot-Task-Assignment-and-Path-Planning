#!/usr/bin/env python3
"""Run one fixed A3.5 model-variant/seed development shard."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from safe_residual_rl.allocation.models import A3AllocationModel
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.pointer_decoder import FeasiblePairPointer
from safe_residual_rl.allocation.pointer_pilot import load_pointer_pilot_config
from safe_residual_rl.allocation.pointer_training import (
    evaluate_pointer_by_cell,
    prepare_pointer_pilot,
    train_pointer_with_validation_selection,
)
from safe_residual_rl.allocation.training import (
    evaluate_a3_by_cell,
    train_a3_with_validation_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/allocation/a3_5_pointer_pilot_v1.json"))
    parser.add_argument("--pilot-root", type=Path, default=Path("outputs/phase1_allocation/a3_5_pointer_pilot_v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_pointer_pilot_config(_resolve(root, args.config))
    variants = {str(item["id"]): item for item in config.raw["models"]["variants"]}
    seeds = [int(item) for item in config.raw["training"]["seeds"]]
    if args.variant not in variants or args.seed not in seeds:
        raise ValueError("variant/seed is outside the frozen A3.5 matrix")
    pilot_root = _resolve(root, args.pilot_root)
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    prepared = prepare_pointer_pilot(pilot_root / "data", pilot_root / "manifest.json", context, config)
    variant = variants[args.variant]
    model_cfg = config.raw["models"]
    training = config.raw["training"]
    output = pilot_root / "shards" / args.variant / f"seed_{args.seed:03d}"
    if (output / "result.json").exists():
        raise FileExistsError("A3.5 shard is non-overwriting")
    started = time.perf_counter()
    static_examples_train = [item.static for item in prepared.train_examples]
    static_examples_validation = [item.static for item in prepared.validation_examples]
    if variant["decoder"] == "static":
        run = train_a3_with_validation_selection(
            static_examples_train, static_examples_validation, context, config.objective_weights,
            family=str(variant["encoder"]), seed=args.seed,
            max_epochs=int(training["max_epochs"]), patience=int(training["early_stopping_patience"]),
            hidden_dim=int(model_cfg["hidden_dim"]), layers=int(model_cfg["message_passing_layers"]),
            heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]),
            learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            assignment_weight=float(training["static_assignment_weight"]), order_weight=float(training["static_order_weight"]),
        )
        model = A3AllocationModel(prepared.train_examples[0].graph, family=str(variant["encoder"]), hidden_dim=int(model_cfg["hidden_dim"]), layers=int(model_cfg["message_passing_layers"]), heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]))
        model.load_state_dict(run.state_dict)
        cells = evaluate_a3_by_cell(model, static_examples_validation, context, config.objective_weights, assignment_weight=float(training["static_assignment_weight"]), order_weight=float(training["static_order_weight"]))
        train_eval = _static_eval(run.train_evaluation)
        validation_eval = _static_eval(run.validation_evaluation)
        cell_eval = {key: _static_eval(value) for key, value in cells.items()}
    else:
        run = train_pointer_with_validation_selection(
            prepared.train_examples, prepared.validation_examples, context, config.objective_weights,
            variant=args.variant, encoder_family=str(variant["encoder"]), seed=args.seed,
            max_epochs=int(training["max_epochs"]), patience=int(training["early_stopping_patience"]),
            hidden_dim=int(model_cfg["hidden_dim"]), layers=int(model_cfg["message_passing_layers"]),
            heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]),
            learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        model = FeasiblePairPointer(prepared.train_examples[0].graph, encoder_family=str(variant["encoder"]), hidden_dim=int(model_cfg["hidden_dim"]), layers=int(model_cfg["message_passing_layers"]), heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]))
        model.load_state_dict(run.state_dict)
        cells = evaluate_pointer_by_cell(model, prepared.validation_examples, context, config.objective_weights)
        train_eval = run.train_evaluation.to_dict()
        validation_eval = run.validation_evaluation.to_dict()
        cell_eval = {key: value.to_dict() for key, value in cells.items()}
    elapsed = time.perf_counter() - started
    result = {
        "version": "a3-5-pointer-shard-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "development_only": True,
        "variant": args.variant,
        "encoder": str(variant["encoder"]),
        "decoder": str(variant["decoder"]),
        "seed": args.seed,
        "config_sha256": config.sha256,
        "manifest_sha256": prepared.manifest.manifest_sha256,
        "access_sha256": prepared.access_sha256,
        "accessed_splits": ["train", "validation"],
        "forbidden_splits_accessed": [],
        "v4_instance_or_witness_accessed": False,
        "vocabulary_sha256": prepared.vocabulary.sha256,
        "normalizer_sha256": prepared.normalizer.sha256,
        "max_epochs": run.max_epochs,
        "epochs_completed": run.epochs_completed,
        "best_epoch": run.best_epoch,
        "stopped_early": run.stopped_early,
        "state_sha256": run.state_sha256,
        "train_evaluation": train_eval,
        "validation_evaluation": validation_eval,
        "validation_cells": cell_eval,
        "training_wall_time_s": elapsed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"},
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(run.state_dict), output / "checkpoint.pt")
    _write_history(output / "history.csv", run.history)
    (output / "failure_library.json").write_text(json.dumps({"train": train_eval["failures"], "validation": validation_eval["failures"]}, indent=2, sort_keys=True), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"variant": args.variant, "seed": args.seed, "coverage": validation_eval["verified_candidate_coverage"], "state_sha256": run.state_sha256, "wall_time_s": elapsed}), flush=True)


def _static_eval(value) -> dict[str, object]:
    raw = value.to_dict()
    return {
        "instances": raw["instance_count"], "mean_loss": raw["mean_loss"],
        "assignment_accuracy": raw["atomic_unit_accuracy"], "teacher_forced_pair_accuracy": None,
        "complete_rollouts": raw["instance_count"], "greedy_rollout_completion_rate": 1.0,
        "verified_candidates": raw["verified_candidates"], "verified_candidate_coverage": raw["verified_candidate_coverage"],
        "hard_mask_violations": 0, "atomicity_violations": 0, "decoder_dead_ends": 0,
        "conditional_weighted_proxy_score": raw["mean_verified_weighted_proxy_score"],
        "conditional_makespan_s": None, "conditional_load_imbalance_s2": None,
        "median_inference_runtime_s": raw["median_inference_runtime_s"], "failures": list(raw["failures"]),
    }


def _write_history(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
