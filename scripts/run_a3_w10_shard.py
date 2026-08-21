#!/usr/bin/env python3
"""Run one preregistered A3 W10 model-family/seed shard."""

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

from safe_residual_rl.allocation.a3_protocol import (
    load_a3_development_config,
    prepare_a3_development,
)
from safe_residual_rl.allocation.models import A3AllocationModel
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.training import (
    evaluate_a3_by_cell,
    train_a3_with_validation_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/allocation/a3_development_v1.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/phase1_allocation/a3_development_v1/data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/phase1_allocation/a3_w10_development_v1/shards"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, args.config)
    data_root = _resolve(root, args.data_root)
    output_root = _resolve(root, args.output_root)
    config, config_sha256 = load_a3_development_config(config_path)
    families = [str(item) for item in config["models"]["families"]]
    seeds = [int(item) for item in config["training"]["seeds"]]
    if args.family not in families or args.seed not in seeds:
        raise ValueError("family/seed is outside the preregistered W10 matrix")
    if config["training"]["device"] != "cpu":
        raise ValueError("W10 v1 is preregistered for CPU")
    destination = output_root / args.family / f"seed_{args.seed:03d}"
    result_path = destination / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_sha256") == config_sha256
            and existing.get("family") == args.family
            and existing.get("seed") == args.seed
            and existing.get("completed") is True
        ):
            print(json.dumps({"status": "reused", "family": args.family, "seed": args.seed}))
            return

    started = time.perf_counter()
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    prepared = prepare_a3_development(data_root, context)
    w9 = json.loads(
        (root / "reports/phase1_allocation/a3_w9_foundation_v1_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if prepared.vocabulary.sha256 != w9["vocabulary_sha256"]:
        raise RuntimeError("W10 vocabulary differs from frozen W9 preprocessing")
    if prepared.normalizer.sha256 != w9["normalizer_sha256"]:
        raise RuntimeError("W10 normalizer differs from frozen W9 preprocessing")
    benchmark = json.loads(
        (root / "configs/allocation/benchmark_v4.json").read_text(encoding="utf-8")
    )
    weights = {key: float(value) for key, value in benchmark["objective_weights"].items()}
    model_config = config["models"]
    training = config["training"]
    loss = config["loss"]
    run = train_a3_with_validation_selection(
        prepared.train_examples,
        prepared.validation_examples,
        context,
        weights,
        family=args.family,
        seed=args.seed,
        max_epochs=int(training["max_epochs"]),
        patience=int(training["early_stopping_patience"]),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["message_passing_layers"]),
        heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        assignment_weight=float(loss["assignment_cross_entropy_weight"]),
        order_weight=float(loss["pairwise_order_bce_weight"]),
    )
    model = A3AllocationModel(
        prepared.train_examples[0].graph,
        family=args.family,
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["message_passing_layers"]),
        heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(run.state_dict)
    validation_cells = evaluate_a3_by_cell(
        model,
        prepared.validation_examples,
        context,
        weights,
        assignment_weight=float(loss["assignment_cross_entropy_weight"]),
        order_weight=float(loss["pairwise_order_bce_weight"]),
    )
    elapsed = time.perf_counter() - started
    result = {
        "version": "a3-w10-shard-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "family": run.family,
        "seed": run.seed,
        "max_epochs": run.max_epochs,
        "epochs_completed": run.epochs_completed,
        "best_epoch": run.best_epoch,
        "stopped_early": run.stopped_early,
        "state_sha256": run.state_sha256,
        "config_sha256": config_sha256,
        "a2_manifest_sha256": config["a2_manifest_sha256"],
        "access_sha256": prepared.access_sha256,
        "access_record_count": prepared.access_record_count,
        "vocabulary_sha256": prepared.vocabulary.sha256,
        "normalizer_sha256": prepared.normalizer.sha256,
        "train_evaluation": run.train_evaluation.to_dict(),
        "validation_evaluation": run.validation_evaluation.to_dict(),
        "validation_cells": {
            key: value.to_dict() for key, value in validation_cells.items()
        },
        "training_wall_time_s": elapsed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": "cpu",
        },
        "boundaries": config["boundaries"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(dict(run.state_dict), destination / "checkpoint.pt")
    _write_history(destination / "history.csv", run.history)
    _atomic_json(destination / "failure_library.json", {
        "family": run.family,
        "seed": run.seed,
        "train": list(run.train_evaluation.failures),
        "validation": list(run.validation_evaluation.failures),
    })
    _atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "family": run.family,
                "seed": run.seed,
                "epochs": run.epochs_completed,
                "best_epoch": run.best_epoch,
                "validation_coverage": run.validation_evaluation.verified_candidate_coverage,
                "validation_accuracy": run.validation_evaluation.atomic_unit_accuracy,
                "state_sha256": run.state_sha256,
                "wall_time_s": elapsed,
            }
        ),
        flush=True,
    )


def _write_history(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
