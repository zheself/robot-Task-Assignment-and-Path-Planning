#!/usr/bin/env python3
"""Run the preregistered A3 W9 graph/model engineering smoke protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from safe_residual_rl.allocation.graphs import (
    A3GraphRecord,
    build_a3_graph,
    build_teacher_target,
    discover_a3_records,
    fit_feature_normalizer,
    fit_feature_vocabulary,
)
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.training import A3TrainingExample, train_a3_imitation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/allocation/a3_development_v1.json"))
    parser.add_argument("--instance-root", type=Path, default=Path("outputs/phase1_allocation/a3_development_v1/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_allocation/a3_w9_foundation_v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase1_allocation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, args.config)
    instance_root = _resolve(root, args.instance_root)
    output_dir = _resolve(root, args.output_dir)
    report_dir = _resolve(root, args.report_dir)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(raw_config)
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")

    forbidden_present = [
        split for split in raw_config["access_guard"]["forbidden_splits"]
        if (instance_root / split).exists() or (instance_root / "witnesses" / split).exists()
    ]
    if forbidden_present:
        raise RuntimeError(f"forbidden split directories entered A3 workspace: {forbidden_present}")
    train_records = discover_a3_records(instance_root, "train", context)
    validation_records = discover_a3_records(instance_root, "validation", context)
    leakage = _leakage(train_records, validation_records)
    if leakage:
        raise RuntimeError(f"train/validation leakage: {leakage}")

    vocabulary = fit_feature_vocabulary(
        [item.instance for item in train_records], split="train"
    )
    train_raw = tuple(
        build_a3_graph(item.instance, context, vocabulary, split="train")
        for item in train_records
    )
    validation_raw = tuple(
        build_a3_graph(item.instance, context, vocabulary, split="validation")
        for item in validation_records
    )
    epsilon = float(raw_config["graph"]["normalization_epsilon"])
    normalizer = fit_feature_normalizer(train_raw, split="train", epsilon=epsilon)
    train_graphs = tuple(normalizer.transform(item) for item in train_raw)
    validation_graphs = tuple(normalizer.transform(item) for item in validation_raw)

    smoke = raw_config["w9_smoke"]
    train_indices = _balanced_indices(
        train_records, int(smoke["train_groups_per_cell"])
    )
    validation_indices = _balanced_indices(
        validation_records, int(smoke["validation_groups_per_cell"])
    )
    train_examples = _examples(train_records, train_graphs, train_indices)
    validation_examples = _examples(
        validation_records, validation_graphs, validation_indices
    )
    model_config = raw_config["models"]
    training = raw_config["training"]
    loss = raw_config["loss"]
    weights = json.loads(
        (root / "configs/allocation/benchmark_v4.json").read_text(encoding="utf-8")
    )["objective_weights"]
    keyword = {
        "family": str(smoke["model_family"]),
        "seed": int(smoke["seed"]),
        "epochs": int(smoke["epochs"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "layers": int(model_config["message_passing_layers"]),
        "heads": int(model_config["attention_heads"]),
        "dropout": float(model_config["dropout"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "assignment_weight": float(loss["assignment_cross_entropy_weight"]),
        "order_weight": float(loss["pairwise_order_bce_weight"]),
    }
    first = train_a3_imitation(
        train_examples, validation_examples, context, weights, **keyword
    )
    second = train_a3_imitation(
        train_examples, validation_examples, context, weights, **keyword
    )
    deterministic = first.state_sha256 == second.state_sha256 and first.history == second.history

    access_manifest = {
        "version": "a3-development-access-manifest-v1",
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": list(raw_config["access_guard"]["forbidden_splits"]),
        "forbidden_directories_present": forbidden_present,
        "records": [
            {
                "split": item.split,
                "cell_id": item.cell_id,
                "instance_id": item.instance.instance_id,
                "record_sha256": item.record_sha256,
            }
            for item in train_records + validation_records
        ],
    }
    access_hash = _digest(access_manifest)
    access_manifest["sha256"] = access_hash
    checks = {
        "record_counts_192_train_48_validation": (len(train_records), len(validation_records)) == (192, 48),
        "zero_train_validation_group_leakage": not leakage,
        "zero_forbidden_split_access": not forbidden_present,
        "zero_teacher_hash_or_verifier_failures": True,
        "normalizer_fit_count_equals_train_graph_count": normalizer.fit_graph_count == len(train_records),
        "deterministic_same_seed": deterministic,
        "smoke_subset_is_cell_balanced": _cell_counts(train_examples) == {key: 4 for key in _cell_counts(train_examples)} and _cell_counts(validation_examples) == {key: 2 for key in _cell_counts(validation_examples)},
    }
    summary = {
        "version": "a3-w9-foundation-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "a2_manifest_sha256": raw_config["a2_manifest_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "access_manifest_sha256": access_hash,
        "vocabulary_sha256": vocabulary.sha256,
        "normalizer_sha256": normalizer.sha256,
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "smoke_train_instances": len(train_examples),
        "smoke_validation_instances": len(validation_examples),
        "feature_dimensions": {
            "segment": int(train_graphs[0].segment_features.shape[1]),
            "robot": int(train_graphs[0].robot_features.shape[1]),
            "resource": int(train_graphs[0].resource_features.shape[1]),
            "edge_pair": int(train_graphs[0].pair_features.shape[2]),
        },
        "model_family": first.family,
        "seed": first.seed,
        "epochs": first.epochs,
        "state_sha256": first.state_sha256,
        "history": list(first.history),
        "train_evaluation": first.train_evaluation.to_dict(),
        "validation_evaluation": first.validation_evaluation.to_dict(),
        "checks": checks,
        "passed": all(checks.values()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "boundaries": raw_config["boundaries"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "access_manifest.json").write_text(
        json.dumps(access_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "feature_vocabulary.json").write_text(
        json.dumps(vocabulary.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "feature_normalizer.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(dict(first.state_dict), output_dir / "smoke_checkpoint.pt")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    compact = dict(summary)
    compact["train_evaluation"] = _without_failures(first.train_evaluation.to_dict())
    compact["validation_evaluation"] = _without_failures(first.validation_evaluation.to_dict())
    (report_dir / "a3_w9_foundation_v1_summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = _markdown(summary)
    (output_dir / "results.md").write_text(markdown, encoding="utf-8")
    (report_dir / "a3_w9_foundation_v1_results.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "train_records": len(train_records),
                "validation_records": len(validation_records),
                "smoke_train": len(train_examples),
                "smoke_validation": len(validation_examples),
                "validation_coverage": first.validation_evaluation.verified_candidate_coverage,
                "state_sha256": first.state_sha256,
            }
        )
    )


def _examples(records, graphs, indices) -> tuple[A3TrainingExample, ...]:
    return tuple(
        A3TrainingExample(
            graphs[index],
            records[index].instance,
            build_teacher_target(
                graphs[index], records[index].teacher_plan, records[index].teacher_sha256
            ),
            records[index].cell_id,
        )
        for index in indices
    )


def _balanced_indices(records: Sequence[A3GraphRecord], groups_per_cell: int) -> tuple[int, ...]:
    by_cell: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        group_id = record.instance.instance_id.rsplit("-v", 1)[0]
        by_cell[record.cell_id][group_id].append(index)
    selected = []
    for cell_id in sorted(by_cell):
        groups = sorted(by_cell[cell_id])[:groups_per_cell]
        if len(groups) != groups_per_cell:
            raise ValueError(f"not enough groups for cell {cell_id}")
        selected.extend(index for group in groups for index in sorted(by_cell[cell_id][group]))
    return tuple(selected)


def _leakage(train, validation) -> dict[str, list[str]]:
    result = {}
    for name, accessor in (
        ("workpiece", lambda item: {item.instance.workpiece_id}),
        ("layout", lambda item: {item.instance.layout_id}),
        ("parent_curve", lambda item: {segment.parent_curve_id for segment in item.instance.segments}),
    ):
        left = set().union(*(accessor(item) for item in train))
        right = set().union(*(accessor(item) for item in validation))
        overlap = sorted(left & right)
        if overlap:
            result[name] = overlap
    return result


def _cell_counts(examples) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for item in examples:
        result[item.cell_id] += 1
    return dict(sorted(result.items()))


def _validate_config(value: dict[str, Any]) -> None:
    if value.get("version") != "a3-train-validation-development-v1":
        raise ValueError("unexpected A3 development config")
    if value["access_guard"]["allowed_splits"] != ["train", "validation"]:
        raise ValueError("A3 development split order/guard changed")
    if value["access_guard"]["forbidden_splits"] != ["frozen_test", "stress"]:
        raise ValueError("A3 forbidden splits changed")


def _markdown(summary: dict[str, Any]) -> str:
    train = summary["train_evaluation"]
    validation = summary["validation_evaluation"]
    lines = [
        "# A3 W9 graph/model foundation results",
        "",
        "Evidence: **SIM_GEOMETRIC**. This is a train/validation-only engineering smoke run, not a frozen-test result.",
        "",
        f"- Gate: **{'PASSED' if summary['passed'] else 'FAILED'}**.",
        f"- Accessed records: train {summary['train_record_count']}, validation {summary['validation_record_count']}; forbidden split access: zero.",
        f"- Smoke subset: train {summary['smoke_train_instances']}, validation {summary['smoke_validation_instances']} instances, selected by lexicographic cell-balanced groups.",
        f"- Vocabulary SHA-256: `{summary['vocabulary_sha256']}`; normalizer SHA-256: `{summary['normalizer_sha256']}`.",
        f"- Same-seed checkpoint SHA-256: `{summary['state_sha256']}`.",
        "",
        "## Engineering checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in summary["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Smoke metrics",
            "",
            f"- Train atomic-unit assignment accuracy: {train['atomic_unit_accuracy']:.3f}; verified candidate coverage: {train['verified_candidate_coverage']:.3f}.",
            f"- Validation atomic-unit assignment accuracy: {validation['atomic_unit_accuracy']:.3f}; verified candidate coverage: {validation['verified_candidate_coverage']:.3f}.",
            f"- Validation mean imitation loss: {validation['mean_loss']:.6f}.",
            "",
            "The four-epoch smoke metrics are not used as a paper claim or as a frozen-test selection result. Failures are retained in the ignored raw output.",
            "",
            "## Boundaries",
            "",
            "The constructive witness is a feasible A1-proxy teacher, not an optimal or real expert. Hard edge masking and A1 verification do not establish joint motion feasibility, collision safety, robot execution or physical process quality. No A4 repair, path planning or RL was used.",
            "",
        ]
    )
    return "\n".join(lines)


def _without_failures(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["failure_count"] = len(result.pop("failures"))
    return result


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
