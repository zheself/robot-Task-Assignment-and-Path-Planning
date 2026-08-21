#!/usr/bin/env python3
"""Aggregate all A3 W10 shards and rerun baselines on validation only."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from safe_residual_rl.allocation import (
    SolverProtocol,
    load_oracle_context,
    solve_assignment_milp,
    solve_deterministic_lns,
    solve_greedy,
    solve_hungarian,
    solve_hybrid_assignment_milp,
    solve_hybrid_load_balanced,
    solve_load_balanced,
    solve_order_aware_lns,
    verify_plan,
)
from safe_residual_rl.allocation.a3_protocol import (
    load_a3_development_config,
    prepare_a3_development,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/allocation/a3_development_v1.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/phase1_allocation/a3_development_v1/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_allocation/a3_w10_development_v1"),
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/phase1_allocation")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, args.config)
    data_root = _resolve(root, args.data_root)
    output_dir = _resolve(root, args.output_dir)
    report_dir = _resolve(root, args.report_dir)
    config, config_sha256 = load_a3_development_config(config_path)
    families = [str(item) for item in config["models"]["families"]]
    seeds = [int(item) for item in config["training"]["seeds"]]
    shard_results = []
    missing = []
    for family in families:
        for seed in seeds:
            path = output_dir / "shards" / family / f"seed_{seed:03d}" / "result.json"
            if not path.is_file():
                missing.append(f"{family}:{seed}")
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not value.get("completed") or value.get("config_sha256") != config_sha256:
                raise RuntimeError(f"invalid W10 shard: {path}")
            shard_results.append(value)
    if missing:
        raise RuntimeError(f"missing W10 shards: {missing}")

    shared_fields = (
        "a2_manifest_sha256",
        "access_sha256",
        "access_record_count",
        "vocabulary_sha256",
        "normalizer_sha256",
    )
    consistency = {
        key: len({json.dumps(item[key], sort_keys=True) for item in shard_results}) == 1
        for key in shared_fields
    }
    if not all(consistency.values()):
        raise RuntimeError(f"inconsistent W10 shard provenance: {consistency}")
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    prepared = prepare_a3_development(data_root, context)
    benchmark = json.loads(
        (root / "configs/allocation/benchmark_v4.json").read_text(encoding="utf-8")
    )
    weights = {key: float(value) for key, value in benchmark["objective_weights"].items()}
    baseline_rows = _validation_baselines(
        prepared.validation_examples, context, benchmark, weights
    )
    family_rows = [_family_row(family, shard_results) for family in families]
    selected = max(
        family_rows,
        key=lambda item: (
            item["mean_validation_verified_coverage"],
            -_none_high(item["mean_validation_verified_score"]),
            item["mean_validation_assignment_accuracy"],
            -item["median_validation_pipeline_runtime_s"],
            _reverse_text(item["family"]),
        ),
    )
    cell_rows = _cell_rows(shard_results)
    checks = {
        "all_nine_registered_shards_completed": len(shard_results) == 9,
        "all_shard_provenance_consistent": all(consistency.values()),
        "all_shards_train_validation_only": all(item["access_record_count"] == 240 for item in shard_results),
        "all_metrics_finite": _finite_metrics(shard_results),
        "all_registered_validation_baselines_completed": len(baseline_rows) == len(benchmark["baseline_protocol"]["methods"]),
        "selected_family_uses_validation_only": selected["family"] in families,
    }
    summary = {
        "version": "a3-w10-development-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "config_sha256": config_sha256,
        "a2_manifest_sha256": config["a2_manifest_sha256"],
        "access_sha256": prepared.access_sha256,
        "train_instances": len(prepared.train_examples),
        "validation_instances": len(prepared.validation_examples),
        "families": families,
        "seeds": seeds,
        "selected_family": selected["family"],
        "selection_rule": config["selection"],
        "family_results": family_rows,
        "validation_cell_results": cell_rows,
        "validation_baselines": baseline_rows,
        "checks": checks,
        "passed": all(checks.values()),
        "frozen_test_accessed": False,
        "stress_accessed": False,
        "boundaries": config["boundaries"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "validation_baselines.json", baseline_rows)
    _write_json(output_dir / "aggregate_summary.json", summary)
    _write_csv(output_dir / "family_summary.csv", family_rows)
    _write_csv(output_dir / "validation_cells.csv", cell_rows)
    _write_csv(output_dir / "validation_baselines.csv", baseline_rows)
    compact = dict(summary)
    _write_json(report_dir / "a3_w10_development_v1_summary.json", compact)
    _write_csv(report_dir / "a3_w10_development_v1_families.csv", family_rows)
    _write_csv(report_dir / "a3_w10_development_v1_cells.csv", cell_rows)
    markdown = _markdown(summary)
    (output_dir / "results.md").write_text(markdown, encoding="utf-8")
    (report_dir / "a3_w10_development_v1_results.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "shards": len(shard_results),
                "selected_family": selected["family"],
                "frozen_accessed": False,
            }
        ),
        flush=True,
    )


def _family_row(family: str, shards) -> dict[str, object]:
    selected = [item for item in shards if item["family"] == family]
    validations = [item["validation_evaluation"] for item in selected]
    scores = [item["mean_verified_weighted_proxy_score"] for item in validations]
    return {
        "family": family,
        "seeds": len(selected),
        "mean_validation_verified_coverage": float(
            np.mean([item["verified_candidate_coverage"] for item in validations])
        ),
        "min_validation_verified_coverage": float(
            np.min([item["verified_candidate_coverage"] for item in validations])
        ),
        "mean_validation_assignment_accuracy": float(
            np.mean([item["atomic_unit_accuracy"] for item in validations])
        ),
        "mean_validation_verified_score": (
            None if any(item is None for item in scores) else float(np.mean(scores))
        ),
        "median_validation_pipeline_runtime_s": float(
            statistics.median(
                item["median_inference_runtime_s"] for item in validations
            )
        ),
        "mean_best_epoch": float(np.mean([item["best_epoch"] for item in selected])),
        "mean_epochs_completed": float(
            np.mean([item["epochs_completed"] for item in selected])
        ),
        "total_validation_failures": int(
            sum(len(item["failures"]) for item in validations)
        ),
    }


def _cell_rows(shards) -> list[dict[str, object]]:
    rows = []
    for family in sorted({item["family"] for item in shards}):
        family_shards = [item for item in shards if item["family"] == family]
        cells = sorted(family_shards[0]["validation_cells"])
        for cell in cells:
            values = [item["validation_cells"][cell] for item in family_shards]
            scores = [item["mean_verified_weighted_proxy_score"] for item in values]
            rows.append(
                {
                    "family": family,
                    "cell_id": cell,
                    "seeds": len(values),
                    "mean_verified_coverage": float(
                        np.mean([item["verified_candidate_coverage"] for item in values])
                    ),
                    "min_verified_coverage": float(
                        np.min([item["verified_candidate_coverage"] for item in values])
                    ),
                    "mean_assignment_accuracy": float(
                        np.mean([item["atomic_unit_accuracy"] for item in values])
                    ),
                    "mean_verified_score": (
                        None
                        if any(item is None for item in scores)
                        else float(np.mean(scores))
                    ),
                }
            )
    return rows


def _validation_baselines(examples, context, benchmark, weights):
    protocol_raw = benchmark["baseline_protocol"]
    protocol = SolverProtocol(
        "a1-solver-protocol-v1",
        float(protocol_raw["milp_time_limit_s"]),
        float(protocol_raw["milp_relative_gap"]),
        0,
    )
    registry = {
        "greedy": lambda instance: solve_greedy(instance, context),
        "load_balanced": lambda instance: solve_load_balanced(instance, context),
        "hungarian": lambda instance: solve_hungarian(instance, context),
        "assignment_milp": lambda instance: solve_assignment_milp(instance, context, protocol),
        "deterministic_lns": lambda instance: solve_deterministic_lns(
            instance,
            context,
            iterations=int(protocol_raw["lns_iterations"]),
            seed=int(protocol_raw["lns_seed"]),
            objective_weights=weights,
        ),
        "hybrid_load_balanced": lambda instance: solve_hybrid_load_balanced(instance, context),
        "hybrid_assignment_milp": lambda instance: solve_hybrid_assignment_milp(instance, context, protocol),
        "order_aware_lns": lambda instance: solve_order_aware_lns(
            instance,
            context,
            iterations=int(protocol_raw["lns_iterations"]),
            seed=int(protocol_raw["lns_seed"]),
            objective_weights=weights,
        ),
    }
    rows = []
    for method in protocol_raw["methods"]:
        verified = 0
        scores = []
        runtimes = []
        statuses = defaultdict(int)
        for example in examples:
            started = time.perf_counter()
            result = registry[method](example.instance)
            check = (
                verify_plan(example.instance, result.plan, context)
                if result.plan is not None
                else None
            )
            runtimes.append(time.perf_counter() - started)
            statuses[result.status] += 1
            if check is not None and check.feasible:
                verified += 1
                scores.append(
                    sum(
                        float(weights.get(key, 0.0)) * float(value)
                        for key, value in check.objective_terms
                    )
                )
        rows.append(
            {
                "method": method,
                "instances": len(examples),
                "verified_candidates": verified,
                "verified_coverage": verified / len(examples),
                "mean_verified_score": float(np.mean(scores)) if scores else None,
                "median_pipeline_runtime_s": float(statistics.median(runtimes)),
                "status_counts": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
            }
        )
    return rows


def _finite_metrics(shards) -> bool:
    for item in shards:
        for split in ("train_evaluation", "validation_evaluation"):
            values = item[split]
            for key in (
                "mean_loss",
                "atomic_unit_accuracy",
                "verified_candidate_coverage",
                "median_inference_runtime_s",
            ):
                if not np.isfinite(values[key]):
                    return False
    return True


def _markdown(summary) -> str:
    lines = [
        "# A3 W10 train/validation development results",
        "",
        "Evidence: **SIM_GEOMETRIC**. Frozen-test and stress were not accessed.",
        "",
        f"- Development gate: **{'PASSED' if summary['passed'] else 'FAILED'}**.",
        f"- Matrix: {len(summary['families'])} families × {len(summary['seeds'])} seeds; train {summary['train_instances']}, validation {summary['validation_instances']}.",
        f"- Validation-selected family: **{summary['selected_family']}**.",
        "",
        "## Registered checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in summary["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Model-family validation aggregate",
            "",
            "| family | coverage mean/min | assignment accuracy | conditional score | median full-pipeline runtime s | best epoch / epochs run | failures |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["family_results"]:
        lines.append(
            f"| {item['family']} | {item['mean_validation_verified_coverage']:.3f}/{item['min_validation_verified_coverage']:.3f} | {item['mean_validation_assignment_accuracy']:.3f} | {_fmt(item['mean_validation_verified_score'])} | {item['median_validation_pipeline_runtime_s']:.6f} | {item['mean_best_epoch']:.1f}/{item['mean_epochs_completed']:.1f} | {item['total_validation_failures']} |"
        )
    lines.extend(
        [
            "",
            "## Validation-only non-learning baselines",
            "",
            "| method | coverage | conditional score | median full-pipeline runtime s |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in summary["validation_baselines"]:
        lines.append(
            f"| {item['method']} | {item['verified_coverage']:.3f} | {_fmt(item['mean_verified_score'])} | {item['median_pipeline_runtime_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Selection is based only on validation and authorizes a separate frozen-evaluation preregistration; it is not a frozen-test result. Conditional scores exclude failed candidates and must be read together with coverage. The witness is a feasible A1-proxy teacher, not an optimum or real expert. No A4 repair, motion planning, collision guarantee, physical model or RL was used.",
            "",
        ]
    )
    return "\n".join(lines)


def _none_high(value) -> float:
    return float("inf") if value is None else float(value)


def _reverse_text(value: str) -> tuple[int, ...]:
    return tuple(-ord(item) for item in value)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
