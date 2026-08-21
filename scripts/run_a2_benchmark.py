#!/usr/bin/env python3
"""Materialise and evaluate the frozen A2 SIM_GEOMETRIC benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from safe_residual_rl.allocation.benchmark import (
    audit_split_leakage,
    load_manifest,
    materialize_benchmark,
    verify_materialized_instances,
    write_manifest,
)
from safe_residual_rl.allocation.generation import load_benchmark_config
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.solvers import (
    SolverProtocol,
    solve_assignment_milp,
    solve_deterministic_lns,
    solve_greedy,
    solve_hungarian,
    solve_load_balanced,
)
from safe_residual_rl.allocation.verifier import verify_plan

SUCCESS = {"feasible", "optimal", "feasible_limit"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/allocation/benchmark_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_allocation/a2_benchmark"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/allocation/a2_benchmark_manifest_v1.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase1_allocation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, args.config)
    output_dir = _resolve(root, args.output_dir)
    manifest_path = _resolve(root, args.manifest)
    report_dir = _resolve(root, args.report_dir)
    config = load_benchmark_config(config_path)
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    manifest, generated = materialize_benchmark(config, root, output_dir.relative_to(root) / "instances")
    if manifest_path.exists():
        if load_manifest(manifest_path) != manifest:
            raise RuntimeError("frozen A2 manifest differs; create a new version instead of overwriting")
    else:
        write_manifest(manifest, manifest_path)
    leakage = audit_split_leakage(manifest.records)
    materialization_failures = verify_materialized_instances(manifest, root)
    if leakage or materialization_failures:
        raise RuntimeError(f"benchmark audit failed: leakage={leakage}, files={materialization_failures}")

    protocol_raw = config.baseline_protocol
    milp_protocol = SolverProtocol(
        "a1-solver-protocol-v1",
        float(protocol_raw["milp_time_limit_s"]),
        float(protocol_raw["milp_relative_gap"]),
        0,
    )
    weights = dict(config.objective_weights)
    methods = (
        ("greedy", lambda instance: solve_greedy(instance, context)),
        ("load_balanced", lambda instance: solve_load_balanced(instance, context)),
        ("hungarian", lambda instance: solve_hungarian(instance, context)),
        ("assignment_milp", lambda instance: solve_assignment_milp(instance, context, milp_protocol)),
        (
            "deterministic_lns",
            lambda instance: solve_deterministic_lns(
                instance,
                context,
                iterations=int(protocol_raw["lns_iterations"]),
                seed=int(protocol_raw["lns_seed"]),
                objective_weights=weights,
            ),
        ),
    )
    rows: list[dict[str, object]] = []
    raw_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    labels: dict[str, list[dict[str, object]]] = {name: [] for name in ("train", "validation", "frozen_test", "stress")}
    records = {item.instance_id: item for item in manifest.records}
    for generated_item in generated:
        instance = generated_item.instance
        instance_results = []
        for method_name, solve in methods:
            result = solve(instance)
            verification = verify_plan(instance, result.plan, context) if result.plan is not None else None
            verified = bool(verification and verification.feasible)
            objectives = dict(verification.objective_terms) if verification else {}
            score = _score(objectives, weights) if verified else None
            row = {
                "instance_id": instance.instance_id,
                "split": generated_item.split,
                "family": generated_item.family,
                "task_group_id": generated_item.task_group_id,
                "seed": generated_item.seed,
                "robot_count": len(instance.robots),
                "segment_count": len(instance.segments),
                "method": method_name,
                "method_id": result.method_id,
                "status": result.status,
                "verified": verified,
                "runtime_s": result.runtime_s,
                "weighted_proxy_score": score,
                "makespan_s": objectives.get("makespan"),
                "load_variance_s2": objectives.get("load_variance"),
                "travel_setup_time_s": objectives.get("travel_setup_time"),
                "priority_tardiness": objectives.get("priority_tardiness"),
                "assignment_mip_gap": result.mip_gap,
                "assignment_best_bound": result.best_bound,
                "violation_codes": ";".join(sorted({item.code for item in verification.violations})) if verification else "",
            }
            rows.append(row)
            raw_results.append({"instance_id": instance.instance_id, "split": generated_item.split, "result": result.to_dict()})
            instance_results.append((row, result))
            if result.status not in SUCCESS or not verified:
                failures.append({
                    "instance_id": instance.instance_id,
                    "split": generated_item.split,
                    "method": method_name,
                    "status": result.status,
                    "verified": verified,
                    "diagnostics": list(result.diagnostics),
                    "violation_codes": row["violation_codes"],
                })
        feasible = [(row, result) for row, result in instance_results if row["verified"] and row["weighted_proxy_score"] is not None]
        if feasible:
            best_row, best_result = min(feasible, key=lambda pair: (float(pair[0]["weighted_proxy_score"]), float(pair[0]["runtime_s"]), str(pair[0]["method"])))
            labels[generated_item.split].append({
                "instance_id": instance.instance_id,
                "task_group_id": generated_item.task_group_id,
                "source_method": best_row["method"],
                "source_status": best_row["status"],
                "weighted_proxy_score": best_row["weighted_proxy_score"],
                "usage": {"train": "fit_only", "validation": "selection_only", "frozen_test": "evaluation_only", "stress": "evaluation_only"}[generated_item.split],
                "plan": best_result.plan.to_dict(),
            })
        else:
            labels[generated_item.split].append({"instance_id": instance.instance_id, "task_group_id": generated_item.task_group_id, "source_method": None, "usage": "failure_only", "plan": None})

    for row in rows:
        peer_scores = [float(item["weighted_proxy_score"]) for item in rows if item["instance_id"] == row["instance_id"] and item["weighted_proxy_score"] is not None]
        score = row["weighted_proxy_score"]
        row["best_observed_relative_gap"] = None if score is None or not peer_scores else (float(score) - min(peer_scores)) / max(abs(min(peer_scores)), 1e-12)
    aggregates = _aggregate(rows)
    run_record = {
        "schema_version": "a2-baseline-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "manifest_sha256": manifest.manifest_sha256,
        "config_sha256": manifest.config_sha256,
        "master_seed": config.master_seed,
        "baseline_protocol": dict(config.baseline_protocol),
        "objective_weights": weights,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "boundaries": list(config.boundaries) + ["best_observed_relative_gap is not a certified optimality gap."],
        "rows": rows,
        "aggregates": aggregates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "baseline_results.json", run_record)
    _write_csv(output_dir / "baseline_results.csv", rows)
    _write_json(output_dir / "failure_library.json", {"manifest_sha256": manifest.manifest_sha256, "failures": failures})
    _write_json(output_dir / "candidate_solver_labels.json", {"manifest_sha256": manifest.manifest_sha256, "split_usage_guard": {"train": "fit", "validation": "selection", "frozen_test": "evaluation_only", "stress": "evaluation_only"}, "labels": labels})
    _write_json(output_dir / "raw_solver_results.json", raw_results)
    _write_json(report_dir / "a2_baseline_summary.json", {"manifest_sha256": manifest.manifest_sha256, "aggregates": aggregates, "failure_count": len(failures)})
    _write_csv(report_dir / "a2_baseline_summary.csv", aggregates)
    markdown = _markdown(manifest, rows, aggregates, failures, labels)
    (output_dir / "baseline_results.md").write_text(markdown, encoding="utf-8")
    (report_dir / "a2_benchmark_results.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"instances": len(generated), "runs": len(rows), "manifest_sha256": manifest.manifest_sha256, "failures": len(failures)}))


def _aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["split"], row["method"]), []).append(row)
    result = []
    for (split, method), values in sorted(grouped.items()):
        verified = [item for item in values if item["verified"]]
        result.append({
            "split": split,
            "method": method,
            "instances": len(values),
            "verified_count": len(verified),
            "verified_rate": len(verified) / len(values),
            "median_runtime_s": statistics.median(float(item["runtime_s"]) for item in values),
            "mean_makespan_s": _mean(item["makespan_s"] for item in verified),
            "mean_load_variance_s2": _mean(item["load_variance_s2"] for item in verified),
            "mean_travel_setup_time_s": _mean(item["travel_setup_time_s"] for item in verified),
        })
    return result


def _markdown(manifest, rows, aggregates, failures, labels):
    failure_statuses = Counter(str(item["status"]) for item in failures)
    lines = [
        "# A2 leakage-safe geometric benchmark pilot results",
        "",
        "Evidence: **SIM_GEOMETRIC**. This frozen engineering pilot contains programmatic continuous curves and proxy workcells, not real factory trajectories or a paper-scale training corpus.",
        "",
        f"- Manifest SHA-256: `{manifest.manifest_sha256}`",
        f"- Instances: {len(manifest.records)}; solver runs: {len(rows)}; recorded failed/unverified runs: {len(failures)}.",
        "- Frozen-test and stress labels are marked `evaluation_only`; they cannot be used for fitting or model selection.",
        "- `assignment_mip_gap` certifies only the assignment MILP formulation. `best_observed_relative_gap` is descriptive, not a certified scheduling optimality gap.",
        "- Failure status counts: " + ", ".join(f"`{key}`={value}" for key, value in sorted(failure_statuses.items())) + ".",
        "",
        "| split | method | verified | median runtime (s) | mean makespan (s) | mean load variance (s²) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(f"| {item['split']} | {item['method']} | {item['verified_count']}/{item['instances']} | {item['median_runtime_s']:.6f} | {_fmt(item['mean_makespan_s'])} | {_fmt(item['mean_load_variance_s2'])} |")
    lines.extend([
        "",
        "## Candidate label availability",
        "",
    ])
    for split, values in labels.items():
        lines.append(f"- {split}: {sum(item['plan'] is not None for item in values)}/{len(values)} instances have a verified candidate; usage is `{values[0]['usage'] if values else 'n/a'}`.")
    lines.extend([
        "",
        "## Boundaries and next gate",
        "",
        "Pilot v1 validates A2 plumbing and freezes its own evidence. It does not establish GNN superiority, real collision safety, real cycle time, path executability, or physical quality improvement. A3 remains blocked until a new paper-scale A2 manifest with substantially more independent groups is preregistered and evaluated.",
        "",
        "## Reproduction",
        "",
        "`PYTHONPATH=src .venv/bin/python scripts/run_a2_benchmark.py`",
        "",
    ])
    return "\n".join(lines)


def _score(objectives, weights):
    return sum(float(weights.get(key, 0.0)) * float(value) for key, value in objectives.items())


def _mean(values):
    selected = [float(value) for value in values if value is not None]
    return None if not selected else statistics.mean(selected)


def _fmt(value):
    return "n/a" if value is None else f"{float(value):.6f}"


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
