#!/usr/bin/env python3
"""Run a preregistered paper-scale A2 benchmark with resumable shards."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from safe_residual_rl.allocation import (
    SolverProtocol,
    audit_paper_leakage,
    evaluate_acceptance,
    frozen_pairwise_statistics,
    load_oracle_context,
    load_paper_config,
    load_paper_manifest,
    materialize_paper_benchmark,
    method_cell_statistics,
    solve_assignment_milp,
    solve_deterministic_lns,
    solve_greedy,
    solve_hungarian,
    solve_load_balanced,
    solve_hybrid_assignment_milp,
    solve_hybrid_load_balanced,
    solve_order_aware_lns,
    verify_paper_instances,
    verify_plan,
    write_paper_manifest,
)

SUCCESS = {"feasible", "optimal", "feasible_limit"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/allocation/benchmark_v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_allocation/a2_paper_v2"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/allocation/a2_paper_manifest_v2.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase1_allocation"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, args.config)
    output_dir = _resolve(root, args.output_dir)
    manifest_path = _resolve(root, args.manifest)
    report_dir = _resolve(root, args.report_dir)
    config = load_paper_config(config_path)
    paper_version = config.version.rsplit("-", 1)[-1]
    runner_version = f"a2-paper-runner-{paper_version}"
    report_stem = f"a2_paper_{paper_version}"
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    manifest, generated = materialize_paper_benchmark(
        config, context, root, output_dir.relative_to(root) / "instances"
    )
    if manifest_path.exists():
        if load_paper_manifest(manifest_path) != manifest:
            raise RuntimeError("frozen paper manifest differs; create a new version instead of overwriting")
    else:
        write_paper_manifest(manifest, manifest_path)
    leakage = audit_paper_leakage(manifest.records)
    hash_failures = verify_paper_instances(manifest, root)
    if leakage or hash_failures:
        raise RuntimeError(f"paper benchmark audit failed: leakage={leakage}, hashes={hash_failures}")

    protocol_raw = config.baseline_protocol
    protocol = SolverProtocol(
        "a1-solver-protocol-v1",
        float(protocol_raw["milp_time_limit_s"]),
        float(protocol_raw["milp_relative_gap"]),
        0,
    )
    weights = dict(config.objective_weights)
    method_registry = {
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
    registered_methods = [str(item) for item in protocol_raw["methods"]]
    unknown = set(registered_methods) - set(method_registry)
    if unknown:
        raise RuntimeError(f"unregistered paper methods: {sorted(unknown)}")
    methods = tuple((name, method_registry[name]) for name in registered_methods)
    shard_dir = output_dir / "run_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    reused = 0
    for index, item in enumerate(generated, start=1):
        shard_path = shard_dir / f"{item.instance.instance_id}.json"
        shard = None if args.no_resume else _load_shard(shard_path, manifest.manifest_sha256, item.instance.instance_id, runner_version)
        if shard is None:
            shard = _run_instance(item, methods, context, weights, manifest.manifest_sha256, runner_version)
            _atomic_json(shard_path, shard)
        else:
            reused += 1
        rows.extend(shard["rows"])
        raw_results.extend(shard["raw_results"])
        if index % 10 == 0 or index == len(generated):
            print(json.dumps({"progress": index, "total": len(generated), "reused": reused}), flush=True)

    _add_best_observed_gap(rows)
    failures = [
        {
            "instance_id": row["instance_id"],
            "split": row["split"],
            "cell_id": row["cell_id"],
            "method": row["method"],
            "status": row["status"],
            "verified": row["verified"],
            "violation_codes": row["violation_codes"],
            "diagnostics": next(
                value["result"]["diagnostics"]
                for value in raw_results
                if value["instance_id"] == row["instance_id"] and value["method"] == row["method"]
            ),
        }
        for row in rows
        if row["status"] not in SUCCESS or not row["verified"]
    ]
    labels = _candidate_labels(rows, raw_results)
    statistics_config = config.statistics
    method_stats = method_cell_statistics(
        rows,
        int(statistics_config["cluster_bootstrap_resamples"]),
        float(statistics_config["confidence_level"]),
        int(statistics_config["bootstrap_seed"]),
    )
    pairwise = frozen_pairwise_statistics(
        rows,
        tuple(str(item) for item in statistics_config["reference_methods"]),
        int(statistics_config["cluster_bootstrap_resamples"]),
        float(statistics_config["confidence_level"]),
        int(statistics_config["bootstrap_seed"]),
    )
    witness_failures = [item for item in hash_failures if item.startswith("WITNESS_") or item.startswith("VERIFY=") or item.startswith("MANIFEST_WITNESS_")]
    audit = {
        "schema_failures": 0,
        "split_leakage": len(leakage),
        "hash_failures": len(hash_failures),
        "witness_failures": len(witness_failures),
    }
    acceptance = evaluate_acceptance(rows, config.acceptance, audit)
    run_record = {
        "schema_version": f"a2-paper-results-{paper_version}",
        "runner_version": runner_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": config.evidence_label.value,
        "manifest_sha256": manifest.manifest_sha256,
        "config_sha256": manifest.config_sha256,
        "master_seed": config.master_seed,
        "instance_count": len(generated),
        "independent_group_count": len({item.task_group_id for item in generated}),
        "run_count": len(rows),
        "baseline_protocol": dict(config.baseline_protocol),
        "statistics_protocol": dict(config.statistics),
        "objective_weights": weights,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "boundaries": list(config.boundaries),
        "acceptance": acceptance,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "baseline_results.json", run_record)
    _write_csv(output_dir / "baseline_results.csv", rows)
    _write_json(output_dir / "method_cell_statistics.json", method_stats)
    _write_csv(output_dir / "method_cell_statistics.csv", method_stats)
    _write_json(output_dir / "frozen_pairwise_statistics.json", pairwise)
    _write_csv(output_dir / "frozen_pairwise_statistics.csv", pairwise)
    _write_json(output_dir / "failure_library.json", {"manifest_sha256": manifest.manifest_sha256, "failures": failures})
    _write_json(output_dir / "candidate_solver_labels.json", {"manifest_sha256": manifest.manifest_sha256, "split_usage_guard": {"train":"fit_only","validation":"selection_only","frozen_test":"evaluation_only","stress":"evaluation_only"}, "labels": labels})
    compact = {
        "manifest_sha256": manifest.manifest_sha256,
        "config_sha256": manifest.config_sha256,
        "instance_count": len(generated),
        "independent_group_count": len({item.task_group_id for item in generated}),
        "run_count": len(rows),
        "failure_count": len(failures),
        "acceptance": acceptance,
        "method_cell_statistics": method_stats,
        "frozen_pairwise_statistics": pairwise,
    }
    _write_json(report_dir / f"{report_stem}_summary.json", compact)
    _write_csv(report_dir / f"{report_stem}_method_cells.csv", method_stats)
    _write_csv(report_dir / f"{report_stem}_pairwise.csv", pairwise)
    markdown = _markdown(manifest, rows, failures, labels, method_stats, pairwise, acceptance, paper_version, report_stem)
    (output_dir / "results.md").write_text(markdown, encoding="utf-8")
    (report_dir / f"{report_stem}_results.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"instances":len(generated),"runs":len(rows),"manifest_sha256":manifest.manifest_sha256,"failures":len(failures),"a2_passed":acceptance["passed"]}), flush=True)


def _run_instance(item, methods, context, weights, manifest_hash, runner_version):
    rows = []
    raw = []
    for method_name, solve in methods:
        result = solve(item.instance)
        verification = verify_plan(item.instance, result.plan, context) if result.plan is not None else None
        verified = bool(verification and verification.feasible)
        objectives = dict(verification.objective_terms) if verification else {}
        row = {
            "instance_id": item.instance.instance_id,
            "split": item.split,
            "cell_id": item.cell_id,
            "paper_role": item.paper_role,
            "task_group_id": item.task_group_id,
            "variant_index": item.variant_index,
            "seed": item.seed,
            "feasibility_policy": item.feasibility_policy,
            "robot_count": len(item.instance.robots),
            "segment_count": len(item.instance.segments),
            "method": method_name,
            "method_id": result.method_id,
            "status": result.status,
            "verified": verified,
            "runtime_s": result.runtime_s,
            "weighted_proxy_score": _score(objectives, weights) if verified else None,
            "makespan_s": objectives.get("makespan"),
            "load_variance_s2": objectives.get("load_variance"),
            "travel_setup_time_s": objectives.get("travel_setup_time"),
            "priority_tardiness": objectives.get("priority_tardiness"),
            "assignment_mip_gap": result.mip_gap,
            "assignment_best_bound": result.best_bound,
            "violation_codes": ";".join(sorted({value.code for value in verification.violations})) if verification else "",
        }
        rows.append(row)
        raw.append({"instance_id":item.instance.instance_id,"method":method_name,"result":result.to_dict()})
    return {"runner_version":runner_version,"manifest_sha256":manifest_hash,"instance_id":item.instance.instance_id,"rows":rows,"raw_results":raw}


def _candidate_labels(rows, raw_results):
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train","validation","frozen_test","stress")}
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance_id"]].append(row)
    raw_map = {(item["instance_id"], item["method"]): item["result"] for item in raw_results}
    usage = {"train":"fit_only","validation":"selection_only","frozen_test":"evaluation_only","stress":"evaluation_only"}
    for instance_id, values in sorted(by_instance.items()):
        first = values[0]
        feasible = [item for item in values if item["verified"] and item["weighted_proxy_score"] is not None]
        if feasible:
            best = min(feasible, key=lambda item: (float(item["weighted_proxy_score"]), float(item["runtime_s"]), str(item["method"])))
            plan = raw_map[(instance_id, best["method"])]["plan"]
            source = best["method"]
            score = best["weighted_proxy_score"]
        else:
            plan, source, score = None, None, None
        result[first["split"]].append({"instance_id":instance_id,"cell_id":first["cell_id"],"task_group_id":first["task_group_id"],"source_method":source,"weighted_proxy_score":score,"usage":usage[first["split"]] if plan is not None else "failure_only","plan":plan})
    return result


def _add_best_observed_gap(rows):
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance_id"]].append(row)
    for values in by_instance.values():
        scores = [float(item["weighted_proxy_score"]) for item in values if item["weighted_proxy_score"] is not None]
        best = min(scores) if scores else None
        for item in values:
            score = item["weighted_proxy_score"]
            item["best_observed_relative_gap"] = None if score is None or best is None else (float(score)-best)/max(abs(best),1e-12)


def _markdown(manifest, rows, failures, labels, method_stats, pairwise, acceptance, paper_version, report_stem):
    status_counts = Counter(str(item["status"]) for item in failures)
    split_counts = Counter(str(item["split"]) for item in {row["instance_id"]:row for row in rows}.values())
    frozen = [item for item in method_stats if item["split"] == "frozen_test"]
    significant = [item for item in pairwise if item["wilcoxon_p_holm"] is not None and item["wilcoxon_p_holm"] < 0.05]
    lines = [
        f"# A2 paper-scale {paper_version} benchmark results",
        "",
        "Evidence: **SIM_GEOMETRIC**. These are programmatic continuous-process workcells, not real trajectories, collision certificates or physical-quality evidence.",
        "",
        f"- Manifest SHA-256: `{manifest.manifest_sha256}`",
        f"- Instances: {len({row['instance_id'] for row in rows})}; independent task groups: {len({row['task_group_id'] for row in rows})}; solver runs: {len(rows)}.",
        f"- Split instances: {dict(sorted(split_counts.items()))}.",
        f"- Failed/unverified method runs: {len(failures)}; status counts: {dict(sorted(status_counts.items()))}.",
        f"- Preregistered A2 gate: **{'PASSED' if acceptance['passed'] else 'FAILED'}**.",
        "",
        "## Acceptance checks",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in acceptance["checks"].items())
    boundary = (
        "Every ordinary v4 instance has a hashed constructive A1-proxy witness. "
        "A candidate `schedule_infeasible` result is therefore a method failure, not instance infeasibility. "
        "Witnesses were audit-only and were not candidate-solver inputs."
        if paper_version == "v4"
        else "`proxy_admissible` is only edge-mask coverage for atomic assignment units. "
        "`schedule_infeasible` is a method failure, not proof of global infeasibility."
    )
    lines.extend([
        f"- Train candidate coverage: {acceptance['train_candidate_coverage']:.3f}; validation: {acceptance['validation_candidate_coverage']:.3f}.",
        f"- Frozen cell coverage: {acceptance['frozen_cell_candidate_coverage']}.",
        f"- Designed-infeasible detection: {acceptance['designed_infeasible_detection_rate']:.3f}.",
        "",
        "## Frozen-test verified-plan rates",
        "",
        "| cell | method | verified groups rate [95% CI] | instances | runtime median [Q1,Q3] s |",
        "|---|---|---:|---:|---:|",
    ])
    for item in frozen:
        lines.append(f"| {item['cell_id']} | {item['method']} | {_fmt(item['group_verified_rate'])} [{_fmt(item['group_verified_ci_low'])},{_fmt(item['group_verified_ci_high'])}] | {item['verified_instances']}/{item['instances']} | {_fmt(item['median_runtime_s'])} [{_fmt(item['runtime_q1_s'])},{_fmt(item['runtime_q3_s'])}] |")
    lines.extend([
        "",
        "## Statistical interpretation",
        "",
        f"Holm-adjusted paired score comparisons with p<0.05: {len(significant)}/{sum(item['wilcoxon_p_holm'] is not None for item in pairwise)} testable comparisons. Full effect sizes, confidence intervals and jointly verified group counts are in `{report_stem}_pairwise.csv`.",
        "Quality comparisons are conditional on pairwise jointly verified variants; failed plans receive no imputed quality score. Stress results are descriptive only.",
        "",
        "## Candidate-label availability",
        "",
    ])
    for split, values in labels.items():
        lines.append(f"- {split}: {sum(item['plan'] is not None for item in values)}/{len(values)} verified candidates; planned use `{next((item['usage'] for item in values if item['plan'] is not None), 'failure_only')}`.")
    lines.extend([
        "",
        "## Boundaries",
        "",
        f"{boundary} Assignment MIP gap is not joint scheduling or path-planning optimality. No GNN was trained in A2.",
        "",
        "## Reproduction",
        "",
        "Run `scripts/run_a2_paper_v2.py` with the recorded config, output, and manifest paths.",
        "",
    ])
    return "\n".join(lines)


def _load_shard(path, manifest_hash, instance_id, runner_version):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if value.get("runner_version") == runner_version and value.get("manifest_sha256") == manifest_hash and value.get("instance_id") == instance_id else None


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _score(objectives, weights):
    return sum(float(weights.get(key,0.0))*float(value) for key,value in objectives.items())


def _fmt(value):
    return "n/a" if value is None else f"{float(value):.4f}"


def _resolve(root, path):
    return path if path.is_absolute() else root/path


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


if __name__ == "__main__":
    main()
