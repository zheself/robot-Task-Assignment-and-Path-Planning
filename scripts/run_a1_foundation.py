#!/usr/bin/env python3
"""Run the deterministic A1 foundation matrix and emit JSON/CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.schema import HandoffPolicy, allocation_instance_from_dict
from safe_residual_rl.allocation.solvers import (
    load_solver_protocol,
    solve_assignment_milp,
    solve_greedy,
    solve_hungarian,
    solve_load_balanced,
)
from safe_residual_rl.allocation.verifier import verify_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/a1_foundation"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/phase1_allocation/a1_foundation_results.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    context_path = root / "configs/allocation/oracle_proxy_v1.json"
    protocol_path = root / "configs/allocation/solver_protocol_v1.json"
    context = load_oracle_context(context_path)
    protocol = load_solver_protocol(protocol_path)
    scenarios = _scenarios(root)
    methods = (
        ("greedy", lambda value: solve_greedy(value, context)),
        ("load_balanced", lambda value: solve_load_balanced(value, context)),
        ("hungarian", lambda value: solve_hungarian(value, context)),
        ("milp", lambda value: solve_assignment_milp(value, context, protocol)),
    )
    rows = []
    detailed = []
    for scenario_id, instance in scenarios:
        for method_name, solve in methods:
            result = solve(instance)
            verified = False
            objectives = {}
            violation_codes: list[str] = []
            if result.plan is not None:
                check = verify_plan(instance, result.plan, context)
                verified = check.feasible
                objectives = dict(check.objective_terms)
                violation_codes = sorted({item.code for item in check.violations})
            row = {
                "scenario_id": scenario_id,
                "evidence_label": instance.evidence_label.value,
                "segment_count": len(instance.segments),
                "robot_count": len(instance.robots),
                "method": method_name,
                "method_id": result.method_id,
                "status": result.status,
                "verified": verified,
                "runtime_s": result.runtime_s,
                "objective_value": result.objective_value,
                "best_bound": result.best_bound,
                "mip_gap": result.mip_gap,
                "makespan_s": objectives.get("makespan"),
                "load_variance_s2": objectives.get("load_variance"),
                "travel_setup_time_s": objectives.get("travel_setup_time"),
                "violation_codes": ";".join(violation_codes),
            }
            rows.append(row)
            detailed.append({"scenario_id": scenario_id, "result": result.to_dict()})

    manifest = {
        "schema_version": "a1-foundation-report-v1",
        "evidence_scope": "SIM_GEOMETRIC",
        "config_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (context_path, protocol_path)
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "boundaries": [
            "Synthetic fixture-derived curves; no real production trajectory evidence.",
            "Analytical reach/no-go proxies; no IK or continuous collision certificate.",
            "Assignment MILP followed by deterministic proxy scheduling; no joint motion-planning optimality claim.",
        ],
        "rows": rows,
        "detailed_results": detailed,
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report_path = args.report_path if args.report_path.is_absolute() else root / args.report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = _markdown(rows, manifest)
    (output_dir / "results.md").write_text(markdown, encoding="utf-8")
    report_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report_path": str(report_path), "rows": len(rows)}))


def _scenarios(root: Path):
    fixture_dir = root / "data/fixtures/allocation"
    base = allocation_instance_from_dict(load_auditable_fixture(fixture_dir / "01_valid_minimal.json")["instance"])
    template = base.segments[0]
    segments = []
    for index, offset in enumerate((0.00, 0.15, 0.30, 0.45)):
        start, end = (offset, 0.0, 0.0), (offset + 0.1, 0.0, 0.0)
        segments.append(replace(template, id=f"seg-{index}", parent_curve_id=f"curve-{index}", segment_index=0, sampled_curve_m=(start, end), start_pose=replace(template.start_pose, position_m=start), end_pose=replace(template.end_pose, position_m=end), predecessor_ids=(), handoff_policy=HandoffPolicy.FREE, priority=4-index))
    robot_1 = replace(base.robots[0], id="robot-1", base_pose=replace(base.robots[0].base_pose, position_m=(0.55, 0.0, 0.0)))
    balanced = replace(base, instance_id="a1-balanced-4x2", segments=tuple(segments), robots=(base.robots[0], robot_1))
    resource = allocation_instance_from_dict(load_auditable_fixture(fixture_dir / "04_valid_shared_zone.json")["instance"]).resources[0]
    resource_case = replace(balanced, instance_id="a1-resource-2x2", segments=tuple(replace(item, shared_resource_ids=(resource.id,)) for item in balanced.segments[:2]), resources=(resource,))
    infeasible = replace(balanced, instance_id="a1-infeasible-mask", robots=tuple(replace(item, capabilities=()) for item in balanced.robots))
    return (("balanced_4_segments_2_robots", balanced), ("capacity1_resource_2_segments", resource_case), ("infeasible_capability_mask", infeasible))


def _markdown(rows, manifest) -> str:
    feasible = [row for row in rows if row["status"] in {"feasible", "optimal", "feasible_limit"}]
    failures = [row for row in rows if row["status"] not in {"feasible", "optimal", "feasible_limit", "infeasible"}]
    lines = [
        "# A1 foundation results",
        "",
        "Evidence: **SIM_GEOMETRIC**. This is a deterministic engineering smoke matrix, not the A2 paper benchmark.",
        "",
        f"- Runs: {len(rows)}; feasible/optimal plans: {len(feasible)}; unexpected failures: {len(failures)}.",
        "- The deliberately capability-infeasible scenario must return `infeasible` for every method.",
        "- MILP numbers concern assignment proxy load; they do not establish joint schedule/path optimality.",
        "",
        "| scenario | method | status | verified | makespan (s) | load variance (s²) | runtime (s) | MIP gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {scenario_id} | {method_id} | {status} | {verified} | {makespan_s} | {load_variance_s2} | {runtime_s:.6f} | {mip_gap} |".format(**row))
    lines.extend(["", "## Evidence boundaries", ""] + [f"- {item}" for item in manifest["boundaries"]])
    lines.extend(["", "## Reproduction", "", "`PYTHONPATH=src .venv/bin/python scripts/run_a1_foundation.py`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
