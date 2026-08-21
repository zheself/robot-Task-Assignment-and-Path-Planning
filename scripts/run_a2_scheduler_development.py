#!/usr/bin/env python3
"""Evaluate A2 scheduler revisions on v2 train/validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    generate_paper_benchmark,
    load_oracle_context,
    load_paper_config,
    load_paper_manifest,
    solve_deadline_aware_assignment_milp,
    solve_deadline_aware_load_balanced,
    solve_hybrid_assignment_milp,
    solve_hybrid_load_balanced,
    solve_load_balanced,
    solve_beam_alns,
    solve_assignment_beam_sequence,
    solve_order_aware_lns,
    verify_plan,
)
from safe_residual_rl.allocation.generation import canonical_instance_bytes

RUNNER_VERSION = "a2-scheduler-development-runner-v1"
SUCCESS = {"feasible", "optimal", "feasible_limit"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/allocation/a2_scheduler_development_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_allocation/a2_scheduler_development_v1"),
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/phase1_allocation")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    _validate_access_policy(config)
    paper_config = load_paper_config(root / config["source_benchmark"])
    manifest = load_paper_manifest(root / config["source_manifest"])
    if manifest.manifest_sha256 != config["source_manifest_sha256"]:
        raise RuntimeError("source manifest hash differs from development registration")

    generated = tuple(
        item
        for item in generate_paper_benchmark(paper_config)
        if item.split in set(config["allowed_splits"])
    )
    _verify_generated_subset(generated, manifest)
    if {item.split for item in generated} - {"train", "validation"}:
        raise RuntimeError("development runner attempted forbidden split access")

    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    protocol = SolverProtocol(
        "a1-solver-protocol-v1",
        float(config["milp_time_limit_s"]),
        float(config["milp_relative_gap"]),
        0,
    )
    weights = dict(paper_config.objective_weights)
    method_registry = {
        "load_balanced_v1": lambda item: solve_load_balanced(item, context),
        "deadline_load_balanced_v2": lambda item: solve_deadline_aware_load_balanced(item, context),
        "deadline_assignment_milp_v2": lambda item: solve_deadline_aware_assignment_milp(item, context, protocol),
        "hybrid_load_balanced_v2": lambda item: solve_hybrid_load_balanced(item, context),
        "hybrid_assignment_milp_v2": lambda item: solve_hybrid_assignment_milp(item, context, protocol),
        "order_aware_lns_v2": lambda item: solve_order_aware_lns(
                item,
                context,
                iterations=int(config["lns_iterations"]),
                seed=int(config["lns_seed"]),
                objective_weights=weights,
            ),
        "beam_alns_v1": lambda item: solve_beam_alns(
                item,
                context,
                iterations=int(config["beam_alns_iterations"]),
                seed=int(config["beam_alns_seed"]),
                beam_width=int(config["beam_width"]),
                beam_node_limit=int(config["beam_node_limit"]),
                objective_weights=weights,
            ),
        "assignment_beam_sequence_v1": lambda item: solve_assignment_beam_sequence(
                item,
                context,
                assignment_beam_width=int(config["assignment_beam_width"]),
                sequence_beam_width=int(config["sequence_beam_width"]),
                sequence_node_limit=int(config["sequence_node_limit"]),
                objective_weights=weights,
            ),
    }
    unknown = set(config["methods"]) - set(method_registry)
    if unknown:
        raise RuntimeError(f"unregistered development methods: {sorted(unknown)}")
    methods = tuple((name, method_registry[name]) for name in config["methods"])

    rows: list[dict[str, Any]] = []
    for index, generated_item in enumerate(generated, start=1):
        for method_name, method in methods:
            result = method(generated_item.instance)
            verified = False
            score = None
            if result.plan is not None:
                checked = verify_plan(generated_item.instance, result.plan, context)
                verified = checked.feasible and result.status in SUCCESS
                if verified:
                    terms = dict(checked.objective_terms)
                    score = sum(weights.get(key, 0.0) * value for key, value in terms.items())
            rows.append(
                {
                    "instance_id": generated_item.instance.instance_id,
                    "task_group_id": generated_item.task_group_id,
                    "split": generated_item.split,
                    "cell_id": generated_item.cell_id,
                    "method": method_name,
                    "status": result.status,
                    "verified": verified,
                    "weighted_proxy_score": score,
                    "runtime_s": result.runtime_s,
                    "diagnostics": "|".join(result.diagnostics),
                }
            )
        if index % 20 == 0 or index == len(generated):
            print(json.dumps({"progress": index, "total": len(generated)}), flush=True)

    summary = _summarize(rows)
    record = {
        "schema_version": "a2-scheduler-development-results-v1",
        "runner_version": RUNNER_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "source_manifest_sha256": manifest.manifest_sha256,
        "accessed_splits": sorted({item.split for item in generated}),
        "instance_count": len(generated),
        "run_count": len(rows),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "summary": summary,
        "boundaries": config["boundaries"],
        "rows": rows,
    }
    output_dir = root / args.output_dir
    report_dir = root / args.report_dir
    result_stem = str(config["version"]).replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "results.csv", rows)
    compact = {key: value for key, value in record.items() if key != "rows"}
    (report_dir / f"{result_stem}_summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(report_dir / f"{result_stem}_cells.csv", summary)
    (report_dir / f"{result_stem}_results.md").write_text(
        _markdown(record), encoding="utf-8"
    )
    print(json.dumps(compact, indent=2, sort_keys=True))


def _validate_access_policy(config: dict[str, Any]) -> None:
    if config.get("allowed_splits") != ["train", "validation"]:
        raise RuntimeError("A2 development must be restricted to train and validation")
    if set(config.get("forbidden_splits", ())) != {"frozen_test", "stress"}:
        raise RuntimeError("A2 development must explicitly forbid frozen_test and stress")


def _verify_generated_subset(generated, manifest) -> None:
    records = {item.instance_id: item for item in manifest.records}
    for item in generated:
        record = records.get(item.instance.instance_id)
        digest = hashlib.sha256(canonical_instance_bytes(item.instance)).hexdigest()
        if record is None or record.split != item.split or record.sha256 != digest:
            raise RuntimeError(f"generated instance differs from frozen v2 manifest: {item.instance.instance_id}")


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["cell_id"], row["method"])].append(row)
    result = []
    for (split, cell_id, method), values in sorted(groups.items()):
        verified = [item for item in values if item["verified"]]
        statuses = Counter(item["status"] for item in values)
        result.append(
            {
                "split": split,
                "cell_id": cell_id,
                "method": method,
                "instances": len(values),
                "verified_instances": len(verified),
                "candidate_coverage": len(verified) / len(values),
                "mean_score_verified": (
                    sum(float(item["weighted_proxy_score"]) for item in verified) / len(verified)
                    if verified
                    else None
                ),
                "median_runtime_s": float(np.median([item["runtime_s"] for item in values])),
                "status_counts": json.dumps(statuses, sort_keys=True),
            }
        )
    return result


def _markdown(record: dict[str, Any]) -> str:
    lines = [
        "# A2 scheduler development v1 results",
        "",
        "Evidence: `SIM_GEOMETRIC`; usage: development only.",
        f"Source manifest: `{record['source_manifest_sha256']}`.",
        f"Accessed splits: `{', '.join(record['accessed_splits'])}`. Frozen-test and stress were not accessed.",
        "",
        "| split | cell | method | coverage | verified/total | mean score (verified only) | median runtime s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in record["summary"]:
        score = "n/a" if item["mean_score_verified"] is None else f"{item['mean_score_verified']:.4f}"
        lines.append(
            f"| {item['split']} | {item['cell_id']} | {item['method']} | "
            f"{item['candidate_coverage']:.4f} | {item['verified_instances']}/{item['instances']} | "
            f"{score} | {item['median_runtime_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`schedule_infeasible` remains a method failure, not proof of global infeasibility. "
            "The assignment MILP remains assignment-only and its gap is not a joint scheduling gap. "
            "No frozen-test or stress result was used for selection.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
