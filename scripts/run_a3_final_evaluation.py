#!/usr/bin/env python3
"""Develop on validation or execute the single sealed A3 final evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

from safe_residual_rl.allocation.a3_protocol import prepare_a3_development
from safe_residual_rl.allocation.final_evaluation import (
    EXPECTED_METHODS,
    FinalEvaluationItem,
    audit_witnesses_after_prediction,
    classify_final_result,
    evaluate_items,
    load_final_items,
    load_locked_models,
    method_cell_metrics,
    sha256_file,
    strong_pairwise_statistics,
    verify_protocol_locks,
)
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.paper_benchmark import (
    PaperRecord,
    audit_paper_leakage,
    load_paper_config,
    load_paper_manifest,
)

PROTOCOL_SHA256 = "ce574b6b62c2218f8a2f7b3130646444cc00b60ccc69c6822bdde5d3f48ab756"
SEALED_CONFIRMATION = "RUN_A3_FINAL_V1_ONCE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "sealed"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/allocation/a3_final_evaluation_v1.json"),
    )
    parser.add_argument(
        "--development-root",
        type=Path,
        default=Path("outputs/phase1_allocation/a3_development_v1/data"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/allocation/a2_paper_manifest_v4.json"),
    )
    parser.add_argument(
        "--seal",
        type=Path,
        default=Path("configs/allocation/a3_final_evaluator_seal_v1.json"),
    )
    parser.add_argument("--confirm", default="")
    parser.add_argument("--development-instances", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = _resolve(root, args.protocol)
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("final protocol checksum changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    lock_failures = verify_protocol_locks(root, protocol)
    if lock_failures:
        raise RuntimeError("locked source/provenance mismatch: " + ",".join(lock_failures))
    prepared = prepare_a3_development(_resolve(root, args.development_root), context)
    models = load_locked_models(root, protocol, prepared)
    if args.mode == "development":
        _development(root, protocol, prepared, models, context, args.development_instances)
        return
    if args.confirm != SEALED_CONFIRMATION:
        raise PermissionError("sealed execution requires the exact one-time confirmation token")
    seal_path = _resolve(root, args.seal)
    seal = _verify_seal(root, seal_path, protocol_path)
    _sealed(root, protocol, prepared, models, context, _resolve(root, args.manifest), seal)


def _development(root, protocol, prepared, models, context, count):
    if count < 1:
        raise ValueError("development instance count must be positive")
    selected = []
    seen_cells = set()
    for example in prepared.validation_examples:
        if example.cell_id in seen_cells:
            continue
        seen_cells.add(example.cell_id)
        instance = example.instance
        record = PaperRecord(
            instance_id=instance.instance_id,
            split="validation",
            cell_id=example.cell_id,
            paper_role="development_smoke",
            task_group_id=instance.instance_id.rsplit("-v", 1)[0],
            variant_index=0,
            seed=0,
            feasibility_policy="constructive_witness_required",
            proxy_admissible=True,
            workpiece_id=instance.workpiece_id,
            layout_id=instance.layout_id,
            parent_curve_ids=tuple(sorted({item.parent_curve_id for item in instance.segments})),
            evidence_label="SIM_GEOMETRIC",
            robot_count=len(instance.robots),
            segment_count=len(instance.segments),
            relative_path="validation-only-in-memory",
            sha256=hashlib.sha256(
                json.dumps(
                    instance.to_dict(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )
        selected.append(FinalEvaluationItem(record, instance))
        if len(selected) == count:
            break
    rows, raw = evaluate_items(selected, models, context, prepared, protocol)
    expected = len(selected) * (len(models) + len(EXPECTED_METHODS))
    checks = {
        "validation_only": all(item["split"] == "validation" for item in rows),
        "complete_method_matrix": len(rows) == expected,
        "all_metrics_finite": all(np.isfinite(float(item["runtime_s"])) for item in rows),
        "no_evaluation_exception": all(item["status"] != "evaluation_exception" for item in rows),
    }
    output = root / "outputs/phase1_allocation/a3_final_evaluator_development_v1"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", {"checks": checks, "passed": all(checks.values()), "rows": rows})
    _write_json(output / "raw_predictions.json", raw)
    print(json.dumps({"mode": "development", "instances": len(selected), "rows": len(rows), "passed": all(checks.values())}), flush=True)
    if not all(checks.values()):
        raise RuntimeError("development evaluator smoke failed")


def _sealed(root, protocol, prepared, models, context, manifest_path, seal):
    output = root / protocol["outputs"]["ignored_root"]
    if output.exists():
        raise FileExistsError("one-time final output already exists; sealed rerun is forbidden")
    output.mkdir(parents=True)
    started = {
        "version": "a3-final-attempt-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STARTED_BEFORE_FROZEN_ACCESS",
        "protocol_sha256": PROTOCOL_SHA256,
        "seal_sha256": sha256_file(root / "configs/allocation/a3_final_evaluator_seal_v1.json"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }
    _write_json(output / "attempt_started.json", started)
    try:
        manifest = load_paper_manifest(manifest_path)
        if manifest.manifest_sha256 != protocol["source_locks"]["a2_manifest_sha256"]:
            raise RuntimeError("A2 manifest hash differs from final protocol")
        benchmark = load_paper_config(root / "configs/allocation/benchmark_v4.json")
        if manifest.config_sha256 != benchmark.config_sha256:
            raise RuntimeError("manifest config hash differs from final protocol")
        leakage = audit_paper_leakage(manifest.records)
        items = load_final_items(root, manifest, ("frozen_test", "stress"))
        frozen = [item for item in items if item.record.split == "frozen_test"]
        stress = [item for item in items if item.record.split == "stress"]
        if len(frozen) != 144 or len(stress) != 24:
            raise RuntimeError("sealed split count mismatch")
        rows, raw = evaluate_items(items, models, context, prepared, protocol)
        # This file proves predictions existed before any witness was opened.
        _write_json(output / "predictions_before_witness.json", raw)
        witness_failures, agreements = audit_witnesses_after_prediction(root, items, raw, context)
        for row in rows:
            key = (str(row["instance_id"]), str(row["method"]))
            if key in agreements:
                row["teacher_agreement"] = agreements[key]
        stats = protocol["statistics"]
        cell_metrics = method_cell_metrics(
            rows,
            resamples=int(stats["cluster_bootstrap_resamples"]),
            confidence=float(stats["confidence_level"]),
            seed=int(stats["bootstrap_seed"]),
        )
        pairwise = strong_pairwise_statistics(rows, protocol)
        methods = set(EXPECTED_METHODS) | {f"edge_mlp_seed_{seed}" for seed in (17, 29, 43)}
        frozen_matrix = len(frozen) * len(methods)
        stress_matrix = len(stress) * len(methods)
        negative = [row for row in rows if row["cell_id"] == "designed_edge_infeasible"]
        finite = all(
            np.isfinite(float(row["runtime_s"]))
            and (row["weighted_proxy_score"] is None or np.isfinite(float(row["weighted_proxy_score"])))
            for row in rows
        )
        integrity = {
            "all_source_and_checkpoint_hashes_match": True,
            "all_144_frozen_instances_evaluated_by_all_three_checkpoints_and_all_eight_baselines": sum(row["split"] == "frozen_test" for row in rows) == frozen_matrix,
            "complete_stress_matrix": sum(row["split"] == "stress" for row in rows) == stress_matrix,
            "zero_schema_or_manifest_failures": not leakage,
            "zero_nan_or_unexpected_exceptions": finite and all(row["status"] != "evaluation_exception" for row in rows),
            "hard_mask_and_atomic_units_never_violated": all(
                row["status"] != "constraint_integrity_failure" for row in rows
            ),
            "designed_edge_infeasible_detection_rate_equals_one": bool(negative) and all(not row["verified"] and row["status"] == "infeasible" for row in negative),
            "zero_witness_hash_or_verifier_failures": not witness_failures,
        }
        decision = classify_final_result(rows, protocol, pairwise, integrity)
        failures = [
            {
                "instance_id": row["instance_id"],
                "split": row["split"],
                "cell_id": row["cell_id"],
                "method": row["method"],
                "status": row["status"],
                "violation_codes": row["violation_codes"],
            }
            for row in rows
            if not row["verified"]
        ]
        run_manifest = {
            **started,
            "status": "COMPLETED",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest.manifest_sha256,
            "evaluator_seal": seal,
            "instance_counts": {"frozen_test": len(frozen), "stress": len(stress)},
            "row_count": len(rows),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "device": "cpu",
            },
            "integrity_checks": integrity,
        }
        _write_json(output / "run_manifest.json", run_manifest)
        _write_json(output / "rows.json", rows)
        _write_csv(output / "rows.csv", rows)
        _write_json(output / "raw_predictions.json", raw)
        _write_json(output / "method_cell_metrics.json", cell_metrics)
        _write_csv(output / "method_cell_metrics.csv", cell_metrics)
        _write_json(output / "strong_pairwise_statistics.json", pairwise)
        _write_csv(output / "strong_pairwise_statistics.csv", pairwise)
        _write_json(output / "failure_library.json", {"failures": failures, "witness_failures": witness_failures})
        _write_json(output / "decision.json", decision)
        report = root / protocol["outputs"]["compact_report_root"]
        report.mkdir(parents=True, exist_ok=True)
        summary = {
            "version": "a3-final-evaluation-results-v1",
            "evidence_label": "SIM_GEOMETRIC",
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": manifest.manifest_sha256,
            "instance_counts": run_manifest["instance_counts"],
            "row_count": len(rows),
            "integrity_checks": integrity,
            "decision": decision,
            "method_cell_metrics": cell_metrics,
            "strong_pairwise_statistics": pairwise,
            "failure_count": len(failures),
            "boundaries": protocol["boundaries"],
        }
        _write_json(report / "a3_final_evaluation_v1_summary.json", summary)
        _write_csv(report / "a3_final_evaluation_v1_cells.csv", cell_metrics)
        _write_csv(report / "a3_final_evaluation_v1_pairwise.csv", pairwise)
        markdown = _markdown(summary)
        (output / "results.md").write_text(markdown, encoding="utf-8")
        (report / "a3_final_evaluation_v1_results.md").write_text(markdown, encoding="utf-8")
        print(json.dumps({"mode": "sealed", "result_class": decision["result_class"], "a3_final_passed": decision["a3_final_passed"], "rows": len(rows)}), flush=True)
    except Exception as exc:
        _write_json(
            output / "attempt_failure.json",
            {
                "status": "INFRASTRUCTURE_OR_EVALUATOR_FAILURE",
                "type": type(exc).__name__,
                "message": str(exc),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


def _verify_seal(root, seal_path, protocol_path):
    if not seal_path.is_file():
        raise FileNotFoundError("final evaluator is not sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_AFTER_DEVELOPMENT_TESTS_BEFORE_FROZEN_ACCESS":
        raise RuntimeError("evaluator seal status invalid")
    if seal.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("seal protocol hash mismatch")
    for relative, expected in seal["files"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"sealed evaluator file changed: {relative}")
    return seal


def _markdown(summary):
    decision = summary["decision"]
    lines = [
        "# A3 one-time frozen evaluation result",
        "",
        "Evidence: **SIM_GEOMETRIC**. This is the single preregistered A3 frozen evaluation.",
        "",
        f"- Result class: **{decision['result_class']}**.",
        f"- A3 minimum final gate: **{'PASSED' if decision['a3_final_passed'] else 'FAILED'}**.",
        f"- Learned overall verified coverage: {decision.get('learned_overall_coverage', 'n/a')}.",
        f"- Best context/weak overall coverage: {decision.get('best_context_overall_coverage', 'n/a')}.",
        f"- Required wording: {decision.get('required_wording', 'No performance conclusion is allowed.')} ",
        "",
        "## Integrity and gate checks",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in decision["checks"].items())
    lines.extend(["", "## Learned per-difficulty coverage", "", "| cell | coverage |", "|---|---:|"])
    for cell, value in decision.get("learned_cell_coverage", {}).items():
        lines.append(f"| {cell} | {value:.4f} |")
    lines.extend(["", "## Strong-baseline paired coverage", "", "| reference | difference [95% CI] | Holm p |", "|---|---:|---:|"])
    for item in summary["strong_pairwise_statistics"]:
        lines.append(f"| {item['reference_method']} | {item['coverage_difference']:.4f} [{item['coverage_ci_low']:.4f}, {item['coverage_ci_high']:.4f}] | {_fmt(item['coverage_wilcoxon_p_holm'])} |")
    lines.extend([
        "",
        "## Boundary",
        "",
        "The selected model is an edge-MLP, not a GNN. Results concern only the A1 geometric/timing proxy. They do not establish motion-level collision safety, robot execution, real production, physical quality or sim-to-real success.",
        "",
    ])
    return "\n".join(lines)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root, path):
    return path if path.is_absolute() else root / path


def _fmt(value):
    return "n/a" if value is None else f"{float(value):.4g}"


if __name__ == "__main__":
    main()
