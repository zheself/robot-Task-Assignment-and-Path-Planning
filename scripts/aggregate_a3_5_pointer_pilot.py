#!/usr/bin/env python3
"""Aggregate the fixed A3.5 development pilot and apply its continuation gate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.pointer_pilot import load_pointer_pilot_config
from safe_residual_rl.allocation.pointer_training import prepare_pointer_pilot
from safe_residual_rl.allocation.solvers import (
    SolverProtocol, solve_hybrid_assignment_milp, solve_hybrid_load_balanced, solve_order_aware_lns,
)
from safe_residual_rl.allocation.verifier import verify_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/allocation/a3_5_pointer_pilot_v1.json"))
    parser.add_argument("--pilot-root", type=Path, default=Path("outputs/phase1_allocation/a3_5_pointer_pilot_v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase1_allocation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_pointer_pilot_config(_resolve(root, args.config))
    pilot_root = _resolve(root, args.pilot_root)
    report_dir = _resolve(root, args.report_dir)
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    prepared = prepare_pointer_pilot(pilot_root / "data", pilot_root / "manifest.json", context, config)
    variants = [str(item["id"]) for item in config.raw["models"]["variants"]]
    seeds = [int(item) for item in config.raw["training"]["seeds"]]
    shards = []
    for variant in variants:
        for seed in seeds:
            path = pilot_root / "shards" / variant / f"seed_{seed:03d}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing A3.5 shard: {variant}:{seed}")
            item = json.loads(path.read_text(encoding="utf-8"))
            if not item.get("completed") or item["config_sha256"] != config.sha256 or item["manifest_sha256"] != prepared.manifest.manifest_sha256:
                raise RuntimeError(f"invalid A3.5 shard provenance: {path}")
            shards.append(item)
    variant_rows = [_variant_row(variant, shards) for variant in variants]
    cell_rows = _cell_rows(shards)
    pointer_rows = [item for item in variant_rows if item["decoder"] == "feasible_pair_pointer"]
    selected = min(pointer_rows, key=lambda item: (-item["mean_coverage"], _none_high(item["mean_conditional_score"]), -item["mean_pair_accuracy"], item["median_runtime_s"], item["variant"]))
    matched_static = selected["variant"].replace("_pair_pointer", "_static")
    decision = _continuation_decision(config, shards, cell_rows, selected["variant"], matched_static)
    baselines = _run_baselines(prepared.validation_examples, context, config)
    integrity = {
        "all_fifteen_shards_complete": len(shards) == len(variants) * len(seeds),
        "all_provenance_consistent": len({(item["config_sha256"], item["manifest_sha256"], item["access_sha256"], item["vocabulary_sha256"], item["normalizer_sha256"]) for item in shards}) == 1,
        "train_validation_only": all(item["accessed_splits"] == ["train", "validation"] and not item["forbidden_splits_accessed"] and item["v4_instance_or_witness_accessed"] is False for item in shards),
        "zero_hard_mask_violations": sum(item["validation_evaluation"]["hard_mask_violations"] for item in shards) == 0,
        "zero_atomicity_violations": sum(item["validation_evaluation"]["atomicity_violations"] for item in shards) == 0,
        "manifest_train_validation_only": {item.split for item in prepared.manifest.records} == {"train", "validation"},
        "all_metrics_finite": _finite(shards),
    }
    integrity_passed = all(integrity.values())
    decision["integrity_passed"] = integrity_passed
    decision["pilot_passed"] = bool(integrity_passed and decision["performance_gate_passed"])
    wording_key = "pass_wording" if decision["pilot_passed"] else "fail_wording"
    decision["required_wording"] = config.raw["continuation_gate"][wording_key]
    failures = [
        {"variant": item["variant"], "seed": item["seed"], **failure}
        for item in shards for failure in item["validation_evaluation"]["failures"]
    ]
    failure_counts = dict(sorted(Counter(failure.get("failure_class", failure.get("status", "unknown")) for failure in failures).items()))
    teacher_counts = Counter(item.teacher_method for item in prepared.manifest.records)
    summary = {
        "version": "a3-5-pointer-pilot-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "development_only": True,
        "config_sha256": config.sha256,
        "manifest_sha256": prepared.manifest.manifest_sha256,
        "source_hashes": dict(prepared.manifest.source_hashes),
        "train_instances": len(prepared.train_examples),
        "validation_instances": len(prepared.validation_examples),
        "validation_groups": len({item.task_group_id for item in prepared.validation_examples}),
        "variants": variants,
        "seeds": seeds,
        "selected_pair_pointer": selected["variant"],
        "matched_static_decoder": matched_static,
        "variant_results": variant_rows,
        "cell_results": cell_rows,
        "strong_baselines": baselines,
        "teacher_method_counts": dict(sorted(teacher_counts.items())),
        "teacher_constructive_fallback_count": sum(item.teacher_fallback for item in prepared.manifest.records),
        "failure_counts": failure_counts,
        "integrity_checks": integrity,
        "decision": decision,
        "frozen_test_generated_or_accessed": False,
        "stress_generated_or_accessed": False,
        "v4_instance_or_witness_accessed": False,
        "boundaries": config.raw["boundaries"],
    }
    pilot_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(pilot_root / "aggregate_summary.json", summary)
    _write_json(pilot_root / "failure_library.json", failures)
    _write_csv(pilot_root / "variant_results.csv", variant_rows)
    _write_csv(pilot_root / "cell_results.csv", cell_rows)
    _write_csv(pilot_root / "strong_baselines.csv", baselines)
    compact_prefix = "a3_5_pointer_pilot_v1"
    _write_json(report_dir / f"{compact_prefix}_summary.json", summary)
    _write_csv(report_dir / f"{compact_prefix}_variants.csv", variant_rows)
    _write_csv(report_dir / f"{compact_prefix}_cells.csv", cell_rows)
    _write_csv(report_dir / f"{compact_prefix}_baselines.csv", baselines)
    markdown = _markdown(summary)
    (pilot_root / "results.md").write_text(markdown, encoding="utf-8")
    (report_dir / f"{compact_prefix}_results.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"pilot_passed": decision["pilot_passed"], "selected_pointer": selected["variant"], "matched_static": matched_static, "frozen_accessed": False}), flush=True)


def _variant_row(variant: str, shards) -> dict[str, object]:
    selected = [item for item in shards if item["variant"] == variant]
    values = [item["validation_evaluation"] for item in selected]
    pair = [item["teacher_forced_pair_accuracy"] for item in values if item["teacher_forced_pair_accuracy"] is not None]
    scores = [item["conditional_weighted_proxy_score"] for item in values]
    return {
        "variant": variant,
        "encoder": selected[0]["encoder"],
        "decoder": selected[0]["decoder"],
        "seeds": len(selected),
        "seed_coverages": json.dumps({str(item["seed"]): item["validation_evaluation"]["verified_candidate_coverage"] for item in selected}, sort_keys=True),
        "mean_coverage": float(np.mean([item["verified_candidate_coverage"] for item in values])),
        "std_coverage": float(np.std([item["verified_candidate_coverage"] for item in values])),
        "mean_pair_accuracy": float(np.mean(pair)) if pair else None,
        "mean_rollout_completion": float(np.mean([item["greedy_rollout_completion_rate"] for item in values])),
        "mean_conditional_score": None if any(item is None for item in scores) else float(np.mean(scores)),
        "median_runtime_s": float(statistics.median(item["median_inference_runtime_s"] for item in values)),
        "decoder_dead_ends": int(sum(item["decoder_dead_ends"] for item in values)),
        "validation_failures": int(sum(len(item["failures"]) for item in values)),
    }


def _cell_rows(shards) -> list[dict[str, object]]:
    rows = []
    for variant in sorted({item["variant"] for item in shards}):
        selected = [item for item in shards if item["variant"] == variant]
        for cell in sorted(selected[0]["validation_cells"]):
            values = [item["validation_cells"][cell] for item in selected]
            rows.append({
                "variant": variant, "cell_id": cell, "seeds": len(values),
                "seed_coverages": json.dumps({str(item["seed"]): item["validation_cells"][cell]["verified_candidate_coverage"] for item in selected}, sort_keys=True),
                "mean_coverage": float(np.mean([item["verified_candidate_coverage"] for item in values])),
                "std_coverage": float(np.std([item["verified_candidate_coverage"] for item in values])),
                "mean_pair_accuracy": _optional_mean([item["teacher_forced_pair_accuracy"] for item in values]),
                "mean_rollout_completion": float(np.mean([item["greedy_rollout_completion_rate"] for item in values])),
                "conditional_score": _optional_mean([item["conditional_weighted_proxy_score"] for item in values]),
                "decoder_dead_ends": int(sum(item["decoder_dead_ends"] for item in values)),
            })
    return rows


def _continuation_decision(config, shards, cell_rows, pointer: str, static: str) -> dict[str, object]:
    seeds = [int(item) for item in config.raw["training"]["seeds"]]
    pointer_shards = {item["seed"]: item for item in shards if item["variant"] == pointer}
    static_shards = {item["seed"]: item for item in shards if item["variant"] == static}
    seed_diffs = {str(seed): pointer_shards[seed]["validation_evaluation"]["verified_candidate_coverage"] - static_shards[seed]["validation_evaluation"]["verified_candidate_coverage"] for seed in seeds}
    pointer_cells = {item["cell_id"]: item["mean_coverage"] for item in cell_rows if item["variant"] == pointer}
    static_cells = {item["cell_id"]: item["mean_coverage"] for item in cell_rows if item["variant"] == static}
    cell_diffs = {cell: pointer_cells[cell] - static_cells[cell] for cell in pointer_cells}
    gate = config.raw["continuation_gate"]
    pointer_mean = float(np.mean([item["validation_evaluation"]["verified_candidate_coverage"] for item in pointer_shards.values()]))
    static_mean = float(np.mean([item["validation_evaluation"]["verified_candidate_coverage"] for item in static_shards.values()]))
    checks = {
        "at_least_two_seed_wins": sum(value > 0 for value in seed_diffs.values()) >= int(gate["minimum_seed_wins_over_matched_static"]),
        "one_overall_group_equivalent_improvement": pointer_mean - static_mean >= float(gate["minimum_overall_group_equivalent_improvement"]) - 1e-12,
        "two_constraint_cells_improve": sum(cell_diffs[cell] > 0 for cell in gate["constraint_cells"]) >= int(gate["required_improved_constraint_cells"]),
        "no_cell_regresses_over_one_group": all(value >= -float(gate["maximum_cell_regression"]) - 1e-12 for value in cell_diffs.values()),
        "zero_pointer_dead_ends": sum(item["validation_evaluation"]["decoder_dead_ends"] for item in pointer_shards.values()) == 0,
        "no_repair_used": config.raw["selection"]["repair"] == "forbidden",
        "final_verifier_coverage_is_decision_metric": gate["improvement_metric"] == "final_verifier_coverage",
    }
    return {
        "selected_pointer": pointer, "matched_static": static,
        "pointer_mean_coverage": pointer_mean, "static_mean_coverage": static_mean,
        "mean_coverage_difference": pointer_mean - static_mean,
        "seed_coverage_differences": seed_diffs, "cell_coverage_differences": cell_diffs,
        "checks": checks, "performance_gate_passed": all(checks.values()),
    }


def _run_baselines(examples, context, config) -> list[dict[str, object]]:
    raw = config.raw["baselines"]
    protocol = SolverProtocol("a1-solver-protocol-v1", float(raw["milp_time_limit_s"]), float(raw["milp_relative_gap"]), int(raw["lns_seed"]))
    registry = {
        "hybrid_assignment_milp": lambda instance: solve_hybrid_assignment_milp(instance, context, protocol),
        "order_aware_lns": lambda instance: solve_order_aware_lns(instance, context, iterations=int(raw["lns_iterations"]), seed=int(raw["lns_seed"]), objective_weights=config.objective_weights),
        "hybrid_load_balanced": lambda instance: solve_hybrid_load_balanced(instance, context),
    }
    rows = []
    for method in raw["methods"]:
        verified = 0; scores = []; runtimes = []; status = Counter()
        for example in examples:
            started = time.perf_counter(); result = registry[method](example.instance); runtimes.append(time.perf_counter() - started); status[result.status] += 1
            check = verify_plan(example.instance, result.plan, context) if result.plan is not None else None
            if check is not None and check.feasible:
                verified += 1; scores.append(sum(config.objective_weights.get(key, 0.0) * value for key, value in check.objective_terms))
        rows.append({"method": method, "instances": len(examples), "verified": verified, "coverage": verified / len(examples), "conditional_score": float(np.mean(scores)) if scores else None, "median_runtime_s": float(statistics.median(runtimes)), "status_counts": json.dumps(dict(sorted(status.items())), sort_keys=True)})
    return rows


def _finite(shards) -> bool:
    for item in shards:
        for split in ("train_evaluation", "validation_evaluation"):
            for key in ("mean_loss", "greedy_rollout_completion_rate", "verified_candidate_coverage", "median_inference_runtime_s"):
                if not np.isfinite(item[split][key]): return False
    return True


def _markdown(summary) -> str:
    decision = summary["decision"]
    lines = [
        "# A3.5 Feasible-Pair Pointer development pilot results", "",
        "Evidence: **SIM_GEOMETRIC, development only**. No frozen-test/stress was generated or accessed.", "",
        f"- Pilot decision: **{'CONTINUE_TO_NEW_PREREGISTRATION' if decision['pilot_passed'] else 'STOP_POINTER_BRANCH'}**.",
        f"- Selected pointer: `{summary['selected_pair_pointer']}`; matched static decoder: `{summary['matched_static_decoder']}`.",
        f"- Mean coverage: {decision['pointer_mean_coverage']:.3f} pointer vs {decision['static_mean_coverage']:.3f} static (difference {decision['mean_coverage_difference']:+.3f}).", "",
        "## Integrity and continuation checks", "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in summary["integrity_checks"].items())
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in decision["checks"].items())
    lines.extend(["", "## Model variants", "", "| variant | coverage mean±std | pair accuracy | completion | conditional score | runtime s | dead ends |", "|---|---:|---:|---:|---:|---:|---:|"])
    for item in summary["variant_results"]:
        lines.append(f"| {item['variant']} | {item['mean_coverage']:.3f}±{item['std_coverage']:.3f} | {_fmt(item['mean_pair_accuracy'])} | {item['mean_rollout_completion']:.3f} | {_fmt(item['mean_conditional_score'])} | {item['median_runtime_s']:.5f} | {item['decoder_dead_ends']} |")
    lines.extend(["", "## Strong validation baselines", "", "| method | coverage | conditional score | runtime s |", "|---|---:|---:|---:|"])
    for item in summary["strong_baselines"]:
        lines.append(f"| {item['method']} | {item['coverage']:.3f} | {_fmt(item['conditional_score'])} | {item['median_runtime_s']:.5f} |")
    lines.extend(["", "## Registered conclusion", "", decision["required_wording"], "", "Conditional scores exclude failed candidates and cannot replace coverage. This pilot neither changes the A3 v4 failure nor establishes GNN/Pointer superiority, motion safety, real execution or physical-process improvement.", ""])
    return "\n".join(lines)


def _optional_mean(values):
    return None if any(item is None for item in values) else float(np.mean(values))
def _none_high(value): return float("inf") if value is None else float(value)
def _fmt(value): return "n/a" if value is None else f"{float(value):.4f}"
def _write_json(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
def _write_csv(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def _resolve(root, value): return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
