#!/usr/bin/env python3
"""Generate, develop, seal and evaluate the A4a development-only pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.allocation.graphs import build_a3_graph
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import identical_repair
from safe_residual_rl.allocation.solvers import solve_order_aware_lns
from safe_residual_rl.allocation.verifier import verify_plan
from safe_residual_rl.allocation.warm_start import (
    canonical_digest, dependency_versions, evaluate_raw_references,
    export_locked_preprocessor, generate_a4_data, initialize_all,
    load_a4_config, load_a4_items, load_fixed_initializers,
    load_locked_preprocessor, sha256_file,
)

CONFIG_PATH = ROOT / "configs/allocation/a4_warm_start_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/phase1_allocation/a4_warm_start_pilot_v1"
SOURCE_FILES = (
    "src/safe_residual_rl/allocation/warm_start.py",
    "src/safe_residual_rl/allocation/repair/identical.py",
    "scripts/run_a4_warm_start_pilot.py",
    "tests/allocation/test_a4_warm_start_pilot.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export-preprocessor", "generate", "develop", "preflight", "seal", "evaluate", "aggregate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--legacy-pilot-root", type=Path)
    parser.add_argument("--cell")
    args = parser.parse_args()
    config = load_a4_config(CONFIG_PATH)
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.command == "export-preprocessor":
        if args.legacy_pilot_root is None:
            raise SystemExit("--legacy-pilot-root required")
        registry = json.loads((ROOT / "configs/allocation/a3_5_sealed_final_v1.json").read_text())
        artifact = export_locked_preprocessor(
            args.legacy_pilot_root.parents[2], args.legacy_pilot_root,
            output / "immutable_preprocessing.json", context,
            registry["training_lock"]["vocabulary_sha256"], registry["training_lock"]["normalizer_sha256"],
        )
        _write_json(output / "preprocessor_export_record.json", artifact)
    elif args.command == "generate":
        manifest = generate_a4_data(ROOT, config, output / "corpus", context)
        _write_json(output / "generation_record.json", {"manifest_sha256": manifest["manifest_sha256"], "config_sha256": config["config_sha256"], "command": " ".join(sys.argv), "versions": dependency_versions()})
    elif args.command == "develop":
        targets = develop(config, context, output)
        _write_json(output / "train_frozen_targets.json", targets)
    elif args.command == "preflight":
        run_evaluation(config, context, output, split="train", cell=args.cell or "iid_small", limit=1, destination=output / "preflight_rows.jsonl", validation_access=False)
        _write_json(output / "preflight_record.json", _job_record("train-only evaluator preflight; validation not accessed"))
    elif args.command == "seal":
        seal(config, output)
    elif args.command == "evaluate":
        if not args.cell:
            raise SystemExit("--cell is required")
        validate_seal(config, output)
        marker = output / "validation_access" / f"{args.cell}.started"
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            raise RuntimeError(f"validation cell already accessed: {args.cell}")
        marker.write_text(json.dumps(_job_record("single validation access"), indent=2))
        run_evaluation(config, context, output, split="validation", cell=args.cell, limit=None, destination=output / "validation_shards" / f"{args.cell}.jsonl", validation_access=True)
        (marker.with_suffix(".complete")).write_text(json.dumps(_job_record("single validation access completed"), indent=2))
    else:
        aggregate(config, output)


def _prepared(config, context, output, split, cell, limit):
    vocabulary, normalizer, artifact_hash = load_locked_preprocessor(output / "immutable_preprocessing.json")
    items, manifest_hash = load_a4_items(output / "corpus", split, context)
    selected = [(record, instance) for record, instance in items if record["cell_id"] == cell]
    if limit is not None:
        selected = selected[:limit]
    graphs = [(record, instance, normalizer.transform(build_a3_graph(instance, context, vocabulary, split=split))) for record, instance in selected]
    registry = json.loads((ROOT / "configs/allocation/a3_5_sealed_final_v1.json").read_text())
    models = load_fixed_initializers(ROOT, registry, graphs[0][2])
    return graphs, models, manifest_hash, artifact_hash


def develop(config, context, output):
    weights = config["repair"]["objective_weights"]
    cell_targets = {}
    smoke = []
    for cell in config["data"]["cells"]:
        graphs, models, manifest_hash, artifact_hash = _prepared(config, context, output, "train", cell, None)
        scores = []
        for record, instance, graph in graphs:
            lns = solve_order_aware_lns(instance, context, iterations=int(config["repair"]["original_lns_iterations"]), seed=0, objective_weights=weights)
            if lns.plan is not None and verify_plan(instance, lns.plan, context).feasible and lns.objective_value is not None:
                scores.append(float(lns.objective_value))
        if not scores:
            raise RuntimeError(f"no verified train LNS target for {cell}")
        cell_targets[cell] = float(statistics.median(scores))
        # Repair development is deliberately limited to the first train item.
        record, instance, graph = graphs[0]
        initializers = initialize_all(instance, graph, models, context, config)
        for name in ("pair_pointer_seed_101", "static_seed_101", "hybrid_load_balanced", "hybrid_assignment_milp", "cold_start"):
            state, runtime, status = initializers[name]
            result = identical_repair(instance, context, state, iterations=10, random_seed=401, weights=weights, target_score=cell_targets[cell], destroy_fractions=config["repair"]["destroy_fractions"])
            smoke.append({"cell_id": cell, "instance_id": instance.instance_id, "initializer": name, "initializer_status": status, "initializer_runtime_s": runtime, "repair": result.to_dict()})
    targets = {
        "version": "a4-train-frozen-targets-v1", "selection_split": "train", "validation_accessed": False,
        "rule": config["target_score"]["rule"], "targets": cell_targets,
        "manifest_sha256": manifest_hash, "preprocessing_artifact_sha256": artifact_hash,
        "smoke_sha256": canonical_digest(smoke), "config_sha256": config["config_sha256"],
    }
    targets["target_record_sha256"] = canonical_digest(targets)
    _write_json(output / "development_smoke.json", smoke)
    return targets


def seal(config, output):
    targets = json.loads((output / "train_frozen_targets.json").read_text())
    if targets["validation_accessed"]:
        raise RuntimeError("target selection touched validation")
    if (output / "validation_access").exists():
        raise RuntimeError("cannot seal after validation access")
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_FILES}
    registry = json.loads((ROOT / "configs/allocation/a3_5_sealed_final_v1.json").read_text())
    checkpoint_hashes = {}
    for method in registry["fixed_neural_methods"].values():
        for seed, path in method["checkpoint_files"].items():
            checkpoint_hashes[f"{method['variant']}:{seed}"] = sha256_file(ROOT / path)
    seal_payload = {
        "version": "a4-warm-start-development-seal-v1", "config_sha256": config["config_sha256"],
        "manifest_file_sha256": sha256_file(output / "corpus/manifest.json"),
        "preprocessor_file_sha256": sha256_file(output / "immutable_preprocessing.json"),
        "target_file_sha256": sha256_file(output / "train_frozen_targets.json"),
        "source_hashes": source_hashes, "checkpoint_hashes": checkpoint_hashes,
        "dependencies": dependency_versions(), "commands": {"evaluate": "OMP_NUM_THREADS=1 python scripts/run_a4_warm_start_pilot.py evaluate --cell <registered-cell>"},
        "slurm": _job_record("seal"),
    }
    seal_payload["seal_sha256"] = canonical_digest(seal_payload)
    _write_json(output / "seal.json", seal_payload)


def validate_seal(config, output):
    seal = json.loads((output / "seal.json").read_text())
    expected = seal.pop("seal_sha256")
    if canonical_digest(seal) != expected or seal["config_sha256"] != config["config_sha256"]:
        raise RuntimeError("A4a seal integrity failure")
    for path, digest in seal["source_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise RuntimeError(f"A4a source changed after seal: {path}")
    if sha256_file(output / "corpus/manifest.json") != seal["manifest_file_sha256"] or sha256_file(output / "train_frozen_targets.json") != seal["target_file_sha256"]:
        raise RuntimeError("A4a data/target changed after seal")


def run_evaluation(config, context, output, *, split, cell, limit, destination, validation_access):
    graphs, models, manifest_hash, artifact_hash = _prepared(config, context, output, split, cell, limit)
    weights = config["repair"]["objective_weights"]
    targets = json.loads((output / "train_frozen_targets.json").read_text())["targets"]
    rows = []
    for record, instance, graph in graphs:
        initializers, raw = evaluate_raw_references(instance, graph, models, context, config, weights)
        for item in raw:
            rows.append(_base(record, "raw", None, None, item["initializer"], None, item["verified"], item["objective"], item["initializer_runtime_s"], 0.0, item["initializer_runtime_s"], item["failure_reason"], None))
        for initializer, (state, init_runtime, init_status) in initializers.items():
            for repair_seed in config["repair"]["random_seeds"]:
                for iterations in config["repair"]["fixed_iterations"]:
                    result = identical_repair(instance, context, state, iterations=int(iterations), random_seed=int(repair_seed), weights=weights, target_score=float(targets[cell]), destroy_fractions=config["repair"]["destroy_fractions"])
                    rows.append(_repair_row(record, "fixed_iterations", float(iterations), initializer, repair_seed, init_status, init_runtime, result, None))
                for budget in config["repair"]["fixed_end_to_end_time_s"]:
                    remaining = float(budget) - init_runtime
                    if remaining <= 0:
                        rows.append(_base(record, "end_to_end_time", float(budget), None, initializer, repair_seed, False, None, init_runtime, 0.0, init_runtime, "initializer_timeout", {"initializer_status": init_status, "timed_out": True, "iterations_completed": 0}))
                    else:
                        result = identical_repair(instance, context, state, iterations=int(config["repair"]["maximum_iterations_per_time_budget"]), random_seed=int(repair_seed), weights=weights, time_limit_s=remaining, target_score=float(targets[cell]), destroy_fractions=config["repair"]["destroy_fractions"])
                        rows.append(_repair_row(record, "end_to_end_time", float(budget), initializer, repair_seed, init_status, init_runtime, result, float(budget)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for row in rows:
            row["manifest_sha256"] = manifest_hash; row["preprocessing_artifact_sha256"] = artifact_hash
            row["repair_config_sha256"] = canonical_digest(config["repair"]); row["validation_access"] = validation_access
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _repair_row(record, view, budget, initializer, repair_seed, init_status, init_runtime, result, time_budget):
    end_to_end = init_runtime + result.repair_runtime_s
    first = None if result.first_feasible_time_s is None else init_runtime + result.first_feasible_time_s
    target = None if result.time_to_target_s is None else init_runtime + result.time_to_target_s
    detail = {
        "initializer_status": init_status, "first_feasible_iteration": result.first_feasible_iteration,
        "first_feasible_time_s": first, "time_to_target_s": target,
        "iterations_completed": result.iterations_completed, "destroy_reinsert_count": result.destroy_reinsert_count,
        "assignment_modifications": result.assignment_modifications, "order_modifications": result.order_modifications,
        "modified_atomic_units": result.modified_atomic_units, "initializer_assignment_retention": result.initializer_assignment_retention,
        "timed_out": result.timed_out, "initial_verified": result.initial_evaluation.verified,
        "initial_objective": result.initial_evaluation.objective,
    }
    verified = result.final_evaluation.verified and (time_budget is None or end_to_end <= time_budget + 0.02)
    failure = None if verified else ("repair_timeout" if time_budget is not None and end_to_end > time_budget + 0.02 else result.final_evaluation.failure_reason or result.status)
    return _base(record, view, budget, result.iterations_completed, initializer, repair_seed, verified, result.final_evaluation.objective if verified else None, init_runtime, result.repair_runtime_s, end_to_end, failure, detail)


def _base(record, view, budget, iterations, initializer, repair_seed, verified, objective, init_runtime, repair_runtime, end_runtime, failure, detail):
    return {
        "split": record["split"], "cell_id": record["cell_id"], "task_group_id": record["task_group_id"],
        "variant_index": record["variant_index"], "instance_id": record["instance_id"], "view": view,
        "budget": budget, "iterations_completed": iterations, "initializer": initializer, "model_seed": _model_seed(initializer),
        "repair_seed": repair_seed, "verified": bool(verified), "final_objective": objective,
        "initializer_runtime_s": float(init_runtime), "repair_runtime_s": float(repair_runtime), "end_to_end_runtime_s": float(end_runtime),
        "failure_reason": failure, "detail": detail,
    }


def aggregate(config, output):
    validate_seal(config, output)
    rows = []
    for cell in config["data"]["cells"]:
        path = output / "validation_shards" / f"{cell}.jsonl"
        if not path.is_file() or not (output / "validation_access" / f"{cell}.complete").is_file():
            raise RuntimeError(f"missing validation shard {cell}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    integrity = _integrity(config, rows)
    primary = [x for x in rows if x["view"] == "end_to_end_time" and x["budget"] == float(config["repair"]["primary_end_to_end_time_s"])]
    fixed = [x for x in rows if x["view"] == "fixed_iterations" and x["budget"] == float(config["repair"]["primary_fixed_iterations"])]
    raw = [x for x in rows if x["view"] == "raw"]
    summary = {"primary": _summaries(primary), "fixed_50": _summaries(fixed), "raw": _summaries(raw)}
    gates = _gates(config, primary, fixed, raw, integrity)
    decision = config["continue_gate"]["pass_class"] if all(gates.values()) else config["continue_gate"]["stop_class"]
    failure_library = [x for x in rows if not x["verified"]]
    result = {
        "version": "a4-warm-start-pilot-results-v1", "evidence_label": "SIM_GEOMETRIC_DEVELOPMENT_ONLY",
        "decision": decision, "integrity": integrity, "gates": gates, "summaries": summary,
        "row_count": len(rows), "failure_count": len(failure_library), "config_sha256": config["config_sha256"],
        "seal_sha256": json.loads((output / "seal.json").read_text())["seal_sha256"],
        "job": _job_record("aggregate"), "boundaries": config["boundaries"],
    }
    _write_json(output / "summary.json", result)
    _write_json(output / "failure_library.json", failure_library)
    _write_csv(output / "validation_rows.csv", rows)
    _write_json(output / "slurm_aggregate_record.json", _job_record("aggregate"))
    print(json.dumps(result, indent=2, sort_keys=True))


def _integrity(config, rows):
    expected_instances = int(config["data"]["validation_instances_total"])
    raw_per_instance = 10  # 3 pointer + 3 static + load + MILP + LNS; cold has no raw
    repaired_per_instance = len(config["initializers"]) * len(config["repair"]["random_seeds"]) * (len(config["repair"]["fixed_iterations"]) + len(config["repair"]["fixed_end_to_end_time_s"]))
    expected = expected_instances * (raw_per_instance - 1 + repaired_per_instance)
    groups = {x["task_group_id"] for x in rows}
    return {
        "complete_matrix": len(rows) == expected,
        "expected_rows": expected, "actual_rows": len(rows),
        "group_count": len(groups), "group_count_ok": len(groups) == int(config["data"]["validation_groups_total"]),
        "zero_forbidden_access": all(x.get("validation_access") is True for x in rows),
        "zero_mask_atomicity_failure": not any(x["failure_reason"] in {"mask_integrity_failure", "atomicity_failure"} for x in rows),
        "three_seed_pairing": {x["model_seed"] for x in rows if x["initializer"].startswith("pair_pointer") and x["view"] != "raw"} == {101, 211, 307} and {x["model_seed"] for x in rows if x["initializer"].startswith("static_seed") and x["view"] != "raw"} == {101, 211, 307},
    }


def _summaries(rows):
    result = {}
    for initializer in sorted({x["initializer"] for x in rows}):
        chosen = [x for x in rows if x["initializer"] == initializer]
        runtimes = [x["end_to_end_runtime_s"] for x in chosen]
        verified = [x for x in chosen if x["verified"]]
        result[initializer] = {
            "rows": len(chosen), "coverage": sum(x["verified"] for x in chosen) / len(chosen),
            "conditional_objective": None if not verified else float(np.mean([x["final_objective"] for x in verified])),
            "median_initializer_runtime_s": float(np.median([x["initializer_runtime_s"] for x in chosen])),
            "median_repair_runtime_s": float(np.median([x["repair_runtime_s"] for x in chosen])),
            "median_end_to_end_runtime_s": float(np.median(runtimes)), "runtime_iqr_s": [float(np.quantile(runtimes, .25)), float(np.quantile(runtimes, .75))],
            "mean_retention": _detail_mean(chosen, "initializer_assignment_retention"),
            "restricted_mean_time_to_target_s": _restricted_target(chosen),
        }
    return result


def _gates(config, primary, fixed, raw, integrity):
    one = float(config["statistics"]["overall_one_group_equivalent"])
    pointer = _family_coverage(primary, "pair_pointer_seed_")
    static = _family_coverage(primary, "static_seed_")
    pair_raw = _family_coverage(raw, "pair_pointer_seed_")
    seed_wins = sum(_coverage(primary, f"pair_pointer_seed_{s}") > _coverage(primary, f"static_seed_{s}") for s in (101,211,307))
    pair_rows = [x for x in primary if x["initializer"].startswith("pair_pointer_seed_")]
    load_rows = [x for x in primary if x["initializer"] == "hybrid_load_balanced"]
    load = _coverage(primary, "hybrid_load_balanced")
    coverage_gain = pointer - load >= one - 1e-12
    noninferior = pointer >= load - one - 1e-12
    target_advantage = _restricted_target(pair_rows) <= 0.9 * _restricted_target(load_rows)
    pair_obj = _conditional_objective(pair_rows); load_obj = _conditional_objective(load_rows)
    objective_advantage = pair_obj is not None and load_obj is not None and pair_obj <= 0.95 * load_obj
    scale_pair = _cell_family_coverage(primary, "scale", "pair_pointer_seed_")
    scale_load = _cell_family_coverage(primary, "scale", "hybrid_load_balanced")
    scale_advantage = scale_pair - scale_load >= 0.25 - 1e-12 and all(_cell_family_coverage(primary, c, "pair_pointer_seed_") >= _cell_family_coverage(primary, c, "hybrid_load_balanced") - 0.25 - 1e-12 for c in config["data"]["cells"] if c != "scale")
    retention = _detail_mean(pair_rows, "initializer_assignment_retention") or 0.0
    return {
        "integrity": all(v is True or isinstance(v, int) for k, v in integrity.items() if k not in {"expected_rows", "actual_rows", "group_count"}) and integrity["complete_matrix"] and integrity["group_count_ok"],
        "pair_repair_above_raw": pointer > pair_raw + 1e-12,
        "at_least_two_seed_wins_vs_static": seed_wins >= 2,
        "mean_pair_vs_static_at_least_one_group": pointer - static >= one - 1e-12,
        "practical_advantage_vs_load_balanced": coverage_gain or (noninferior and target_advantage) or (noninferior and objective_advantage) or scale_advantage,
        "initializer_retention_at_least_half": retention >= float(config["continue_gate"]["minimum_initializer_assignment_retention"]),
    }


def _coverage(rows, initializer):
    selected = [x for x in rows if x["initializer"] == initializer]
    return 0.0 if not selected else sum(x["verified"] for x in selected) / len(selected)


def _family_coverage(rows, prefix):
    selected = [x for x in rows if x["initializer"].startswith(prefix)]
    return 0.0 if not selected else sum(x["verified"] for x in selected) / len(selected)


def _cell_family_coverage(rows, cell, prefix):
    selected = [x for x in rows if x["cell_id"] == cell and (x["initializer"].startswith(prefix) if prefix.endswith("_") else x["initializer"] == prefix)]
    return 0.0 if not selected else sum(x["verified"] for x in selected) / len(selected)


def _restricted_target(rows):
    if not rows: return math.inf
    values = []
    for x in rows:
        budget = float(x["budget"] or max(y["end_to_end_runtime_s"] for y in rows))
        value = None if x["detail"] is None else x["detail"].get("time_to_target_s")
        values.append(min(float(value), budget) if value is not None else budget)
    return float(np.mean(values))


def _conditional_objective(rows):
    values = [float(x["final_objective"]) for x in rows if x["verified"] and x["final_objective"] is not None]
    return None if not values else float(np.mean(values))


def _detail_mean(rows, key):
    values = [float(x["detail"][key]) for x in rows if x.get("detail") and x["detail"].get(key) is not None]
    return None if not values else float(np.mean(values))


def _model_seed(name):
    try: return int(name.rsplit("_", 1)[1])
    except (ValueError, IndexError): return None


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _write_csv(path, rows):
    flat = []
    for row in rows:
        item = {k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in row.items()}; flat.append(item)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for x in flat for k in x})); writer.writeheader(); writer.writerows(flat)


def _job_record(note):
    return {"note": note, "job_id": os.environ.get("SLURM_JOB_ID"), "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "node": os.environ.get("SLURMD_NODENAME"), "command": " ".join(sys.argv), "exit_code": 0, "timestamp": time.time(), "dependencies": dependency_versions()}


if __name__ == "__main__":
    main()
