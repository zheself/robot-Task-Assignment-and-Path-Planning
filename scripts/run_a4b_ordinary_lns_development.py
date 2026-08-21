#!/usr/bin/env python3
"""Run the preregistered A4b ordinary-LNS recovery v2 protocol.

There is deliberately no neural-training command. Fixed-time and exact-
iteration evidence are produced by different search invocations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.search.alns_v2 import (
    AlnsV2Config,
    repair_destroyed_state_v2,
    run_search_v2,
)
from safe_residual_rl.allocation.search.anytime import build_hybrid_load_balanced_initializer
from safe_residual_rl.allocation.search.data_v2 import (
    generate_a4b_v2_data,
    load_a4b_v2_config,
    load_a4b_v2_items,
    sha256_file,
)
from safe_residual_rl.allocation.search.diagnostics import analyze_state, evaluate_state_timed
from safe_residual_rl.allocation.search.metrics import normalized_primal_integral, time_to_target
from safe_residual_rl.allocation.search.operators import DESTROY_OPERATORS, HANDCRAFTED_OPERATORS
from safe_residual_rl.allocation.search.operators_v2 import select_destroy_set_v2
from safe_residual_rl.allocation.search.trace import canonical_hash, state_from_dict, state_hash, state_to_dict
from safe_residual_rl.allocation.solvers.common import allocation_units, edge_mask_and_costs

CONFIG_PATH = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2.json"
AMENDMENT_PATH = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_runtime_amendment.json"
WATCHDOG_AMENDMENT_PATH = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_watchdog_amendment.json"
PARALLEL_AMENDMENT_PATH = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_parallel_amendment.json"
RECOVERY_AMENDMENT_PATH = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_metadata_recovery_amendment.json"
DEFAULT_OUTPUT = ROOT / "outputs/phase1_allocation/a4b_ordinary_lns_dev_v2"
PROCESS_STARTED_UNIX_S = time.time()
SOURCE_FILES = (
    "configs/allocation/a4b_ordinary_lns_dev_v2.json",
    "configs/allocation/a4b_ordinary_lns_dev_v2_runtime_amendment.json",
    "configs/allocation/a4b_ordinary_lns_dev_v2_watchdog_amendment.json",
    "configs/allocation/a4b_ordinary_lns_dev_v2_parallel_amendment.json",
    "configs/allocation/a4b_ordinary_lns_dev_v2_metadata_recovery_amendment.json",
    "docs/31_a4b_ordinary_lns_recovery_preregistration.md",
    "docs/33_a4b_v2_runtime_hardware_amendment.md",
    "docs/34_a4b_v2_watchdog_recovery_amendment.md",
    "docs/35_a4b_v2_cpu_parallel_execution_amendment.md",
    "docs/36_a4b_v2_calibration_metadata_recovery_amendment.md",
    "src/safe_residual_rl/allocation/search/anytime.py",
    "src/safe_residual_rl/allocation/search/diagnostics.py",
    "src/safe_residual_rl/allocation/search/operators.py",
    "src/safe_residual_rl/allocation/search/operators_v2.py",
    "src/safe_residual_rl/allocation/search/alns_v2.py",
    "src/safe_residual_rl/allocation/search/metrics.py",
    "src/safe_residual_rl/allocation/search/trace.py",
    "src/safe_residual_rl/allocation/search/data_v2.py",
    "scripts/run_a4b_ordinary_lns_development.py",
    "scripts/run_a4b_cpu_worker.sh",
    "scripts/submit_a4b_v2_parallel_chain.sh",
    "scripts/submit_a4b_v2_metadata_recovery_chain.sh",
    "tests/allocation/test_a4b_v2_recovery.py",
    "slurm/a4b_v2_train_gate.sbatch",
    "slurm/a4b_v2_calibration_array.sbatch",
    "slurm/a4b_v2_calibration_packed.sbatch",
    "slurm/a4b_v2_merge_gate_labels.sbatch",
    "slurm/a4b_v2_recover_merge_gate_labels.sbatch",
    "slurm/a4b_v2_development_smoke.sbatch",
    "slurm/a4b_v2_development_array.sbatch",
    "slurm/a4b_v2_development_packed.sbatch",
    "slurm/a4b_v2_finalize.sbatch",
)
METHODS = (
    ("random_lns", "random_lns", None),
    ("handcrafted_round_robin", "handcrafted_round_robin", None),
    ("best_single_train_selected", "single_operator", "selected"),
    ("adaptive_alns", "adaptive_alns", None),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("generate", "calibrate", "calibrate-cell", "recover-calibration-metadata", "merge-calibration", "train-gate", "labels", "smoke", "run-cell", "replay", "aggregate"),
    )
    parser.add_argument("--cell")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_a4b_v2_config(CONFIG_PATH)
    amendment = json.loads(AMENDMENT_PATH.read_text())
    if (
        amendment.get("status") != "FROZEN_BEFORE_FIRST_SEARCH_EXECUTION"
        or amendment.get("base_config_sha256") != config["config_sha256"]
        or amendment.get("scope") != "runtime_hardware_only"
        or amendment["runtime_hardware"].get("gpu_requested")
        or amendment["runtime_hardware"].get("gres_requested")
    ):
        raise RuntimeError("invalid A4b v2 runtime hardware amendment")
    config["runtime_amendment_sha256"] = sha256_file(AMENDMENT_PATH)
    config["runtime_hardware"] = amendment["runtime_hardware"]
    watchdog_amendment = json.loads(WATCHDOG_AMENDMENT_PATH.read_text())
    if (
        watchdog_amendment.get("status")
        != "FROZEN_AFTER_FAILED_TRAIN_GATE_BEFORE_RESTART"
        or watchdog_amendment.get("base_config_sha256") != config["config_sha256"]
        or watchdog_amendment.get("scope") != "operational_safety_watchdog_only"
        or watchdog_amendment.get("failed_job_id") != "981071"
        or float(watchdog_amendment["replacement"].get("search.safety_watchdog_s", 0.0))
        <= float(config["search"]["safety_watchdog_s"])
    ):
        raise RuntimeError("invalid A4b v2 watchdog recovery amendment")
    config["watchdog_amendment_sha256"] = sha256_file(WATCHDOG_AMENDMENT_PATH)
    config["search"]["safety_watchdog_s"] = float(
        watchdog_amendment["replacement"]["search.safety_watchdog_s"]
    )
    parallel_amendment = json.loads(PARALLEL_AMENDMENT_PATH.read_text())
    parallel = parallel_amendment.get("parallel_execution", {})
    expected = parallel_amendment.get("expected_calibration_matrix", {})
    if (
        parallel_amendment.get("status")
        != "FROZEN_DURING_SERIAL_CALIBRATION_BEFORE_PARALLEL_SUBMISSION"
        or parallel_amendment.get("base_config_sha256") != config["config_sha256"]
        or parallel_amendment.get("runtime_amendment_sha256")
        != config["runtime_amendment_sha256"]
        or parallel_amendment.get("watchdog_amendment_sha256")
        != config["watchdog_amendment_sha256"]
        or parallel_amendment.get("scope") != "execution_sharding_and_audit_only"
        or parallel.get("cells") != config["data"]["cells"]
        or int(parallel.get("worker_count", 0)) != len(config["data"]["cells"])
        or int(expected.get("required_completed_iterations", 0))
        != int(max(config["search"]["fixed_iterations"]))
        or float(config["search"]["safety_watchdog_s"]) < 1800.0
    ):
        raise RuntimeError("invalid A4b v2 CPU-parallel execution amendment")
    config["parallel_amendment_sha256"] = sha256_file(PARALLEL_AMENDMENT_PATH)
    config["parallel_execution"] = parallel
    config["expected_calibration_matrix"] = expected
    recovery_amendment = json.loads(RECOVERY_AMENDMENT_PATH.read_text())
    _validate_recovery_amendment(config, recovery_amendment)
    config["recovery_amendment_sha256"] = sha256_file(RECOVERY_AMENDMENT_PATH)
    config["metadata_recovery"] = recovery_amendment
    if os.environ.get("SLURM_JOB_ID"):
        _validate_worker_environment(config)
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.command == "generate":
        manifest = generate_a4b_v2_data(ROOT, config, output / "corpus", context)
        _write_json(output / "generation_record.json", _record(config, manifest_sha256=manifest["manifest_sha256"]))
    elif args.command == "calibrate":
        _write_json(output / "train_calibration.json", calibrate(config, context, output))
    elif args.command == "calibrate-cell":
        if args.cell not in config["data"]["cells"]:
            raise SystemExit("--cell must be one of the six preregistered cells")
        shard = calibrate_cell(config, context, output, args.cell)
        if not shard["passed"]:
            raise SystemExit(f"A4b v2 calibration shard failed: {args.cell}")
    elif args.command == "recover-calibration-metadata":
        recover_calibration_metadata(config, context, output)
    elif args.command == "merge-calibration":
        _write_json(output / "train_calibration.json", merge_calibration(config, context, output))
    elif args.command == "train-gate":
        gate = train_gate(config, context, output)
        _write_json(output / "train_gate.json", gate)
        if not gate["passed"]:
            raise SystemExit("A4b v2 train gate failed; labels/development remain blocked")
    elif args.command == "labels":
        generate_labels(config, context, output)
    elif args.command == "smoke":
        smoke = run_cell(config, context, output, "iid_small", limit=1, smoke=True)
        _write_json(output / "development_smoke.json", smoke)
    elif args.command == "run-cell":
        if args.cell not in config["data"]["cells"]:
            raise SystemExit("--cell must be one of the six preregistered cells")
        run_cell(config, context, output, args.cell, limit=None, smoke=False)
    elif args.command == "replay":
        _write_json(output / "replay_audit.json", replay_audit(config, context, output))
    else:
        aggregate(config, output)


def _validate_recovery_amendment(config, amendment):
    expected = amendment.get("expected_matrix", {})
    if (
        amendment.get("status")
        != "FROZEN_AFTER_JOB_984111_METADATA_ONLY_FAILURE_BEFORE_RECOVERY"
        or amendment.get("scope") != "metadata_reconstruction_only_no_search"
        or amendment.get("formal_action") != "HOLD_A4B_LEARNED_DESTROY_TRAINING"
        or amendment.get("base_config_sha256") != config["config_sha256"]
        or amendment.get("runtime_amendment_sha256") != config["runtime_amendment_sha256"]
        or amendment.get("watchdog_amendment_sha256") != config["watchdog_amendment_sha256"]
        or amendment.get("parallel_amendment_sha256") != config["parallel_amendment_sha256"]
        or any(
            expected.get(key) != value
            for key, value in config["expected_calibration_matrix"].items()
        )
        or amendment.get("failed_job", {}).get("job_id") != "984111"
        or amendment.get("failed_job", {}).get("state") != "FAILED"
        or amendment.get("recovery_rules", {}).get("search_execution_allowed") is not False
        or amendment.get("recovery_rules", {}).get("trace_rewrite_allowed") is not False
    ):
        raise RuntimeError("invalid A4b v2 calibration metadata-recovery amendment")


def _config(config, seed, budget_mode, *, update_scheme="segmented", iterations=None, time_s=None):
    search, alns = config["search"], config["alns"]
    return AlnsV2Config(
        protocol_id=config["protocol_id"],
        budget_mode=budget_mode,
        iterations=int(max(search["fixed_iterations"]) if iterations is None else iterations),
        end_to_end_time_s=(None if budget_mode == "fixed_iterations" else float(max(search["fixed_end_to_end_time_s"]) if time_s is None else time_s)),
        safety_watchdog_s=float(search["safety_watchdog_s"]),
        destroy_ratios=tuple(float(item) for item in search["destroy_ratios"]),
        repair_candidate_evaluation_budget=int(search["repair_candidate_evaluation_budget"]),
        random_seed=int(seed),
        objective_weights=search["objective_weights"],
        update_scheme=update_scheme,
        segment_length=int(alns["segment_length"]),
        reaction_factor=float(alns["reaction_factor"]),
        initial_temperature_fraction=float(alns["initial_temperature_fraction"]),
        cooling_rate=float(alns["cooling_rate"]),
        reward_global_best=float(alns["rewards"]["global_best"]),
        reward_new_feasible=float(alns["rewards"]["new_feasible"]),
        reward_strict_improvement=float(alns["rewards"]["strict_improvement"]),
        reward_unseen_diversification=float(alns["rewards"]["unseen_diversification"]),
        restart_no_improvement=int(search["restart_no_improvement"]),
        allow_infeasible_current=bool(search["allow_infeasible_current"]),
    )


def _run(instance, context, config, record, method, mode, single, seed, budget_mode, *, update_scheme, iterations=None, time_s=None):
    initializer = build_hybrid_load_balanced_initializer(instance, context, config["search"]["objective_weights"])
    outcome = run_search_v2(
        instance,
        context,
        initializer,
        _config(config, seed, budget_mode, update_scheme=update_scheme, iterations=iterations, time_s=time_s),
        mode=mode,
        single_operator=single,
        task_group_id=record["task_group_id"],
        difficulty=record["cell_id"],
        split=record["split"],
    )
    trace = outcome.trace
    trace["method_id"] = method
    trace["trace_sha256"] = canonical_hash({k: v for k, v in trace.items() if k != "trace_sha256"})
    return trace


def _best(events, *, iteration=None, cutoff=None):
    eligible = []
    for event in events:
        if iteration is not None and int(event["iteration"]) > iteration:
            continue
        if cutoff is not None and float(event["elapsed_s"]) > cutoff + 1e-12:
            continue
        eligible.append(event)
    return None if not eligible else min(eligible, key=lambda item: (float(item["objective"]), float(item["elapsed_s"]), int(item["iteration"])))


def _failure_reason(trace, *, cutoff=None, iteration=None):
    steps = [item for item in trace["steps"] if item["completed_before_cutoff"]]
    if cutoff is not None:
        steps = [item for item in steps if float(item["elapsed_s"]) <= cutoff + 1e-12]
    if iteration is not None:
        steps = [item for item in steps if int(item["iteration"]) <= iteration]
    if steps:
        return steps[-1]["after_failure_reason"] or "no_feasible_incumbent"
    provenance = trace["initializer"]
    if cutoff is not None and float(provenance["completion_elapsed_s"]) > cutoff + 1e-12:
        return "initializer_timeout"
    return provenance.get("verifier_failure_reason") or "no_feasible_incumbent"


def _identity(record, method, seed):
    return {
        "split": record["split"], "cell_id": record["cell_id"],
        "task_group_id": record["task_group_id"], "variant_index": record["variant_index"],
        "instance_id": record["instance_id"], "method": method, "search_seed": int(seed),
    }


def _fixed_iteration_rows(config, record, method, seed, trace):
    rows = []
    for budget in config["search"]["fixed_iterations"]:
        if not trace["fixed_iteration_complete"] or int(trace["iterations_completed"]) < int(budget):
            continue
        event = _best(trace["incumbents"], iteration=int(budget))
        rows.append(
            {
                **_identity(record, method, seed), "view": "fixed_iterations", "budget": int(budget),
                "verified": event is not None, "objective": None if event is None else event["objective"],
                "incumbent_elapsed_s": None if event is None else event["elapsed_s"],
                "failure_reason": None if event is not None else _failure_reason(trace, iteration=int(budget)),
                "exact_iteration_evidence": True, "iterations_completed": trace["iterations_completed"],
                "trace_sha256": trace["trace_sha256"], "initializer_actual": trace["initializer"]["actual_initializer"],
                "fallback_used": trace["initializer"]["fallback_used"],
            }
        )
    return rows


def _fixed_time_rows(config, record, method, seed, trace, calibration):
    rows = []
    cell_metric = calibration["metric_references"][record["cell_id"]]
    for budget in config["search"]["fixed_end_to_end_time_s"]:
        event = _best(trace["incumbents"], cutoff=float(budget))
        rows.append(
            {
                **_identity(record, method, seed), "view": "fixed_end_to_end_time", "budget": float(budget),
                "verified": event is not None, "objective": None if event is None else event["objective"],
                "incumbent_elapsed_s": None if event is None else event["elapsed_s"],
                "failure_reason": None if event is not None else _failure_reason(trace, cutoff=float(budget)),
                "normalized_primal_integral": normalized_primal_integral(
                    trace["incumbents"], float(budget), target=cell_metric["target"], reference=cell_metric["reference"]
                ),
                "time_to_target_s": time_to_target(trace["incumbents"], float(budget), target=cell_metric["target"]),
                "trace_sha256": trace["trace_sha256"], "initializer_actual": trace["initializer"]["actual_initializer"],
                "fallback_used": trace["initializer"]["fallback_used"],
                "initializer_runtime_s": trace["initializer"]["completion_elapsed_s"],
                "cutoff_overrun_s": trace["cutoff_overrun_s"],
            }
        )
    return rows


def _calibration_items(config, context, output):
    items, manifest_hash = load_a4b_v2_items(output / "corpus", "train", context)
    selected = []
    limit = int(config["data"]["operator_selection_groups_per_cell"])
    for cell in config["data"]["cells"]:
        groups = sorted({r["task_group_id"] for r, _ in items if r["cell_id"] == cell})[:limit]
        selected.extend((r, i) for r, i in items if r["task_group_id"] in groups)
    return selected, manifest_hash


def _run_calibration_items(config, context, items):
    seed = int(config["search"]["random_seeds"][0])
    rows, traces = [], []
    for record, instance in items:
        for operator in HANDCRAFTED_OPERATORS:
            trace = _run(instance, context, config, record, operator, "single_operator", operator, seed, "fixed_iterations", update_scheme="segmented")
            traces.append(trace)
            rows.extend(_fixed_iteration_rows(config, record, operator, seed, trace))
        for scheme in config["alns"]["candidate_update_schemes"]:
            method = f"adaptive_alns_{scheme}"
            for budget_mode in ("fixed_iterations", "fixed_time"):
                trace = _run(instance, context, config, record, method, "adaptive_alns", None, seed, budget_mode, update_scheme=scheme, time_s=1.0)
                traces.append(trace)
                if budget_mode == "fixed_iterations":
                    rows.extend(_fixed_iteration_rows(config, record, method, seed, trace))
                else:
                    event = _best(trace["incumbents"], cutoff=1.0)
                    rows.append({**_identity(record, method, seed), "view": "fixed_end_to_end_time", "budget": 1.0, "verified": event is not None, "objective": None if event is None else event["objective"], "failure_reason": None if event else _failure_reason(trace, cutoff=1.0), "trace_sha256": trace["trace_sha256"]})
        for budget_mode in ("fixed_iterations", "fixed_time"):
            trace = _run(instance, context, config, record, "random_lns", "random_lns", None, seed, budget_mode, update_scheme="segmented", time_s=1.0)
            traces.append(trace)
            if budget_mode == "fixed_iterations":
                rows.extend(_fixed_iteration_rows(config, record, "random_lns", seed, trace))
            else:
                event = _best(trace["incumbents"], cutoff=1.0)
                rows.append({**_identity(record, "random_lns", seed), "view": "fixed_end_to_end_time", "budget": 1.0, "verified": event is not None, "objective": None if event is None else event["objective"], "failure_reason": None if event else _failure_reason(trace, cutoff=1.0), "trace_sha256": trace["trace_sha256"]})

    return rows, traces


def _reconstruct_calibration_rows(config, records_by_instance, traces):
    rows = []
    for trace in traces:
        instance_id = trace["instance_id"]
        if instance_id not in records_by_instance:
            raise RuntimeError(f"trace instance is outside frozen train calibration: {instance_id}")
        record = records_by_instance[instance_id]
        method = trace["method_id"]
        seed = int(trace["random_seed"])
        if (
            trace.get("split") != "train"
            or trace.get("difficulty") != record["cell_id"]
        ):
            raise RuntimeError(f"trace split/cell provenance mismatch: {instance_id}")
        if trace["budget_mode"] == "fixed_iterations":
            rows.extend(_fixed_iteration_rows(config, record, method, seed, trace))
        elif trace["budget_mode"] == "fixed_time":
            event = _best(trace["incumbents"], cutoff=1.0)
            rows.append(
                {
                    **_identity(record, method, seed),
                    "view": "fixed_end_to_end_time",
                    "budget": 1.0,
                    "verified": event is not None,
                    "objective": None if event is None else event["objective"],
                    "failure_reason": None if event else _failure_reason(trace, cutoff=1.0),
                    "trace_sha256": trace["trace_sha256"],
                }
            )
        else:
            raise RuntimeError(f"unknown preserved calibration budget mode: {trace['budget_mode']}")
    return sorted(rows, key=lambda row: _calibration_row_sort_key(config, row))


def _calibration_method_order(config):
    return (
        *HANDCRAFTED_OPERATORS,
        *(f"adaptive_alns_{scheme}" for scheme in config["alns"]["candidate_update_schemes"]),
        "random_lns",
    )


def _trace_identity(trace):
    return (
        trace["instance_id"], trace["method_id"], int(trace["random_seed"]),
        trace["budget_mode"],
    )


def _trace_transition_signature(trace):
    return tuple(
        (
            int(step["iteration"]), step["operator"], tuple(step["destroy_set"]),
            step["candidate_state_sha256"], bool(step["accepted"]),
            bool(step["incumbent_updated"]),
        )
        for step in trace["steps"]
    )


def _calibration_matrix_status(config, traces, *, cell=None):
    expected = config["expected_calibration_matrix"]
    fixed = [trace for trace in traces if trace["budget_mode"] == "fixed_iterations"]
    timed = [trace for trace in traces if trace["budget_mode"] == "fixed_time"]
    identities = [_trace_identity(trace) for trace in traces]
    if cell is None:
        expected_fixed = int(expected["fixed_iteration_traces_total"])
        expected_timed = int(expected["fixed_time_traces_total"])
        expected_total = int(expected["traces_total"])
    else:
        expected_fixed = int(expected["fixed_iteration_traces_per_cell"])
        expected_timed = int(expected["fixed_time_traces_per_cell"])
        expected_total = int(expected["traces_per_cell"])
    complete = sum(
        bool(trace["fixed_iteration_complete"])
        and int(trace["iterations_completed"]) == int(expected["required_completed_iterations"])
        for trace in fixed
    )
    passed = (
        len(traces) == expected_total
        and len(fixed) == expected_fixed
        and len(timed) == expected_timed
        and len(identities) == len(set(identities))
        and complete == expected_fixed
        and (cell is None or all(trace.get("difficulty") == cell for trace in traces))
    )
    return {
        "traces": len(traces), "fixed_iterations": len(fixed),
        "fixed_time": len(timed), "fixed_iteration_complete": complete,
        "unique_identities": len(set(identities)), "passed": passed,
    }


def _worker_affinity_matrix_status(records, expected_workers):
    affinities = [tuple(int(cpu) for cpu in record.get("cpu_affinity", ())) for record in records]
    return {
        "workers": len(records),
        "single_cpu_workers": sum(len(affinity) == 1 for affinity in affinities),
        "unique_worker_cpus": len(set(affinities)),
        "worker_cpus": [affinity[0] if len(affinity) == 1 else None for affinity in affinities],
        "passed": (
            len(records) == int(expected_workers)
            and all(len(affinity) == 1 for affinity in affinities)
            and len(set(affinities)) == int(expected_workers)
        ),
    }


def _calibration_trace_sort_key(config, trace):
    cells = {cell: index for index, cell in enumerate(config["data"]["cells"])}
    methods = {method: index for index, method in enumerate(_calibration_method_order(config))}
    modes = {"fixed_iterations": 0, "fixed_time": 1}
    return (
        cells[trace["difficulty"]], trace["instance_id"], methods[trace["method_id"]],
        modes[trace["budget_mode"]], int(trace["random_seed"]),
    )


def _calibration_row_sort_key(config, row):
    cells = {cell: index for index, cell in enumerate(config["data"]["cells"])}
    methods = {method: index for index, method in enumerate(_calibration_method_order(config))}
    views = {"fixed_iterations": 0, "fixed_end_to_end_time": 1}
    return (
        cells[row["cell_id"]], row["instance_id"], methods[row["method"]],
        views[row["view"]], float(row["budget"]), int(row["search_seed"]),
    )


def _build_calibration_payload(config, rows, traces, manifest_hash, *, merge_sha256=None):
    rows = sorted(rows, key=lambda row: _calibration_row_sort_key(config, row))
    traces = sorted(traces, key=lambda trace: _calibration_trace_sort_key(config, trace))

    summaries = _summaries(rows)
    operators = list(HANDCRAFTED_OPERATORS)
    selected_operator = min(operators, key=lambda op: _selection_key(rows, op))
    schemes = list(config["alns"]["candidate_update_schemes"])
    selected_scheme = min(schemes, key=lambda scheme: _scheme_key(rows, f"adaptive_alns_{scheme}"))
    references = {}
    for cell in config["data"]["cells"]:
        objectives = [float(e["objective"]) for t in traces if t["difficulty"] == cell for e in t["incumbents"]]
        if not objectives:
            raise RuntimeError(f"no verified train objective for metric reference: {cell}")
        target = float(statistics.median(objectives))
        reference = max(max(objectives), target * 1.25, target + 1e-9)
        references[cell] = {"target": target, "reference": reference, "source": "train_calibration_only"}
    payload = {
        "version": "a4b-v2-train-calibration-v1", "selection_split": "train",
        "future_data_accessed": False, "selected_operator": selected_operator,
        "selected_update_scheme": selected_scheme, "metric_references": references,
        "summaries": summaries, "rows": rows, "manifest_sha256": manifest_hash,
        "config_sha256": config["config_sha256"],
        "runtime_amendment_sha256": config["runtime_amendment_sha256"],
        "watchdog_amendment_sha256": config["watchdog_amendment_sha256"],
        "parallel_amendment_sha256": config["parallel_amendment_sha256"],
        "merge_sha256": merge_sha256,
    }
    payload["calibration_sha256"] = canonical_hash(payload)
    return payload, traces


def calibrate(config, context, output):
    items, manifest_hash = _calibration_items(config, context, output)
    rows, traces = _run_calibration_items(config, context, items)
    payload, traces = _build_calibration_payload(config, rows, traces, manifest_hash)
    _write_jsonl(output / "train_calibration_traces.jsonl", traces)
    return payload


def calibrate_cell(config, context, output, cell):
    items, manifest_hash = _calibration_items(config, context, output)
    selected = [(record, instance) for record, instance in items if record["cell_id"] == cell]
    rows, traces = _run_calibration_items(config, context, selected)
    rows = sorted(rows, key=lambda row: _calibration_row_sort_key(config, row))
    traces = sorted(traces, key=lambda trace: _calibration_trace_sort_key(config, trace))
    root = output / "train_calibration_shards"
    trace_path = root / f"{cell}.jsonl"
    _write_jsonl(trace_path, traces)
    expected = config["expected_calibration_matrix"]
    status = _calibration_matrix_status(config, traces, cell=cell)
    payload = {
        "version": "a4b-v2-calibration-cell-shard-v1", "cell": cell,
        "passed": (
            len(selected) == int(expected["instances_per_cell"])
            and status["passed"]
        ),
        "instance_ids": sorted(record["instance_id"] for record, _ in selected),
        "counts": {
            "instances": len(selected), **status, "rows": len(rows),
        },
        "rows": rows, "rows_sha256": canonical_hash(rows),
        "trace_file": trace_path.name, "trace_file_sha256": sha256_file(trace_path),
        "manifest_sha256": manifest_hash, "config_sha256": config["config_sha256"],
        "runtime_amendment_sha256": config["runtime_amendment_sha256"],
        "watchdog_amendment_sha256": config["watchdog_amendment_sha256"],
        "parallel_amendment_sha256": config["parallel_amendment_sha256"],
        "source_hashes": _source_hashes(), "record": _record(config, cell=cell),
    }
    payload["shard_sha256"] = canonical_hash(payload)
    _write_json(root / f"{cell}.json", payload)
    return payload


def _recovered_worker_record(config, cell, traces, artifact, log_text):
    start_match = re.search(r"^worker_start_utc=(\S+)$", log_text, re.MULTILINE)
    load_match = re.search(r"^worker_loadavg=(.+)$", log_text, re.MULTILINE)
    cpu = int(artifact["cpu"])
    trace_start = min(int(trace["start_monotonic_ns"]) for trace in traces)
    trace_end = max(int(trace["return_monotonic_ns"]) for trace in traces)
    record = {
        "record_type": "recovered_failed_worker_provenance_v1",
        "recovery_status": "RECONSTRUCTED_NOT_NATIVE",
        "protocol_id": config["protocol_id"],
        "cell": cell,
        "original_job_id": config["metadata_recovery"]["failed_job"]["job_id"],
        "original_job_state": "FAILED",
        "original_job_exit_code": config["metadata_recovery"]["failed_job"]["exit_code"],
        "node": config["metadata_recovery"]["failed_job"]["node"],
        "cpu_affinity": [cpu],
        "cpu_affinity_count": 1,
        "worker_start_log_value": None if start_match is None else start_match.group(1),
        "worker_start_loadavg_log_value": None if load_match is None else load_match.group(1),
        "trace_start_monotonic_ns": trace_start,
        "trace_end_monotonic_ns": trace_end,
        "trace_wall_s": (trace_end - trace_start) / 1e9,
        "worker_ended_unix_s": None,
        "worker_end_loadavg": None,
        "unavailable_original_fields": [
            "worker_ended_unix_s",
            "worker_end_loadavg",
            "numpy_version",
        ],
        "environment": {
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "openblas_num_threads": "1",
            "numexpr_num_threads": "1",
            "cuda_visible_devices": "",
        },
        "environment_evidence": "frozen original worker-wrapper hash and preserved launch log",
        "trace_file_sha256": artifact["trace_sha256"],
        "stderr_log_sha256": artifact["log_sha256"],
        "execution_runner_sha256": config["metadata_recovery"]["failed_job"]["runner_sha256"],
        "execution_worker_wrapper_sha256": config["metadata_recovery"]["failed_job"]["worker_wrapper_sha256"],
        "recovery_amendment_sha256": config["recovery_amendment_sha256"],
    }
    record["record_sha256"] = canonical_hash(record)
    return record


def recover_calibration_metadata(config, context, output):
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("metadata recovery is restricted to the frozen A4b v2 output namespace")
    amendment = config["metadata_recovery"]
    root = output / "train_calibration_shards"
    cells = list(config["data"]["cells"])
    if sorted(path.stem for path in root.glob("*.json")):
        raise RuntimeError("metadata recovery requires no pre-existing cell JSON metadata")
    if sorted(path.stem for path in root.glob("*.jsonl")) != sorted(cells):
        raise RuntimeError("metadata recovery requires exactly the six frozen JSONL shards")
    manifest_path = output / "corpus/manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    if (
        sha256_file(manifest_path) != amendment["corpus_manifest_file_sha256"]
        or manifest_payload.get("manifest_sha256")
        != amendment["corpus_manifest_semantic_sha256"]
    ):
        raise RuntimeError("metadata recovery corpus manifest mismatch")

    items, manifest_hash = _calibration_items(config, context, output)
    records_by_instance = {record["instance_id"]: record for record, _ in items}
    active_recovery_hashes = _source_hashes()
    prepared = {}
    all_traces = []
    all_rows = []
    for cell in cells:
        artifact = amendment["artifacts"][cell]
        trace_path = (ROOT / artifact["trace_path"]).resolve()
        log_path = (ROOT / artifact["log_path"]).resolve()
        if trace_path != (root / f"{cell}.jsonl").resolve():
            raise RuntimeError(f"metadata recovery trace path mismatch: {cell}")
        if (
            sha256_file(trace_path) != artifact["trace_sha256"]
            or sha256_file(log_path) != artifact["log_sha256"]
        ):
            raise RuntimeError(f"metadata recovery artifact hash mismatch: {cell}")
        log_text = log_path.read_text()
        cpu = int(artifact["cpu"])
        if (
            f"worker_node={amendment['failed_job']['node']} slurm_job_id={amendment['failed_job']['job_id']}" not in log_text
            or f"worker_bound_cpu={cpu}" not in log_text
            or amendment["failed_job"]["failure_signature"] not in log_text
        ):
            raise RuntimeError(f"metadata recovery worker log evidence mismatch: {cell}")
        traces = _read_jsonl(trace_path)
        status = _calibration_matrix_status(config, traces, cell=cell)
        if (
            not status["passed"]
            or any(
                canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"})
                != trace["trace_sha256"]
                for trace in traces
            )
            or any(
                trace["cutoff_monotonic_ns"] - trace["start_monotonic_ns"]
                != 1_000_000_000
                for trace in traces
                if trace["budget_mode"] == "fixed_time"
            )
        ):
            raise RuntimeError(f"metadata recovery trace completeness failure: {cell}")
        rows = _reconstruct_calibration_rows(config, records_by_instance, traces)
        expected_instances = sorted(
            record["instance_id"] for record, _ in items if record["cell_id"] == cell
        )
        record = _recovered_worker_record(config, cell, traces, artifact, log_text)
        payload = {
            "version": "a4b-v2-recovered-calibration-cell-shard-v1",
            "cell": cell,
            "passed": True,
            "recovery_status": "METADATA_RECONSTRUCTED_WITHOUT_SEARCH",
            "instance_ids": expected_instances,
            "counts": {"instances": len(expected_instances), **status, "rows": len(rows)},
            "rows": rows,
            "rows_sha256": canonical_hash(rows),
            "trace_file": trace_path.name,
            "trace_file_sha256": artifact["trace_sha256"],
            "manifest_sha256": manifest_hash,
            "config_sha256": config["config_sha256"],
            "runtime_amendment_sha256": config["runtime_amendment_sha256"],
            "watchdog_amendment_sha256": config["watchdog_amendment_sha256"],
            "parallel_amendment_sha256": config["parallel_amendment_sha256"],
            "recovery_amendment_sha256": config["recovery_amendment_sha256"],
            "execution_source_hashes": amendment["execution_source_hashes"],
            "recovery_source_hashes": active_recovery_hashes,
            "record": record,
        }
        payload["shard_sha256"] = canonical_hash(payload)
        prepared[cell] = payload
        all_traces.extend(traces)
        all_rows.extend(rows)

    ordered_traces = sorted(
        all_traces, key=lambda trace: _calibration_trace_sort_key(config, trace)
    )
    matrix = _calibration_matrix_status(config, ordered_traces)
    transition_signature = canonical_hash(
        [
            (_trace_identity(trace), _trace_transition_signature(trace))
            for trace in ordered_traces
        ]
    )
    expected = amendment["expected_matrix"]
    row_ids = [
        (row["instance_id"], row["method"], int(row["search_seed"]), row["view"], float(row["budget"]))
        for row in all_rows
    ]
    if (
        not matrix["passed"]
        or transition_signature != expected["transition_signature_sha256"]
        or len(all_rows) != int(expected["reconstructed_rows_total"])
        or len(row_ids) != len(set(row_ids))
    ):
        raise RuntimeError("metadata recovery global matrix/signature failure")

    for cell in cells:
        _write_json(root / f"{cell}.json", prepared[cell])
    recovery_record = {
        "version": "a4b-v2-calibration-metadata-recovery-v1",
        "status": "RECOVERED_WITHOUT_SEARCH",
        "original_job_id": amendment["failed_job"]["job_id"],
        "cells": cells,
        "trace_count": len(ordered_traces),
        "row_count": len(all_rows),
        "matrix": matrix,
        "transition_signature_sha256": transition_signature,
        "trace_artifact_sha256": {
            cell: amendment["artifacts"][cell]["trace_sha256"] for cell in cells
        },
        "stderr_artifact_sha256": {
            cell: amendment["artifacts"][cell]["log_sha256"] for cell in cells
        },
        "execution_source_hashes": amendment["execution_source_hashes"],
        "recovery_source_hashes": active_recovery_hashes,
        "recovery_amendment_sha256": config["recovery_amendment_sha256"],
        "record": _record(config),
    }
    recovery_record["recovery_sha256"] = canonical_hash(recovery_record)
    _write_json(output / "calibration_metadata_recovery_record.json", recovery_record)
    return recovery_record


def _validate_calibration_shard(config, context, output, cell, payload, traces):
    expected = config["expected_calibration_matrix"]
    saved_hash = payload.get("shard_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "shard_sha256"}
    if canonical_hash(unsigned) != saved_hash or payload.get("cell") != cell or not payload.get("passed"):
        raise RuntimeError(f"invalid or failed calibration shard: {cell}")
    trace_path = output / "train_calibration_shards" / f"{cell}.jsonl"
    if payload.get("trace_file_sha256") != sha256_file(trace_path):
        raise RuntimeError(f"calibration trace file hash mismatch: {cell}")
    active_hashes = _source_hashes()
    recovered = payload.get("version") == "a4b-v2-recovered-calibration-cell-shard-v1"
    required_hashes = {
        "manifest_sha256": json.loads((output / "corpus/manifest.json").read_text())["manifest_sha256"],
        "config_sha256": config["config_sha256"],
        "runtime_amendment_sha256": config["runtime_amendment_sha256"],
        "watchdog_amendment_sha256": config["watchdog_amendment_sha256"],
        "parallel_amendment_sha256": config["parallel_amendment_sha256"],
    }
    if any(payload.get(key) != value for key, value in required_hashes.items()):
        raise RuntimeError(f"calibration shard provenance mismatch: {cell}")
    if recovered:
        amendment = config["metadata_recovery"]
        if (
            payload.get("recovery_status") != "METADATA_RECONSTRUCTED_WITHOUT_SEARCH"
            or payload.get("recovery_amendment_sha256") != config["recovery_amendment_sha256"]
            or payload.get("execution_source_hashes") != amendment["execution_source_hashes"]
            or payload.get("recovery_source_hashes") != active_hashes
            or payload.get("trace_file_sha256") != amendment["artifacts"][cell]["trace_sha256"]
        ):
            raise RuntimeError(f"recovered calibration shard provenance mismatch: {cell}")
    elif payload.get("source_hashes") != active_hashes:
        raise RuntimeError(f"calibration shard source mismatch: {cell}")
    items, manifest_hash = _calibration_items(config, context, output)
    expected_instances = sorted(record["instance_id"] for record, _ in items if record["cell_id"] == cell)
    if payload.get("manifest_sha256") != manifest_hash or payload.get("instance_ids") != expected_instances:
        raise RuntimeError(f"calibration shard instance matrix mismatch: {cell}")
    fixed = [trace for trace in traces if trace["budget_mode"] == "fixed_iterations"]
    timed = [trace for trace in traces if trace["budget_mode"] == "fixed_time"]
    status = _calibration_matrix_status(config, traces, cell=cell)
    if (
        len(expected_instances) != int(expected["instances_per_cell"])
        or not status["passed"]
        or any(trace["difficulty"] != cell for trace in traces)
        or any(canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"}) != trace["trace_sha256"] for trace in traces)
        or any(not trace["fixed_iteration_complete"] or int(trace["iterations_completed"]) != int(expected["required_completed_iterations"]) for trace in fixed)
        or any(trace["cutoff_monotonic_ns"] - trace["start_monotonic_ns"] != 1_000_000_000 for trace in timed)
    ):
        raise RuntimeError(f"calibration shard trace completeness failure: {cell}")
    rows = sorted(payload["rows"], key=lambda row: _calibration_row_sort_key(config, row))
    row_ids = [
        (row["instance_id"], row["method"], int(row["search_seed"]), row["view"], float(row["budget"]))
        for row in rows
    ]
    if canonical_hash(rows) != payload.get("rows_sha256") or len(row_ids) != len(set(row_ids)):
        raise RuntimeError(f"calibration shard row integrity failure: {cell}")
    record = payload.get("record", {})
    environment = record.get("environment" if recovered else "dependencies", {})
    if (
        int(record.get("cpu_affinity_count", 0)) != 1
        or environment.get("omp_num_threads") != "1"
        or environment.get("mkl_num_threads") != "1"
        or environment.get("cuda_visible_devices") != ""
        or (
            recovered
            and (
                record.get("record_type") != "recovered_failed_worker_provenance_v1"
                or record.get("recovery_status") != "RECONSTRUCTED_NOT_NATIVE"
                or record.get("original_job_id") != config["metadata_recovery"]["failed_job"]["job_id"]
                or record.get("worker_ended_unix_s") is not None
                or record.get("worker_end_loadavg") is not None
            )
        )
    ):
        raise RuntimeError(f"calibration shard worker audit failure: {cell}")


def merge_calibration(config, context, output):
    root = output / "train_calibration_shards"
    expected_cells = list(config["data"]["cells"])
    actual_json = sorted(path.stem for path in root.glob("*.json"))
    actual_jsonl = sorted(path.stem for path in root.glob("*.jsonl"))
    if actual_json != sorted(expected_cells) or actual_jsonl != sorted(expected_cells):
        raise RuntimeError("missing duplicate or foreign calibration shard")
    recovery_record_path = output / "calibration_metadata_recovery_record.json"
    if not recovery_record_path.exists():
        raise RuntimeError("merge requires the frozen metadata-recovery record")
    recovery_record = json.loads(recovery_record_path.read_text())
    if (
        recovery_record.get("status") != "RECOVERED_WITHOUT_SEARCH"
        or canonical_hash(
            {key: value for key, value in recovery_record.items() if key != "recovery_sha256"}
        )
        != recovery_record.get("recovery_sha256")
        or recovery_record.get("recovery_amendment_sha256")
        != config["recovery_amendment_sha256"]
        or recovery_record.get("recovery_source_hashes") != _source_hashes()
    ):
        raise RuntimeError("merge metadata-recovery record is invalid")
    rows, traces, shard_hashes, worker_records = [], [], {}, []
    manifest_hash = None
    for cell in expected_cells:
        payload = json.loads((root / f"{cell}.json").read_text())
        cell_traces = _read_jsonl(root / f"{cell}.jsonl")
        _validate_calibration_shard(config, context, output, cell, payload, cell_traces)
        rows.extend(payload["rows"])
        traces.extend(cell_traces)
        shard_hashes[cell] = payload["shard_sha256"]
        worker_records.append(payload["record"])
        manifest_hash = payload["manifest_sha256"] if manifest_hash is None else manifest_hash
        if payload["manifest_sha256"] != manifest_hash:
            raise RuntimeError("calibration shard manifest mismatch")
    identities = [_trace_identity(trace) for trace in traces]
    fixed = [trace for trace in traces if trace["budget_mode"] == "fixed_iterations"]
    timed = [trace for trace in traces if trace["budget_mode"] == "fixed_time"]
    status = _calibration_matrix_status(config, traces)
    affinity_status = _worker_affinity_matrix_status(
        worker_records, config["parallel_execution"]["worker_count"]
    )
    if not status["passed"] or not affinity_status["passed"]:
        raise RuntimeError("merged calibration matrix is incomplete or duplicated")
    traces = sorted(traces, key=lambda trace: _calibration_trace_sort_key(config, trace))
    rows = sorted(rows, key=lambda row: _calibration_row_sort_key(config, row))
    merge_record = {
        "version": "a4b-v2-calibration-merge-v1", "cells": expected_cells,
        "shard_sha256": shard_hashes, "trace_count": len(traces),
        "fixed_iteration_complete": sum(trace["fixed_iteration_complete"] for trace in fixed),
        "fixed_time_count": len(timed), "trace_identity_sha256": canonical_hash(identities),
        "worker_affinity_matrix": affinity_status,
        "transition_signature_sha256": canonical_hash([
            (_trace_identity(trace), _trace_transition_signature(trace)) for trace in traces
        ]),
        "manifest_sha256": manifest_hash, "source_hashes": _source_hashes(),
        "metadata_recovery_sha256": recovery_record["recovery_sha256"],
        "recovery_amendment_sha256": config["recovery_amendment_sha256"],
        "execution_source_hashes": config["metadata_recovery"]["execution_source_hashes"],
        "record": _record(config),
    }
    merge_record["merge_sha256"] = canonical_hash(merge_record)
    _write_jsonl(output / "train_calibration_traces.jsonl", traces)
    _write_json(output / "train_calibration_merge_record.json", merge_record)
    payload, _ = _build_calibration_payload(
        config, rows, traces, manifest_hash, merge_sha256=merge_record["merge_sha256"]
    )
    return payload


def _selection_key(rows, method):
    selected = [r for r in rows if r["method"] == method and r["view"] == "fixed_iterations" and r["budget"] == 10]
    coverage = sum(r["verified"] for r in selected) / max(len(selected), 1)
    values = [float(r["objective"]) for r in selected if r["verified"]]
    return (-coverage, math.inf if not values else float(np.mean(values)), method)


def _scheme_key(rows, method):
    exact = [r for r in rows if r["method"] == method and r["view"] == "fixed_iterations" and r["budget"] == 30]
    timed = [r for r in rows if r["method"] == method and r["view"] == "fixed_end_to_end_time"]
    exact_cov = sum(r["verified"] for r in exact) / max(len(exact), 1)
    time_cov = sum(r["verified"] for r in timed) / max(len(timed), 1)
    return (-exact_cov, -time_cov, method)


def train_gate(config, context, output):
    calibration = json.loads((output / "train_calibration.json").read_text())
    calibration_hash = calibration.get("calibration_sha256")
    if canonical_hash({key: value for key, value in calibration.items() if key != "calibration_sha256"}) != calibration_hash:
        raise RuntimeError("merged calibration content hash mismatch")
    merge = json.loads((output / "train_calibration_merge_record.json").read_text())
    if (
        canonical_hash({key: value for key, value in merge.items() if key != "merge_sha256"})
        != merge.get("merge_sha256")
        or calibration.get("merge_sha256") != merge.get("merge_sha256")
    ):
        raise RuntimeError("train gate requires a valid complete calibration merge")
    rows = calibration["rows"]
    scheme_method = f"adaptive_alns_{calibration['selected_update_scheme']}"
    gates = {}
    for view, budget, name in (("fixed_iterations", 30, "exact_30"), ("fixed_end_to_end_time", 1.0, "one_second")):
        random_rows = [r for r in rows if r["method"] == "random_lns" and r["view"] == view and r["budget"] == budget]
        alns_rows = [r for r in rows if r["method"] == scheme_method and r["view"] == view and r["budget"] == budget]
        random_cov = sum(r["verified"] for r in random_rows) / max(len(random_rows), 1)
        alns_cov = sum(r["verified"] for r in alns_rows) / max(len(alns_rows), 1)
        gates[name] = {"random_coverage": random_cov, "alns_coverage": alns_cov, "passed": bool(alns_cov + 1e-12 >= random_cov), "rows_per_method": len(random_rows)}
    traces = _read_jsonl(output / "train_calibration_traces.jsonl")
    recovery = sum(
        not t["initializer"]["verifier_feasible"] and bool(t["incumbents"])
        for t in traces if t["method_id"].startswith("adaptive_alns") or t["method_id"] == "random_lns"
    )
    incomplete_exact = sum(t["budget_mode"] == "fixed_iterations" and not t["fixed_iteration_complete"] for t in traces)
    fixed_traces = [trace for trace in traces if trace["budget_mode"] == "fixed_iterations"]
    timed_traces = [trace for trace in traces if trace["budget_mode"] == "fixed_time"]
    identities = [_trace_identity(trace) for trace in traces]
    matrix = _calibration_matrix_status(config, traces)
    matrix_passed = matrix["passed"] and incomplete_exact == 0
    gates["initializer_failure_recovery"] = {"count": recovery, "passed": recovery > 0}
    gates["exact_iteration_completeness"] = {"incomplete": incomplete_exact, "passed": incomplete_exact == 0}
    gates["trace_matrix_completeness"] = {
        "traces": len(traces), "fixed_iterations": len(fixed_traces),
        "fixed_time": len(timed_traces), "unique_identities": len(set(identities)),
        "passed": matrix_passed,
    }
    payload = {
        "version": "a4b-v2-train-gate-v1", "split": "train", "gates": gates,
        "passed": all(item["passed"] for item in gates.values()),
        "calibration_sha256": calibration["calibration_sha256"],
        "config_sha256": config["config_sha256"], "record": _record(config),
    }
    payload["gate_sha256"] = canonical_hash(payload)
    return payload


def run_cell(config, context, output, cell, *, limit, smoke):
    gate = json.loads((output / "train_gate.json").read_text())
    label_gate = json.loads((output / "label_gate.json").read_text())
    if not gate["passed"] or not label_gate["passed"]:
        raise RuntimeError("A4b v2 train/label gate failed; development is blocked")
    calibration = json.loads((output / "train_calibration.json").read_text())
    items, manifest_hash = load_a4b_v2_items(output / "corpus", "development", context)
    selected = [(r, i) for r, i in items if r["cell_id"] == cell]
    if limit is not None:
        selected = selected[:limit]
    traces, metrics = [], []
    seeds = config["search"]["random_seeds"][:1] if smoke else config["search"]["random_seeds"]
    for record, instance in selected:
        for seed in seeds:
            for method, mode, single in METHODS:
                actual_single = calibration["selected_operator"] if single == "selected" else single
                for budget_mode in ("fixed_time", "fixed_iterations"):
                    trace = _run(
                        instance, context, config, record, method, mode, actual_single, int(seed), budget_mode,
                        update_scheme=calibration["selected_update_scheme"],
                        iterations=4 if smoke and budget_mode == "fixed_iterations" else None,
                        time_s=0.5 if smoke and budget_mode == "fixed_time" else None,
                    )
                    traces.append(trace)
                    if smoke:
                        continue
                    if budget_mode == "fixed_time":
                        metrics.extend(_fixed_time_rows(config, record, method, int(seed), trace, calibration))
                    else:
                        metrics.extend(_fixed_iteration_rows(config, record, method, int(seed), trace))
    if smoke:
        return {"version": "a4b-v2-development-smoke-v1", "trace_count": len(traces), "traces": traces, "manifest_sha256": manifest_hash, "record": _record(config)}
    _write_jsonl(output / "development_traces" / f"{cell}.jsonl", traces)
    _write_jsonl(output / "development_metrics" / f"{cell}.jsonl", metrics)
    _write_json(output / "development_records" / f"{cell}.json", _record(config, cell=cell, trace_count=len(traces), metric_count=len(metrics), manifest_sha256=manifest_hash))
    return {"trace_count": len(traces), "metric_count": len(metrics)}


def _state_features(instance, context, state):
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    segments = {item.id: item for item in instance.segments}
    static = []
    for index, unit in enumerate(units):
        static.append({
            "unit_index": index, "segment_ids": list(unit),
            "duration_s": sum(segments[s].process_duration_s for s in unit),
            "window_start_s": min(segments[s].time_window.start_s for s in unit),
            "window_end_s": min(segments[s].time_window.end_s for s in unit),
            "predecessor_ids": sorted({p for s in unit for p in segments[s].predecessor_ids}),
            "shared_resource_ids": sorted({r for s in unit for r in segments[s].shared_resource_ids}),
            "robot_costs": {robot: costs[index][j] if math.isfinite(costs[index][j]) else None for j, robot in enumerate(robots)},
        })
    robot_dynamic = []
    for robot, order in state.robot_orders:
        load = sum(costs[u][robots.index(robot)] for u in order if math.isfinite(costs[u][robots.index(robot)]))
        last = None
        if order:
            segment = segments[units[order[-1]][-1]]
            last = list(segment.end_pose.position_m)
        robot_dynamic.append({"robot_id": robot, "load_s": load, "completion_proxy_s": load, "last_location_m": last, "unit_order": list(order)})
    return static, robot_dynamic


def generate_labels(config, context, output):
    calibration = json.loads((output / "train_calibration.json").read_text())
    items, manifest_hash = load_a4b_v2_items(output / "corpus", "train", context)
    weights = config["search"]["objective_weights"]
    per_cell = int(config["data"]["label_states_per_cell"])
    ratio = float(config["labels"]["destroy_ratio"])
    repair_cap = int(config["labels"]["same_repair_candidate_evaluation_budget"])
    rows = []
    for cell in config["data"]["cells"]:
        states = []
        for record, instance in [(r, i) for r, i in items if r["cell_id"] == cell]:
            initializer = build_hybrid_load_balanced_initializer(instance, context, weights)
            before = evaluate_state_timed(instance, context, initializer.state, weights)
            states.append((record, instance, initializer.state, before, "initializer"))
            trace = _run(instance, context, config, record, "label_state_search", "random_lns", None, int(config["search"]["random_seeds"][0]), "fixed_iterations", update_scheme=calibration["selected_update_scheme"], iterations=4)
            for step in trace["steps"]:
                candidate_state = state_from_dict(step["candidate_state"])
                candidate_eval = evaluate_state_timed(instance, context, candidate_state, weights)
                category = "failed_search" if not candidate_eval.evaluation.verified else "near_boundary_feasible"
                states.append((record, instance, candidate_state, candidate_eval, category))
            if len(states) >= per_cell * 3:
                break
        selected = []
        for category in ("initializer", "failed_search", "near_boundary_feasible"):
            for item in states:
                if item[4] == category and state_hash(item[2]) not in {state_hash(x[2]) for x in selected}:
                    selected.append(item)
                    break
        for item in states:
            if len(selected) >= per_cell:
                break
            if state_hash(item[2]) not in {state_hash(x[2]) for x in selected}:
                selected.append(item)
        for state_index, (record, instance, before_state, before, category) in enumerate(selected[:per_cell]):
            static, robots = _state_features(instance, context, before_state)
            candidates, seen_sets = [], set()
            for operator_index, operator in enumerate(DESTROY_OPERATORS):
                seed = int(np.random.SeedSequence([config["data"]["master_seed"], len(rows), operator_index]).generate_state(1)[0])
                destroyed = select_destroy_set_v2(operator, instance, context, before_state, before, ratio, np.random.default_rng(seed))
                key = tuple(sorted(destroyed))
                if key in seen_sets:
                    continue
                seen_sets.add(key)
                deadline = time.monotonic_ns() + int(float(config["labels"]["micro_deadline_s"]) * 1e9)
                repaired = repair_destroyed_state_v2(instance, context, before_state, destroyed, weights=weights, candidate_evaluation_budget=repair_cap, deadline_ns=deadline)
                after = repaired.evaluation
                objective_delta = None
                if before.evaluation.objective is not None and after.evaluation.objective is not None:
                    objective_delta = float(before.evaluation.objective) - float(after.evaluation.objective)
                candidates.append({
                    "operator": operator, "destroy_set": list(destroyed), "destroy_ratio": ratio,
                    "repair_seed": seed, "repair_budget": repair_cap,
                    "after_state": state_to_dict(repaired.state), "after_state_sha256": state_hash(repaired.state),
                    "feasible_before": before.evaluation.verified, "feasible_after": after.evaluation.verified,
                    "feasible_improvement": bool(after.evaluation.verified and not before.evaluation.verified),
                    "objective_improvement": objective_delta,
                    "violation_before": before.diagnostic.vector.to_dict(), "violation_after": after.diagnostic.vector.to_dict(),
                    "violation_reduction": before.diagnostic.vector.scalar() - after.diagnostic.vector.scalar(),
                    "time_to_feasible_s": repaired.first_feasible_elapsed_s,
                    "repair_runtime_s": repaired.runtime_s, "repair_selection_runtime_s": repaired.selection_runtime_s,
                    "scheduler_runtime_s": after.timing.scheduler_s, "verifier_runtime_s": after.timing.verifier_s,
                    "candidate_evaluations": repaired.candidate_evaluations, "budget_exhausted": repaired.budget_exhausted,
                    "deadline_exhausted": repaired.deadline_exhausted, "assignment_edits": repaired.assignment_edits,
                    "order_edits": repaired.order_edits, "total_modified_units": repaired.total_modified_units,
                    "objective_after": after.evaluation.objective, "failure_reason_after": after.evaluation.failure_reason,
                    "plan_sha256": None if after.evaluation.plan is None else canonical_hash(after.evaluation.plan.to_dict()),
                })
            for left in candidates:
                left["dominates"] = sorted(right["after_state_sha256"] for right in candidates if _dominates(left, right))
            payload = {
                **_identity(record, "search_generated_neighborhood_labels", int(config["data"]["master_seed"])),
                "state_index": state_index, "state_category": category,
                "label_name": config["labels"]["name"], "not_true_expert_action": True,
                "current_assignment_and_order": state_to_dict(before_state), "current_state_sha256": state_hash(before_state),
                "current_verified": before.evaluation.verified, "current_failure_reason": before.evaluation.failure_reason,
                "current_objective": before.evaluation.objective, "best_so_far_objective": before.evaluation.objective,
                "current_violation": before.diagnostic.vector.to_dict(), "static_atomic_unit_features": static,
                "robot_dynamic_features": robots, "candidates": candidates,
                "manifest_sha256": manifest_hash, "config_sha256": config["config_sha256"],
            }
            payload["record_sha256"] = canonical_hash(payload)
            rows.append(payload)
    _write_jsonl(output / "search_generated_labels.jsonl", rows)
    duplicates = sum(len(row["candidates"]) != len({tuple(sorted(c["destroy_set"])) for c in row["candidates"]}) for row in rows)
    improvements = sum(c["feasible_improvement"] or c["violation_reduction"] > 1e-12 or (c["objective_improvement"] is not None and c["objective_improvement"] > 1e-12) for row in rows for c in row["candidates"])
    gate = {
        "version": "a4b-v2-label-gate-v1", "record_count": len(rows), "duplicate_state_destroy_sets": duplicates,
        "improving_candidates": improvements, "passed": duplicates == 0 and improvements > 0,
        "manifest_sha256": manifest_hash, "config_sha256": config["config_sha256"], "record": _record(config),
    }
    gate["gate_sha256"] = canonical_hash(gate)
    _write_json(output / "label_gate.json", gate)
    _write_json(output / "label_generation_record.json", _record(config, label_records=len(rows), manifest_sha256=manifest_hash))
    if not gate["passed"]:
        raise SystemExit("A4b v2 label gate failed; development remains blocked")


def _dominates(left, right):
    if left["feasible_after"] != right["feasible_after"]:
        return bool(left["feasible_after"])
    left_obj = math.inf if left["objective_after"] is None else left["objective_after"]
    right_obj = math.inf if right["objective_after"] is None else right["objective_after"]
    left_violation = sum(float(v) for v in left["violation_after"].values())
    right_violation = sum(float(v) for v in right["violation_after"].values())
    left_vector = (left_obj, left_violation, left["repair_runtime_s"], left["total_modified_units"])
    right_vector = (right_obj, right_violation, right["repair_runtime_s"], right["total_modified_units"])
    return all(a <= b for a, b in zip(left_vector, right_vector)) and any(a < b for a, b in zip(left_vector, right_vector))


def replay_audit(config, context, output):
    items, _ = load_a4b_v2_items(output / "corpus", "development", context)
    instances = {record["instance_id"]: instance for record, instance in items}
    checked = 0
    for cell in config["data"]["cells"]:
        for trace in _read_jsonl(output / "development_traces" / f"{cell}.jsonl"):
            if canonical_hash({k: v for k, v in trace.items() if k != "trace_sha256"}) != trace["trace_sha256"]:
                raise RuntimeError("trace content hash mismatch")
            instance = instances[trace["instance_id"]]
            for step in trace["steps"]:
                before = state_from_dict(step["before_state"])
                candidate = state_from_dict(step["candidate_state"])
                if state_hash(before) != step["before_state_sha256"] or state_hash(candidate) != step["candidate_state_sha256"]:
                    raise RuntimeError("trace state hash mismatch")
                evaluation = evaluate_state_timed(instance, context, candidate, config["search"]["objective_weights"])
                if evaluation.evaluation.verified != step["after_verified"]:
                    raise RuntimeError("trace verifier replay mismatch")
                checked += 1
    payload = {"version": "a4b-v2-replay-audit-v1", "steps_checked": checked, "passed": True, "config_sha256": config["config_sha256"], "record": _record(config)}
    payload["audit_sha256"] = canonical_hash(payload)
    return payload


def aggregate(config, output):
    gate = json.loads((output / "train_gate.json").read_text())
    label_gate = json.loads((output / "label_gate.json").read_text())
    replay = json.loads((output / "replay_audit.json").read_text())
    if not gate["passed"] or not label_gate["passed"] or not replay["passed"]:
        raise RuntimeError("A4b v2 prerequisite gate failed")
    metrics, traces, records = [], [], []
    for cell in config["data"]["cells"]:
        metrics.extend(_read_jsonl(output / "development_metrics" / f"{cell}.jsonl"))
        traces.extend(_read_jsonl(output / "development_traces" / f"{cell}.jsonl"))
        records.append(json.loads((output / "development_records" / f"{cell}.json").read_text()))
    group_rows = _group_rows(metrics)
    primary = [r for r in group_rows if r["view"] == "fixed_end_to_end_time" and r["budget"] == 1.0]
    random_cov = _group_coverage(primary, "random_lns")
    alns_cov = _group_coverage(primary, "adaptive_alns")
    exact_incomplete = sum(t["budget_mode"] == "fixed_iterations" and not t["fixed_iteration_complete"] for t in traces)
    failures = [row for row in metrics if not row["verified"]]
    payload = {
        "version": "a4b-ordinary-lns-development-results-v2", "evidence_label": "SIM_GEOMETRIC_DEVELOPMENT_ONLY",
        "metric_rows": len(metrics), "trace_count": len(traces), "task_group_count": len({r["task_group_id"] for r in metrics}),
        "summaries": _summaries(metrics), "group_rows": group_rows, "group_summaries": _summaries(group_rows, group=True),
        "failure_taxonomy": dict(sorted(Counter(str(r["failure_reason"]) for r in failures).items())),
        "candidate_failure_taxonomy": dict(sorted(Counter(str(s["after_failure_reason"]) for t in traces for s in t["steps"] if s["after_failure_reason"]).items())),
        "termination_taxonomy": dict(sorted(Counter(t["termination_reason"] for t in traces).items())),
        "exact_iteration_incomplete_traces": exact_incomplete,
        "gate": {"definition": "mean independent-task-group ALNS one-second coverage is not below random LNS", "random_lns_coverage": random_cov, "adaptive_alns_coverage": alns_cov, "alns_not_systematically_weaker_than_random": alns_cov + 1e-12 >= random_cov},
        "train_gate_sha256": gate["gate_sha256"], "label_gate_sha256": label_gate["gate_sha256"], "replay_audit_sha256": replay["audit_sha256"],
        "config_sha256": config["config_sha256"], "source_hashes": {p: sha256_file(ROOT / p) for p in SOURCE_FILES},
        "shard_records": records, "boundaries": config["boundaries"], "record": _record(config),
    }
    payload["result_sha256"] = canonical_hash(payload)
    _write_json(output / "summary.json", payload)
    _write_json(output / "failure_library.json", failures)
    _write_json(output / "group_aggregation.json", group_rows)
    _write_json(ROOT / "reports/phase1_allocation/a4b_ordinary_lns_dev_v2_summary.json", payload)
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


def _summaries(rows, group=False):
    result = {}
    for view, budget, method in sorted({(r["view"], r["budget"], r["method"]) for r in rows}):
        chosen = [r for r in rows if (r["view"], r["budget"], r["method"]) == (view, budget, method)]
        coverage_values = [float(r["coverage"]) for r in chosen] if group else [float(r["verified"]) for r in chosen]
        objectives = [float(r["objective"]) for r in chosen if r.get("objective") is not None] if not group else [float(r["conditional_mean_objective"]) for r in chosen if r.get("conditional_mean_objective") is not None]
        result[f"{view}:{budget}:{method}"] = {"rows" if not group else "independent_task_groups": len(chosen), "coverage" if not group else "mean_group_coverage": float(np.mean(coverage_values)), "conditional_mean_objective": None if not objectives else float(np.mean(objectives))}
    return result


def _group_rows(rows):
    result = []
    keys = sorted({(r["task_group_id"], r["cell_id"], r["view"], r["budget"], r["method"]) for r in rows})
    for group, cell, view, budget, method in keys:
        chosen = [r for r in rows if (r["task_group_id"], r["cell_id"], r["view"], r["budget"], r["method"]) == (group, cell, view, budget, method)]
        objectives = [float(r["objective"]) for r in chosen if r["objective"] is not None]
        result.append({"task_group_id": group, "cell_id": cell, "view": view, "budget": budget, "method": method, "variant_seed_rows": len(chosen), "coverage": sum(r["verified"] for r in chosen) / len(chosen), "conditional_mean_objective": None if not objectives else float(np.mean(objectives)), "failure_rows": sum(not r["verified"] for r in chosen)})
    return result


def _group_coverage(rows, method):
    chosen = [r for r in rows if r["method"] == method]
    return 0.0 if not chosen else float(np.mean([r["coverage"] for r in chosen]))


def _record(config, **extra):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    ended = time.time()
    value = {
        "protocol_id": config["protocol_id"], "config_sha256": config["config_sha256"],
        "runtime_amendment_sha256": config["runtime_amendment_sha256"],
        "watchdog_amendment_sha256": config["watchdog_amendment_sha256"],
        "parallel_amendment_sha256": config["parallel_amendment_sha256"],
        "recovery_amendment_sha256": config["recovery_amendment_sha256"],
        "registered_runtime_hardware": config["runtime_hardware"],
        "registered_parallel_execution": config["parallel_execution"],
        "job_id": os.environ.get("SLURM_JOB_ID"), "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "node": os.environ.get("SLURMD_NODENAME"),
        "command": " ".join(sys.argv), "exit_code": 0, "timestamp_unix_s": ended,
        "worker_started_unix_s": PROCESS_STARTED_UNIX_S, "worker_ended_unix_s": ended,
        "worker_wall_s": ended - PROCESS_STARTED_UNIX_S,
        "cpu_affinity": affinity, "cpu_affinity_count": len(affinity),
        "system_loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"), "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK")},
        **extra,
    }
    value["record_sha256"] = canonical_hash(value)
    return value


def _validate_worker_environment(config):
    required = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    wrong = {key: os.environ.get(key) for key, expected in required.items() if os.environ.get(key) != expected}
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if wrong or len(affinity) != 1:
        raise RuntimeError(f"invalid single-CPU worker environment: env={wrong}, affinity={affinity}")
    parallel = config["parallel_execution"]
    if (
        os.environ.get("SLURM_JOB_PARTITION") != parallel["partition"]
        or os.environ.get("SLURM_JOB_ACCOUNT") != parallel["account"]
        or os.environ.get("SLURMD_NODENAME") != parallel["node"]
    ):
        raise RuntimeError("worker partition/account/node differs from frozen parallel amendment")


def _source_hashes():
    return {path: sha256_file(ROOT / path) for path in SOURCE_FILES}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
