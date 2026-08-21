#!/usr/bin/env python3
"""Generate and run the A4b ordinary-LNS development protocol.

This runner has no neural training command and rejects validation, frozen-test
and stress splits.  Controlled methods differ only in destroy selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import evaluate_state
from safe_residual_rl.allocation.search.alns import AlnsConfig, repair_destroyed_state, run_search
from safe_residual_rl.allocation.search.anytime import build_hybrid_load_balanced_initializer
from safe_residual_rl.allocation.search.data import (
    generate_a4b_data,
    load_a4b_config,
    load_a4b_items,
    sha256_file,
)
from safe_residual_rl.allocation.search.operators import (
    DESTROY_OPERATORS,
    HANDCRAFTED_OPERATORS,
    select_destroy_set,
)
from safe_residual_rl.allocation.search.trace import (
    best_at_budget,
    best_at_iteration,
    canonical_hash,
    state_hash,
    state_to_dict,
)
from safe_residual_rl.allocation.solvers import solve_order_aware_lns
from safe_residual_rl.allocation.solvers.common import allocation_units
from safe_residual_rl.allocation.verifier import verify_plan

CONFIG_PATH = ROOT / "configs/allocation/a4b_neural_lns_dev_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/phase1_allocation/a4b_neural_lns_dev_v1"
SOURCE_FILES = (
    "configs/allocation/a4b_neural_lns_dev_v1.json",
    "src/safe_residual_rl/allocation/solvers/common.py",
    "src/safe_residual_rl/allocation/solvers/heuristics.py",
    "src/safe_residual_rl/allocation/solvers/milp.py",
    "src/safe_residual_rl/allocation/search/__init__.py",
    "src/safe_residual_rl/allocation/search/anytime.py",
    "src/safe_residual_rl/allocation/search/operators.py",
    "src/safe_residual_rl/allocation/search/alns.py",
    "src/safe_residual_rl/allocation/search/trace.py",
    "src/safe_residual_rl/allocation/search/data.py",
    "scripts/run_a4b_baseline_development.py",
    "tests/allocation/test_a4b_anytime_evaluator.py",
    "tests/allocation/test_a4b_alns.py",
    "slurm/a4b_development_smoke.sbatch",
    "slurm/a4b_development_array.sbatch",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("generate", "select-operator", "smoke", "run-cell", "labels", "aggregate"),
    )
    parser.add_argument("--cell")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_a4b_config(CONFIG_PATH)
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.command == "generate":
        manifest = generate_a4b_data(ROOT, config, output / "corpus", context)
        _write_json(output / "generation_record.json", _record(config, manifest_sha256=manifest["manifest_sha256"]))
    elif args.command == "select-operator":
        selection = select_operator(config, context, output)
        _write_json(output / "train_operator_selection.json", selection)
    elif args.command == "smoke":
        smoke = run_cell(config, context, output, "iid_small", limit=1, smoke=True)
        _write_json(output / "development_smoke.json", smoke)
    elif args.command == "run-cell":
        if args.cell not in config["data"]["cells"]:
            raise SystemExit("--cell must be one of the six registered cells")
        run_cell(config, context, output, args.cell, limit=None, smoke=False)
    elif args.command == "labels":
        generate_labels(config, context, output)
    else:
        aggregate(config, output)


def _search_config(config, seed, *, iterations=None, time_s=None) -> AlnsConfig:
    search, alns = config["search"], config["alns"]
    return AlnsConfig(
        protocol_id=config["protocol_id"],
        iterations=int(search["maximum_iterations"] if iterations is None else iterations),
        maximum_end_to_end_time_s=float(max(search["fixed_end_to_end_time_s"]) if time_s is None else time_s),
        destroy_ratios=tuple(float(item) for item in search["destroy_ratios"]),
        repair_candidate_evaluation_budget=int(search["repair_candidate_evaluation_budget"]),
        random_seed=int(seed),
        objective_weights=search["objective_weights"],
        initial_temperature_fraction=float(alns["initial_temperature_fraction"]),
        cooling_rate=float(alns["cooling_rate"]),
        reaction_factor=float(alns["reaction_factor"]),
        reward_global_best=float(alns["rewards"]["global_best"]),
        reward_improving_accepted=float(alns["rewards"]["accepted_improvement"]),
        reward_new_feasible=float(alns["rewards"]["new_feasible"]),
        reward_accepted=float(alns["rewards"]["accepted_non_improvement"]),
        reward_rejected=float(alns["rewards"]["rejected"]),
        restart_no_improvement=int(search["restart_no_improvement"]),
        allow_infeasible_current=bool(search["allow_infeasible_current"]),
    )


def select_operator(config, context, output):
    items, manifest_hash = load_a4b_items(output / "corpus", "train", context)
    group_limit = int(config["data"]["operator_selection_groups_per_cell"])
    chosen = []
    for cell in config["data"]["cells"]:
        groups = sorted({record["task_group_id"] for record, _ in items if record["cell_id"] == cell})[:group_limit]
        chosen.extend((record, instance) for record, instance in items if record["task_group_id"] in groups)
    rows = []
    targets = defaultdict(list)
    seed = int(config["search"]["random_seeds"][0])
    for record, instance in chosen:
        for operator in HANDCRAFTED_OPERATORS:
            initializer = build_hybrid_load_balanced_initializer(instance, context, config["search"]["objective_weights"])
            outcome = run_search(
                instance,
                context,
                initializer,
                _search_config(config, seed, iterations=10, time_s=1.0),
                mode="single_operator",
                single_operator=operator,
                task_group_id=record["task_group_id"],
                difficulty=record["cell_id"],
                split="train",
            )
            snapshot = best_at_iteration(outcome.trace, 10)
            rows.append(
                {
                    "cell_id": record["cell_id"],
                    "task_group_id": record["task_group_id"],
                    "instance_id": record["instance_id"],
                    "operator": operator,
                    "verified": snapshot.verified,
                    "objective": snapshot.objective,
                    "runtime_s": (outcome.trace.return_monotonic_ns - outcome.trace.start_monotonic_ns) / 1e9,
                    "trace_sha256": outcome.trace.sha256,
                }
            )
            if snapshot.verified and snapshot.objective is not None:
                targets[record["cell_id"]].append(snapshot.objective)
    summaries = {}
    for operator in HANDCRAFTED_OPERATORS:
        selected = [row for row in rows if row["operator"] == operator]
        verified = [row for row in selected if row["verified"]]
        summaries[operator] = {
            "rows": len(selected),
            "coverage": sum(row["verified"] for row in selected) / len(selected),
            "mean_objective": None if not verified else float(np.mean([row["objective"] for row in verified])),
            "median_runtime_s": float(np.median([row["runtime_s"] for row in selected])),
        }
    selected_operator = min(
        summaries,
        key=lambda name: (
            -summaries[name]["coverage"],
            math.inf if summaries[name]["mean_objective"] is None else summaries[name]["mean_objective"],
            summaries[name]["median_runtime_s"],
            name,
        ),
    )
    payload = {
        "version": "a4b-train-only-operator-selection-v1",
        "selection_split": "train",
        "future_validation_accessed": False,
        "selected_operator": selected_operator,
        "summaries": summaries,
        "targets": {cell: float(statistics.median(values)) for cell, values in targets.items()},
        "rows": rows,
        "manifest_sha256": manifest_hash,
        "config_sha256": config["config_sha256"],
    }
    payload["selection_sha256"] = canonical_hash(payload)
    return payload


def run_cell(config, context, output, cell, *, limit, smoke):
    selection = json.loads((output / "train_operator_selection.json").read_text())
    items, manifest_hash = load_a4b_items(output / "corpus", "development", context)
    selected = [(record, instance) for record, instance in items if record["cell_id"] == cell]
    if limit is not None:
        selected = selected[:limit]
    methods = (
        ("random_lns", "random_lns", None),
        ("handcrafted_round_robin", "handcrafted_round_robin", None),
        ("best_single_train_selected", "single_operator", selection["selected_operator"]),
        ("adaptive_alns", "adaptive_alns", None),
    )
    trace_rows = []
    metric_rows = []
    for record, instance in selected:
        for seed in config["search"]["random_seeds"]:
            for method_id, mode, single in methods:
                initializer = build_hybrid_load_balanced_initializer(instance, context, config["search"]["objective_weights"])
                outcome = run_search(
                    instance,
                    context,
                    initializer,
                    _search_config(config, int(seed), iterations=4 if smoke else None, time_s=0.5 if smoke else None),
                    mode=mode,
                    single_operator=single,
                    task_group_id=record["task_group_id"],
                    difficulty=cell,
                    split="development",
                )
                trace_payload = outcome.trace.to_dict()
                trace_payload["method_id"] = method_id
                trace_payload["trace_sha256"] = outcome.trace.sha256
                trace_rows.append(trace_payload)
                metric_rows.extend(_metric_rows(config, record, method_id, int(seed), outcome.trace))

            if len(allocation_units(instance)) <= int(config["oracle_destroy"]["maximum_atomic_units"]):
                initializer = build_hybrid_load_balanced_initializer(instance, context, config["search"]["objective_weights"])
                oracle = run_search(
                    instance,
                    context,
                    initializer,
                    _search_config(config, int(seed), iterations=4 if smoke else None, time_s=0.5 if smoke else None),
                    mode="oracle_destroy",
                    task_group_id=record["task_group_id"],
                    difficulty=cell,
                    split="development",
                )
                payload = oracle.trace.to_dict()
                payload["method_id"] = "oracle_destroy_upper_bound"
                payload["trace_sha256"] = oracle.trace.sha256
                trace_rows.append(payload)
                metric_rows.extend(_metric_rows(config, record, "oracle_destroy_upper_bound", int(seed), oracle.trace))

            # The historical order-aware LNS is retained as a fixed-iteration
            # reference.  It is not inserted into the controlled fixed-time
            # family because it predates the shared repair/trace interface.
            started = time.monotonic_ns()
            reference = solve_order_aware_lns(
                instance,
                context,
                iterations=4 if smoke else int(max(config["search"]["fixed_iterations"])),
                seed=int(seed),
                objective_weights=config["search"]["objective_weights"],
            )
            completed = time.monotonic_ns()
            checked = None if reference.plan is None else verify_plan(instance, reference.plan, context)
            metric_rows.append(
                {
                    **_identity(record, "current_order_aware_lns_reference", int(seed)),
                    "view": "fixed_iterations_reference",
                    "budget": 4 if smoke else int(max(config["search"]["fixed_iterations"])),
                    "verified": bool(checked and checked.feasible),
                    "objective": reference.objective_value if checked and checked.feasible else None,
                    "incumbent_elapsed_s": (completed - started) / 1e9 if checked and checked.feasible else None,
                    "failure_reason": None if checked and checked.feasible else reference.status,
                    "anytime_compliant": False,
                    "repair_config_sha256": canonical_hash(config["search"]),
                    "initializer_actual": "internal_order_aware_load_balanced_assignment",
                    "fallback_used": False,
                }
            )
    if smoke:
        return {"traces": trace_rows, "metrics": metric_rows, "manifest_sha256": manifest_hash}
    _write_jsonl(output / "development_traces" / f"{cell}.jsonl", trace_rows)
    _write_jsonl(output / "development_metrics" / f"{cell}.jsonl", metric_rows)
    _write_json(
        output / "development_records" / f"{cell}.json",
        _record(config, cell=cell, trace_count=len(trace_rows), metric_count=len(metric_rows), manifest_sha256=manifest_hash),
    )
    return {"trace_count": len(trace_rows), "metric_count": len(metric_rows)}


def _metric_rows(config, record, method, seed, trace):
    rows = []
    provenance = trace.initializer
    for budget in config["search"]["fixed_end_to_end_time_s"]:
        snapshot = best_at_budget(trace, float(budget))
        rows.append(
            {
                **_identity(record, method, seed),
                "view": "fixed_end_to_end_time",
                "budget": float(budget),
                "verified": snapshot.verified,
                "objective": snapshot.objective,
                "incumbent_elapsed_s": snapshot.incumbent_elapsed_s,
                "failure_reason": snapshot.failure_reason,
                "anytime_compliant": True,
                "repair_config_sha256": canonical_hash(config["search"]),
                "initializer_actual": provenance["actual_initializer"],
                "fallback_used": provenance["fallback_used"],
                "initializer_runtime_s": provenance["completion_elapsed_s"],
                "trace_sha256": trace.sha256,
            }
        )
    for iterations in config["search"]["fixed_iterations"]:
        snapshot = best_at_iteration(trace, int(iterations))
        rows.append(
            {
                **_identity(record, method, seed),
                "view": "fixed_iterations",
                "budget": int(iterations),
                "verified": snapshot.verified,
                "objective": snapshot.objective,
                "incumbent_elapsed_s": snapshot.incumbent_elapsed_s,
                "failure_reason": snapshot.failure_reason,
                "anytime_compliant": True,
                "repair_config_sha256": canonical_hash(config["search"]),
                "initializer_actual": provenance["actual_initializer"],
                "fallback_used": provenance["fallback_used"],
                "initializer_runtime_s": provenance["completion_elapsed_s"],
                "trace_sha256": trace.sha256,
            }
        )
    return rows


def _identity(record, method, seed):
    return {
        "split": record["split"],
        "cell_id": record["cell_id"],
        "task_group_id": record["task_group_id"],
        "variant_index": record["variant_index"],
        "instance_id": record["instance_id"],
        "method": method,
        "search_seed": seed,
    }


def generate_labels(config, context, output):
    items, manifest_hash = load_a4b_items(output / "corpus", "train", context)
    per_cell = int(config["data"]["label_states_per_cell"])
    selected = []
    for cell in config["data"]["cells"]:
        selected.extend([(r, i) for r, i in items if r["cell_id"] == cell][:per_cell])
    rows = []
    weights = config["search"]["objective_weights"]
    repair_budget = int(config["labels"]["same_repair_candidate_evaluation_budget"])
    ratio = 0.25
    for record, instance in selected:
        initializer = build_hybrid_load_balanced_initializer(instance, context, weights)
        before = evaluate_state(instance, context, initializer.state, weights)
        candidates = []
        for index, operator in enumerate(DESTROY_OPERATORS):
            seed = int(np.random.SeedSequence([config["data"]["master_seed"], index, len(rows)]).generate_state(1)[0])
            rng = np.random.default_rng(seed)
            destroyed = select_destroy_set(operator, instance, context, initializer.state, before, ratio, rng)
            repaired = repair_destroyed_state(
                instance,
                context,
                initializer.state,
                destroyed,
                random_seed=seed,
                weights=weights,
                candidate_evaluation_budget=repair_budget,
            )
            objective_improvement = None
            if before.objective is not None and repaired.evaluation.objective is not None:
                objective_improvement = float(before.objective) - float(repaired.evaluation.objective)
            candidates.append(
                {
                    "operator": operator,
                    "destroy_set": list(destroyed),
                    "destroy_ratio": ratio,
                    "repair_seed": seed,
                    "repair_budget": repair_budget,
                    "after_state": state_to_dict(repaired.state),
                    "after_state_sha256": state_hash(repaired.state),
                    "feasible_improvement": bool(repaired.evaluation.verified and not before.verified),
                    "objective_improvement": objective_improvement,
                    "time_to_feasible_s": repaired.runtime_s if repaired.evaluation.verified else None,
                    "repair_cost_s": repaired.runtime_s,
                    "edit_distance": repaired.edit_distance,
                    "candidate_evaluations": repaired.candidate_evaluations,
                    "feasible_after": repaired.evaluation.verified,
                    "failure_reason_after": repaired.evaluation.failure_reason,
                    "objective_after": repaired.evaluation.objective,
                    "plan_sha256": None if repaired.evaluation.plan is None else canonical_hash(repaired.evaluation.plan.to_dict()),
                    "verifier_sha256": _state_evaluation_hash(repaired.evaluation),
                }
            )
        for left in candidates:
            left["dominates"] = sorted(
                right["operator"]
                for right in candidates
                if _dominates(left, right)
            )
        payload = {
            **_identity(record, "candidate_destroy_label_set", int(config["data"]["master_seed"])),
            "label_name": "search-generated neighborhood improvement labels",
            "not_true_expert_action": True,
            "before_state": state_to_dict(initializer.state),
            "before_state_sha256": state_hash(initializer.state),
            "before_verified": before.verified,
            "before_objective": before.objective,
            "before_plan_sha256": None if before.plan is None else canonical_hash(before.plan.to_dict()),
            "before_verifier_sha256": _state_evaluation_hash(before),
            "candidates": candidates,
            "manifest_sha256": manifest_hash,
            "config_sha256": config["config_sha256"],
        }
        payload["record_sha256"] = canonical_hash(payload)
        rows.append(payload)
    _write_jsonl(output / "search_generated_labels.jsonl", rows)
    _write_json(output / "label_generation_record.json", _record(config, label_records=len(rows), manifest_sha256=manifest_hash))


def _state_evaluation_hash(evaluation):
    return canonical_hash(
        {
            "verified": evaluation.verified,
            "objective": evaluation.objective,
            "surrogate": evaluation.surrogate,
            "failure_reason": evaluation.failure_reason,
            "plan": None if evaluation.plan is None else evaluation.plan.to_dict(),
        }
    )


def _dominates(left, right):
    left_feasible, right_feasible = left["feasible_after"], right["feasible_after"]
    if left_feasible and not right_feasible:
        return True
    if left_feasible != right_feasible:
        return False
    left_objective = math.inf if left["objective_after"] is None else left["objective_after"]
    right_objective = math.inf if right["objective_after"] is None else right["objective_after"]
    no_worse = left_objective <= right_objective and left["repair_cost_s"] <= right["repair_cost_s"] and left["edit_distance"] <= right["edit_distance"]
    strict = left_objective < right_objective or left["repair_cost_s"] < right["repair_cost_s"] or left["edit_distance"] < right["edit_distance"]
    return bool(no_worse and strict)


def aggregate(config, output):
    metrics = []
    traces = []
    records = []
    for cell in config["data"]["cells"]:
        metric_path = output / "development_metrics" / f"{cell}.jsonl"
        trace_path = output / "development_traces" / f"{cell}.jsonl"
        record_path = output / "development_records" / f"{cell}.json"
        if not metric_path.is_file() or not trace_path.is_file() or not record_path.is_file():
            raise RuntimeError(f"missing A4b development shard: {cell}")
        metrics.extend(_read_jsonl(metric_path))
        traces.extend(_read_jsonl(trace_path))
        records.append(json.loads(record_path.read_text()))
    controlled = [row for row in metrics if row["anytime_compliant"]]
    summaries = _summaries(controlled)
    group_rows = _group_rows(controlled)
    group_summaries = _group_summaries(group_rows)
    failures = [row for row in controlled if not row["verified"]]
    primary_key = ("fixed_end_to_end_time", 1.0)
    primary = [row for row in group_rows if (row["view"], row["budget"]) == primary_key]
    random_coverage = _group_coverage(primary, "random_lns")
    alns_coverage = _group_coverage(primary, "adaptive_alns")
    candidate_failures = [
        step["after_failure_reason"]
        for trace in traces
        if not trace["method"].startswith("oracle")
        for step in trace["steps"]
        if step["after_failure_reason"] is not None
    ]
    gate = alns_coverage + 1e-12 >= random_coverage
    payload = {
        "version": "a4b-ordinary-lns-development-results-v1",
        "evidence_label": "SIM_GEOMETRIC_DEVELOPMENT_ONLY",
        "row_count": len(metrics),
        "controlled_row_count": len(controlled),
        "trace_count": len(traces),
        "failure_count": len(failures),
        "task_group_count": len({row["task_group_id"] for row in controlled}),
        "summaries": summaries,
        "group_rows": group_rows,
        "group_summaries": group_summaries,
        "failure_taxonomy": dict(sorted(_count(row["failure_reason"] for row in failures).items())),
        "search_candidate_failure_taxonomy": dict(sorted(_count(candidate_failures).items())),
        "termination_taxonomy": dict(
            sorted(_count(trace["termination_reason"] for trace in traces).items())
        ),
        "gate": {
            "definition": "mean independent-task-group adaptive ALNS 1.0-second development coverage is not below fixed random-destroy LNS",
            "independent_task_groups": len({row["task_group_id"] for row in primary}),
            "random_lns_coverage": random_coverage,
            "adaptive_alns_coverage": alns_coverage,
            "alns_not_systematically_weaker_than_random": gate,
        },
        "operator_selection_sha256": json.loads((output / "train_operator_selection.json").read_text())["selection_sha256"],
        "config_sha256": config["config_sha256"],
        "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "shard_records": records,
        "slurm": _record(config),
        "boundaries": config["boundaries"],
    }
    payload["result_sha256"] = canonical_hash(payload)
    _write_json(output / "summary.json", payload)
    _write_json(output / "failure_library.json", failures)
    _write_json(output / "group_aggregation.json", group_rows)
    _write_json(ROOT / "reports/phase1_allocation/a4b_neural_lns_dev_v1_summary.json", payload)
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


def _summaries(rows):
    result = {}
    for key in sorted({(row["view"], row["budget"], row["method"]) for row in rows}):
        view, budget, method = key
        selected = [row for row in rows if (row["view"], row["budget"], row["method"]) == key]
        verified = [row for row in selected if row["verified"]]
        result[f"{view}:{budget}:{method}"] = {
            "rows": len(selected),
            "coverage": sum(row["verified"] for row in selected) / len(selected),
            "conditional_mean_objective": None if not verified else float(np.mean([row["objective"] for row in verified])),
            "median_time_to_incumbent_s": None if not verified else float(np.median([row["incumbent_elapsed_s"] for row in verified])),
        }
    return result


def _group_rows(rows):
    result = []
    keys = sorted({(row["task_group_id"], row["cell_id"], row["view"], row["budget"], row["method"]) for row in rows})
    for group, cell, view, budget, method in keys:
        selected = [row for row in rows if (row["task_group_id"], row["cell_id"], row["view"], row["budget"], row["method"]) == (group, cell, view, budget, method)]
        result.append(
            {
                "task_group_id": group,
                "cell_id": cell,
                "view": view,
                "budget": budget,
                "method": method,
                "variant_seed_rows": len(selected),
                "coverage": sum(row["verified"] for row in selected) / len(selected),
                "mean_objective_with_failures_null": None if not any(row["verified"] for row in selected) else float(np.mean([row["objective"] for row in selected if row["verified"]])),
            }
        )
    return result


def _group_summaries(rows):
    result = {}
    for key in sorted({(row["view"], row["budget"], row["method"]) for row in rows}):
        view, budget, method = key
        selected = [
            row for row in rows
            if (row["view"], row["budget"], row["method"]) == key
        ]
        objectives = [
            row["mean_objective_with_failures_null"]
            for row in selected
            if row["mean_objective_with_failures_null"] is not None
        ]
        result[f"{view}:{budget}:{method}"] = {
            "independent_task_groups": len(selected),
            "mean_group_coverage": float(np.mean([row["coverage"] for row in selected])),
            "conditional_mean_group_objective": (
                None if not objectives else float(np.mean(objectives))
            ),
        }
    return result


def _group_coverage(rows, method):
    selected = [row for row in rows if row["method"] == method]
    return 0.0 if not selected else float(np.mean([row["coverage"] for row in selected]))


def _count(values):
    result = defaultdict(int)
    for value in values:
        result[str(value)] += 1
    return result


def _record(config, **extra):
    value = {
        "protocol_id": config["protocol_id"],
        "config_sha256": config["config_sha256"],
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "command": " ".join(sys.argv),
        "exit_code": 0,
        "timestamp_unix_s": time.time(),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        **extra,
    }
    value["record_sha256"] = canonical_hash(value)
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
