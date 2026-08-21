#!/usr/bin/env python3
"""Protocol-R implementation entry point.

The checked-in candidate config is deliberately a draft, so every command that
could generate Protocol-R data or search evidence fails before creating its
output.  ``audit-draft`` and ``fixture-parity`` are the only review-time modes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.generation import stable_seed
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.schema import allocation_instance_from_dict
from safe_residual_rl.allocation.search.alns_protocol_r import (
    run_search_protocol_r,
    transition_signature,
)
from safe_residual_rl.allocation.search.alns_v2 import AlnsV2Config, run_search_v2
from safe_residual_rl.allocation.search.anytime import build_hybrid_load_balanced_initializer
from safe_residual_rl.allocation.search.data_protocol_r import (
    generate_protocol_r_data,
    load_protocol_r_config,
    load_protocol_r_items,
    require_execution_ready,
    sha256_file,
)
from safe_residual_rl.allocation.search.diagnostics import evaluate_state_timed
from safe_residual_rl.allocation.search.metrics import normalized_primal_integral
from safe_residual_rl.allocation.search.operators_v2 import select_destroy_set_v2
from safe_residual_rl.allocation.search.prepared_repair import prepare_repair_problem
from safe_residual_rl.allocation.search.protocol_r_pipeline import (
    conjunctive_gate,
    deterministic_merge_shards,
    trace_identity,
    validate_trace_matrix,
)
from safe_residual_rl.allocation.search.repair_protocol_r import (
    parity_report,
    repair_destroyed_state_protocol_r,
    repair_destroyed_state_reference_audited,
)
from safe_residual_rl.allocation.search.trace import canonical_hash

CONFIG_PATH = ROOT / "configs/allocation/a4b_protocol_r_freeze_candidate_v1.json"
PROTOCOL_PATH = ROOT / "docs/39_a4b_protocol_r_freeze_package_draft.md"
DEFAULT_OUTPUT = ROOT / "outputs/phase1_allocation/a4b_ordinary_search_recovery_v3"
EXECUTION_TOKEN = "EXECUTE_A4B_PROTOCOL_R_V3"
CELLS = (
    "iid_small",
    "iid_medium",
    "dense_precedence",
    "resource_bottleneck",
    "tight_windows",
    "scale",
)
DESTROY_OPERATORS = (
    "random_destroy",
    "worst_cost_destroy",
    "load_imbalance_destroy",
    "precedence_chain_destroy",
    "critical_slack_destroy",
    "shared_resource_conflict_destroy",
    "relatedness_destroy",
    "compound_destroy",
)
CALIBRATION_METHODS = tuple(f"single:{item}" for item in DESTROY_OPERATORS) + (
    "alns_online",
    "alns_segmented",
    "random_lns",
)
DEVELOPMENT_METHODS = (
    "random_lns",
    "handcrafted_round_robin",
    "best_single_train_selected",
    "adaptive_alns",
)
SOURCE_FILES = (
    "configs/allocation/a4b_protocol_r_freeze_candidate_v1.json",
    "configs/allocation/a3_5_pointer_pilot_v1.json",
    "configs/allocation/oracle_proxy_v1.json",
    "docs/39_a4b_protocol_r_freeze_package_draft.md",
    "docs/40_a4b_protocol_r_implementation_review.md",
    "src/safe_residual_rl/allocation/generation.py",
    "src/safe_residual_rl/allocation/oracle.py",
    "src/safe_residual_rl/allocation/pointer_pilot.py",
    "src/safe_residual_rl/allocation/schema.py",
    "src/safe_residual_rl/allocation/scheduling.py",
    "src/safe_residual_rl/allocation/verifier.py",
    "src/safe_residual_rl/allocation/witness.py",
    "src/safe_residual_rl/allocation/repair/identical.py",
    "src/safe_residual_rl/allocation/solvers/__init__.py",
    "src/safe_residual_rl/allocation/solvers/common.py",
    "src/safe_residual_rl/allocation/solvers/heuristics.py",
    "src/safe_residual_rl/allocation/search/anytime.py",
    "src/safe_residual_rl/allocation/search/diagnostics.py",
    "src/safe_residual_rl/allocation/search/operators.py",
    "src/safe_residual_rl/allocation/search/operators_v2.py",
    "src/safe_residual_rl/allocation/search/metrics.py",
    "src/safe_residual_rl/allocation/search/trace.py",
    "src/safe_residual_rl/allocation/search/prepared_repair.py",
    "src/safe_residual_rl/allocation/search/repair_protocol_r.py",
    "src/safe_residual_rl/allocation/search/alns_protocol_r.py",
    "src/safe_residual_rl/allocation/search/data_protocol_r.py",
    "src/safe_residual_rl/allocation/search/protocol_r_pipeline.py",
    "scripts/run_a4b_protocol_r.py",
    "scripts/run_a4b_protocol_r_worker.sh",
    "scripts/submit_a4b_protocol_r_chain.sh",
    "tests/allocation/test_a4b_protocol_r_repair.py",
    "tests/allocation/test_a4b_protocol_r_data.py",
    "tests/allocation/test_a4b_protocol_r_pipeline.py",
    "slurm/a4b_protocol_r_preflight.sbatch",
    "slurm/a4b_protocol_r_generate.sbatch",
    "slurm/a4b_protocol_r_profile_packed.sbatch",
    "slurm/a4b_protocol_r_profile_gate.sbatch",
    "slurm/a4b_protocol_r_calibration_packed.sbatch",
    "slurm/a4b_protocol_r_train_gate.sbatch",
    "slurm/a4b_protocol_r_smoke.sbatch",
    "slurm/a4b_protocol_r_development_packed.sbatch",
    "slurm/a4b_protocol_r_finalize.sbatch",
)


def _source_hashes():
    return {path: sha256_file(ROOT / path) for path in SOURCE_FILES}


def _source_sha256():
    return canonical_hash(_source_hashes())


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _record(config, **extra):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    payload = {
        "version": "a4b-protocol-r-execution-record-v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config["config_sha256"],
        "protocol_document_sha256": sha256_file(PROTOCOL_PATH),
        "source_sha256": _source_sha256(),
        "source_hashes": _source_hashes(),
        "hostname": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity),
        "loadavg": list(os.getloadavg()),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "timestamp_unix_s": time.time(),
        **extra,
    }
    payload["record_sha256"] = canonical_hash(payload)
    return payload


def _require_runtime(config):
    require_execution_ready(config)
    if os.environ.get("A4B_PROTOCOL_R_EXECUTION_TOKEN") != EXECUTION_TOKEN:
        raise PermissionError("explicit Protocol-R execution token missing")


def _search_config(config, seed, budget_mode, *, update_scheme="segmented", iterations=None, time_s=None):
    search = config["search"]
    alns = config["alns"]
    return AlnsV2Config(
        protocol_id=config["protocol_id"],
        budget_mode=budget_mode,
        iterations=int(iterations or max(search["fixed_iterations"])),
        end_to_end_time_s=(
            float(time_s or max(search["fixed_end_to_end_time_s"]))
            if budget_mode == "fixed_time"
            else None
        ),
        safety_watchdog_s=float(search["safety_watchdog_s"]),
        destroy_ratios=tuple(float(item) for item in search["destroy_ratios"]),
        repair_candidate_evaluation_budget=int(search["repair_candidate_evaluation_budget"]),
        random_seed=int(seed),
        objective_weights={str(k): float(v) for k, v in search["objective_weights"].items()},
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


def _method_spec(method, selected_operator=None, selected_scheme="segmented"):
    if method == "random_lns":
        return "random_lns", None, "segmented"
    if method == "handcrafted_round_robin":
        return "handcrafted_round_robin", None, "segmented"
    if method.startswith("single:"):
        return "single_operator", method.split(":", 1)[1], "segmented"
    if method == "best_single_train_selected":
        return "single_operator", selected_operator, "segmented"
    if method == "alns_online":
        return "adaptive_alns", None, "online"
    if method in {"alns_segmented", "adaptive_alns"}:
        return "adaptive_alns", None, selected_scheme
    raise ValueError(f"unknown Protocol-R method {method}")


def _run_trace(config, context, record, instance, method, seed, budget_mode, *, selected_operator=None, selected_scheme="segmented", iterations=None, time_s=None):
    mode, single, scheme = _method_spec(method, selected_operator, selected_scheme)
    search_config = _search_config(
        config,
        seed,
        budget_mode,
        update_scheme=scheme,
        iterations=iterations,
        time_s=time_s,
    )
    initializer = build_hybrid_load_balanced_initializer(
        instance, context, config["search"]["objective_weights"]
    )
    outcome = run_search_protocol_r(
        instance,
        context,
        initializer,
        search_config,
        mode=mode,
        task_group_id=record["task_group_id"],
        difficulty=record["cell_id"],
        split=record["split"],
        single_operator=single,
    )
    trace = dict(outcome.trace)
    trace.update(
        method_id=method,
        update_scheme=scheme,
        protocol_config_sha256=config["config_sha256"],
    )
    trace["trace_sha256"] = canonical_hash(
        {key: value for key, value in trace.items() if key != "trace_sha256"}
    )
    return trace


def _best_event(trace, *, iteration=None, cutoff=None):
    events = trace["incumbents"]
    if iteration is not None:
        eligible = [item for item in events if int(item["iteration"]) <= iteration]
    else:
        eligible = [item for item in events if float(item["elapsed_s"]) <= cutoff + 1e-12]
    return None if not eligible else min(eligible, key=lambda item: float(item["objective"]))


def _metric_rows(record, trace):
    rows = []
    if trace["budget_mode"] == "fixed_iterations":
        snapshots = (10, 20, 30)
        for budget in snapshots:
            event = _best_event(trace, iteration=budget)
            rows.append(
                {
                    "instance_id": record["instance_id"],
                    "task_group_id": record["task_group_id"],
                    "cell_id": record["cell_id"],
                    "method": trace["method_id"],
                    "search_seed": trace["random_seed"],
                    "view": "fixed_iterations",
                    "budget": budget,
                    "verified": event is not None,
                    "objective": None if event is None else event["objective"],
                    "completed_neighborhoods": sum(
                        bool(step["completed_before_cutoff"])
                        and int(step["iteration"]) <= budget
                        for step in trace["steps"]
                    ),
                    "initializer_failed": not trace["initializer"]["verifier_feasible"],
                }
            )
    else:
        for budget in (0.5, 1.0, 3.0):
            event = _best_event(trace, cutoff=budget)
            rows.append(
                {
                    "instance_id": record["instance_id"],
                    "task_group_id": record["task_group_id"],
                    "cell_id": record["cell_id"],
                    "method": trace["method_id"],
                    "search_seed": trace["random_seed"],
                    "view": "fixed_end_to_end_time",
                    "budget": budget,
                    "verified": event is not None,
                    "objective": None if event is None else event["objective"],
                    "completed_neighborhoods": sum(
                        bool(step["completed_before_cutoff"])
                        and float(step["elapsed_s"]) <= budget + 1e-12
                        for step in trace["steps"]
                    ),
                    "initializer_failed": not trace["initializer"]["verifier_feasible"],
                }
            )
    return rows


def audit_draft(config, context):
    fixture_names = (
        "01_valid_minimal.json",
        "03_valid_explicit_boundary.json",
        "04_valid_shared_zone.json",
    )
    checks = []
    for name in fixture_names:
        instance = allocation_instance_from_dict(
            load_auditable_fixture(ROOT / "data/fixtures/allocation" / name)["instance"]
        )
        initial = build_hybrid_load_balanced_initializer(
            instance, context, config["search"]["objective_weights"]
        )
        prepared = prepare_repair_problem(instance, context)
        reference = repair_destroyed_state_reference_audited(
            instance,
            context,
            initial.state,
            (0,),
            weights=config["search"]["objective_weights"],
            candidate_evaluation_budget=256,
            prepared=prepared,
        )
        candidate = repair_destroyed_state_protocol_r(
            instance,
            context,
            initial.state,
            (0,),
            weights=config["search"]["objective_weights"],
            candidate_evaluation_budget=256,
            prepared=prepared,
        )
        checks.append({"fixture": name, **parity_report(reference, candidate)})
    payload = {
        "version": "a4b-protocol-r-draft-audit-v1",
        "status": config["status"],
        "execution_ready": False,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "record": _record(config),
    }
    payload["audit_sha256"] = canonical_hash(payload)
    return payload


def _calibration_items(config, context, output):
    items, manifest_hash = load_protocol_r_items(output / "corpus", "train", context)
    selected = []
    for record, instance in items:
        group = record["task_group_id"]
        if record["challenge"] or "regular-group-000" in group or "regular-group-001" in group:
            selected.append((record, instance))
    if len(selected) != int(config["expected_matrices"]["train_calibration"]["instances"]):
        raise RuntimeError("Protocol-R calibration subset is not exactly 40 instances")
    return tuple(selected), manifest_hash


def profile_cell(config, context, output, cell):
    items, manifest_hash = _calibration_items(config, context, output)
    items = [(record, instance) for record, instance in items if record["cell_id"] == cell]
    rows = []
    for record, instance in items:
        initial = build_hybrid_load_balanced_initializer(
            instance, context, config["search"]["objective_weights"]
        )
        current = evaluate_state_timed(
            instance, context, initial.state, config["search"]["objective_weights"]
        )
        prepared = prepare_repair_problem(instance, context)
        for operator in DESTROY_OPERATORS:
            for ratio in config["search"]["destroy_ratios"]:
                seed = stable_seed(config["search"]["profile_seed"], record["instance_id"], operator, ratio)
                destroyed = select_destroy_set_v2(
                    operator,
                    instance,
                    context,
                    initial.state,
                    current,
                    float(ratio),
                    np.random.default_rng(seed),
                )
                reference = repair_destroyed_state_reference_audited(
                    instance,
                    context,
                    initial.state,
                    destroyed,
                    weights=config["search"]["objective_weights"],
                    candidate_evaluation_budget=config["search"]["repair_candidate_evaluation_budget"],
                    prepared=prepared,
                )
                candidate = repair_destroyed_state_protocol_r(
                    instance,
                    context,
                    initial.state,
                    destroyed,
                    weights=config["search"]["objective_weights"],
                    candidate_evaluation_budget=config["search"]["repair_candidate_evaluation_budget"],
                    prepared=prepared,
                )
                comparison = parity_report(reference, candidate)
                rows.append(
                    {
                        "instance_id": record["instance_id"],
                        "task_group_id": record["task_group_id"],
                        "cell": cell,
                        "operator": operator,
                        "destroy_ratio": ratio,
                        "destroy_set": list(destroyed),
                        "reference_runtime_s": reference.runtime_s,
                        "candidate_runtime_s": candidate.runtime_s,
                        "speedup": reference.runtime_s / max(candidate.runtime_s, 1e-12),
                        **comparison,
                    }
                )
        for method in ("random_lns", "alns_online", "alns_segmented"):
            mode, single, scheme = _method_spec(method)
            search_config = _search_config(
                config,
                config["search"]["profile_seed"],
                "fixed_iterations",
                update_scheme=scheme,
                iterations=5,
            )
            reference = run_search_v2(
                instance,
                context,
                build_hybrid_load_balanced_initializer(instance, context, config["search"]["objective_weights"]),
                search_config,
                mode=mode,
                task_group_id=record["task_group_id"],
                difficulty=cell,
                split="train",
                single_operator=single,
            )
            candidate = run_search_protocol_r(
                instance,
                context,
                build_hybrid_load_balanced_initializer(instance, context, config["search"]["objective_weights"]),
                search_config,
                mode=mode,
                task_group_id=record["task_group_id"],
                difficulty=cell,
                split="train",
                single_operator=single,
            )
            rows.append(
                {
                    "instance_id": record["instance_id"],
                    "task_group_id": record["task_group_id"],
                    "cell": cell,
                    "prefix_method": method,
                    "prefix_transition_match": transition_signature(reference.trace)
                    == transition_signature(candidate.trace),
                    "passed": transition_signature(reference.trace)
                    == transition_signature(candidate.trace),
                }
            )
    payload = {
        "version": "a4b-protocol-r-profile-cell-v1",
        "cell": cell,
        "passed": all(item["passed"] for item in rows),
        "rows": rows,
        "config_sha256": config["config_sha256"],
        "manifest_sha256": manifest_hash,
        "source_sha256": _source_sha256(),
        "protocol_document_sha256": sha256_file(PROTOCOL_PATH),
        "record": _record(config, cell=cell),
    }
    payload["shard_sha256"] = canonical_hash(payload)
    _write_json(output / "profile_shards" / f"{cell}.json", payload)
    return payload


def merge_profile(config, output):
    paths = [output / "profile_shards" / f"{cell}.json" for cell in CELLS]
    if not all(path.exists() for path in paths):
        raise RuntimeError("missing Protocol-R profile shard")
    shards = [json.loads(path.read_text()) for path in paths]
    for shard in shards:
        saved = shard.pop("shard_sha256")
        if canonical_hash(shard) != saved or not shard["passed"]:
            raise RuntimeError("invalid or failed Protocol-R profile shard")
        shard["shard_sha256"] = saved
    rows = [row for shard in shards for row in shard["rows"]]
    repairs = [row for row in rows if "speedup" in row]
    candidate_times = [float(row["candidate_runtime_s"]) for row in repairs]
    speedups = [float(row["speedup"]) for row in repairs]
    scale_times = [float(row["candidate_runtime_s"]) for row in repairs if row["cell"] == "scale"]
    gates = conjunctive_gate(
        {
            "all_semantic_parity": all(row["passed"] for row in rows),
            "global_median_neighborhood": statistics.median(candidate_times)
            <= float(config["gates"]["speed"]["global_median_complete_neighborhood_s_max"]),
            "global_median_speedup": statistics.median(speedups)
            >= float(config["gates"]["speed"]["global_median_repair_path_speedup_min"]),
            "scale_median_neighborhood": statistics.median(scale_times)
            <= float(config["gates"]["speed"]["scale_median_complete_neighborhood_s_max"]),
        }
    )
    payload = {
        "version": "a4b-protocol-r-profile-gate-v1",
        "passed": gates["passed"],
        "gates": gates,
        "repair_scenarios": len(repairs),
        "prefix_scenarios": len(rows) - len(repairs),
        "global_median_candidate_s": statistics.median(candidate_times),
        "global_median_speedup": statistics.median(speedups),
        "scale_median_candidate_s": statistics.median(scale_times),
        "shard_sha256": {shard["cell"]: shard["shard_sha256"] for shard in shards},
        "record": _record(config),
    }
    payload["gate_sha256"] = canonical_hash(payload)
    _write_json(output / "profile_gate.json", payload)
    if not payload["passed"]:
        raise RuntimeError("Protocol-R profile/parity/speed gate failed")
    return payload


def calibrate_cell(config, context, output, cell):
    profile = json.loads((output / "profile_gate.json").read_text())
    if not profile["passed"]:
        raise RuntimeError("Protocol-R calibration blocked by profile gate")
    items, manifest_hash = _calibration_items(config, context, output)
    traces, rows = [], []
    seed = int(config["search"]["profile_seed"])
    for record, instance in items:
        if record["cell_id"] != cell:
            continue
        for method in tuple(f"single:{item}" for item in DESTROY_OPERATORS) + (
            "alns_online",
            "alns_segmented",
        ):
            trace = _run_trace(config, context, record, instance, method, seed, "fixed_iterations")
            traces.append(trace)
            rows.extend(_metric_rows(record, trace))
        for method in ("random_lns", "alns_online", "alns_segmented"):
            trace = _run_trace(config, context, record, instance, method, seed, "fixed_time")
            traces.append(trace)
            rows.extend(_metric_rows(record, trace))
    payload = {
        "version": "a4b-protocol-r-calibration-cell-v1",
        "cell": cell,
        "passed": True,
        "traces": traces,
        "rows": rows,
        "config_sha256": config["config_sha256"],
        "manifest_sha256": manifest_hash,
        "source_sha256": _source_sha256(),
        "protocol_document_sha256": sha256_file(PROTOCOL_PATH),
        "record": _record(config, cell=cell),
    }
    payload["shard_sha256"] = canonical_hash(payload)
    _write_json(output / "calibration_shards" / f"{cell}.json", payload)
    return payload


def _coverage(rows, method, budget):
    selected = [
        row
        for row in rows
        if row["method"] == method
        and row["view"] == "fixed_end_to_end_time"
        and float(row["budget"]) == budget
    ]
    groups = {}
    for row in selected:
        groups.setdefault(row["task_group_id"], []).append(bool(row["verified"]))
    return sum(sum(values) / len(values) for values in groups.values()), len(groups)


def merge_train_gate(config, output):
    shards = [
        json.loads((output / "calibration_shards" / f"{cell}.json").read_text())
        for cell in CELLS
    ]
    traces, merge = deterministic_merge_shards(
        shards, cells=CELLS, methods=CALIBRATION_METHODS
    )
    expected = config["expected_matrices"]["train_calibration"]
    matrix = validate_trace_matrix(
        traces,
        fixed_iteration_expected=int(expected["fixed_iteration_traces"]),
        fixed_time_expected=int(expected["fixed_time_traces"]),
        required_iterations=30,
    )
    rows = [row for shard in shards for row in shard["rows"]]
    exact_rows = [row for row in rows if row["view"] == "fixed_iterations" and row["budget"] == 30]
    def selection_key(method):
        selected = [row for row in exact_rows if row["method"] == method]
        coverage = sum(row["verified"] for row in selected)
        objectives = [float(row["objective"]) for row in selected if row["verified"]]
        return (-coverage, statistics.mean(objectives) if objectives else float("inf"), method)
    selected_operator_method = min(
        (f"single:{item}" for item in DESTROY_OPERATORS), key=selection_key
    )
    selected_scheme_method = min(("alns_online", "alns_segmented"), key=selection_key)
    selected_operator = selected_operator_method.split(":", 1)[1]
    selected_scheme = selected_scheme_method.split("_", 1)[1]

    timed = [row for row in rows if row["view"] == "fixed_end_to_end_time"]
    opportunity = config["gates"]["search_opportunity"]
    opportunity_checks = {}
    for method in ("random_lns", "alns_online", "alns_segmented"):
        at1 = [row["completed_neighborhoods"] for row in timed if row["method"] == method and row["budget"] == 1.0]
        at3 = [row["completed_neighborhoods"] for row in timed if row["method"] == method and row["budget"] == 3.0]
        opportunity_checks[f"{method}_median_1s"] = statistics.median(at1) >= opportunity["per_method_global_median_completed_by_1s_min"]
        opportunity_checks[f"{method}_median_3s"] = statistics.median(at3) >= opportunity["per_method_global_median_completed_by_3s_min"]
        opportunity_checks[f"{method}_fraction_3s"] = sum(value >= 1 for value in at3) / len(at3) >= opportunity["per_method_fraction_with_one_completed_by_3s_min"]
        for cell in CELLS:
            cell_values = [row["completed_neighborhoods"] for row in timed if row["method"] == method and row["budget"] == 3.0 and row["cell_id"] == cell]
            minimum = opportunity["iid_small_and_iid_medium_median_completed_by_3s_min"] if cell in {"iid_small", "iid_medium"} else opportunity["per_method_per_cell_median_completed_by_3s_min"]
            opportunity_checks[f"{method}_{cell}_median_3s"] = statistics.median(cell_values) >= minimum

    random1, groups1 = _coverage(rows, "random_lns", 1.0)
    alns1, _ = _coverage(rows, selected_scheme_method, 1.0)
    random3, groups3 = _coverage(rows, "random_lns", 3.0)
    alns3, _ = _coverage(rows, selected_scheme_method, 3.0)
    alns_gate = config["gates"]["alns_non_systematic_weakness"]
    challenge_groups = {
        trace["task_group_id"]
        for trace in traces
        if "-challenge-group-" in trace["task_group_id"]
    }
    chosen_alns = [trace for trace in traces if trace["method_id"] == selected_scheme_method and trace["budget_mode"] == "fixed_time"]
    random_traces = [trace for trace in traces if trace["method_id"] == "random_lns" and trace["budget_mode"] == "fixed_time"]

    challenge_instances = {}
    for trace in traces:
        if "-challenge-group-" in trace["task_group_id"]:
            challenge_instances.setdefault(trace["task_group_id"], set()).add(trace["instance_id"])

    def recovered_groups(method):
        recovered_instances = {
            (trace["task_group_id"], trace["instance_id"])
            for trace in traces
            if trace["method_id"] == method
            and trace["budget_mode"] == "fixed_time"
            and "-challenge-group-" in trace["task_group_id"]
            and any(float(event["elapsed_s"]) <= 3.0 for event in trace["incumbents"])
        }
        return {
            group
            for group, instances in challenge_instances.items()
            if all((group, instance) in recovered_instances for instance in instances)
        }

    random_recovered = recovered_groups("random_lns")
    alns_recovered = recovered_groups(selected_scheme_method)
    recovered = random_recovered | alns_recovered

    completed_alns = [sum(step["completed_before_cutoff"] for step in trace["steps"]) for trace in chosen_alns]
    completed_random = [sum(step["completed_before_cutoff"] for step in trace["steps"]) for trace in random_traces]
    operator_choices = {step["operator"] for trace in chosen_alns for step in trace["steps"] if step["completed_before_cutoff"]}

    by_instance = {}
    for trace in random_traces + chosen_alns:
        by_instance.setdefault(trace["instance_id"], []).append(trace)
    paired_integrals = []
    for instance_id, instance_traces in by_instance.items():
        values = [
            float(event["objective"])
            for trace in instance_traces
            for event in trace["incumbents"]
            if float(event["elapsed_s"]) <= 3.0
        ]
        if values:
            target = min(values)
            reference = max(values)
            if reference <= target:
                reference = target + max(1.0, abs(target) * 0.1)
        else:
            target, reference = 0.0, 1.0
        per_method = {}
        for trace in instance_traces:
            per_method[trace["method_id"]] = normalized_primal_integral(
                trace["incumbents"], 3.0, target=target, reference=reference
            )
        if "random_lns" in per_method and selected_scheme_method in per_method:
            paired_integrals.append(
                per_method[selected_scheme_method] - per_method["random_lns"]
            )
    npi_excess = statistics.mean(paired_integrals) if paired_integrals else float("inf")

    unequal_reward_seen = any(
        len({float(step["reward"]) for step in trace["steps"]}) > 1
        for trace in chosen_alns
    )
    nonuniform_weights_seen = any(
        len({round(float(value), 12) for value in trace["operator_weights"].values()}) > 1
        for trace in chosen_alns
    )
    nontrivial_cells = {"iid_medium", "dense_precedence", "resource_bottleneck", "tight_windows", "scale"}
    cell_outcomes = {
        cell: {
            (
                tuple(step["destroy_set"]),
                step["candidate_state_sha256"],
            )
            for trace in chosen_alns
            if trace["difficulty"] == cell
            for step in trace["steps"]
            if step["completed_before_cutoff"]
        }
        for cell in nontrivial_cells
    }
    random_by_instance = {trace["instance_id"]: trace for trace in random_traces}
    alns_by_instance = {trace["instance_id"]: trace for trace in chosen_alns}
    identitywise_difference = any(
        (
            [event["state_sha256"] for event in random_by_instance[instance]["incumbents"]],
            [step["destroy_set"] for step in random_by_instance[instance]["steps"]],
        )
        != (
            [event["state_sha256"] for event in alns_by_instance[instance]["incumbents"]],
            [step["destroy_set"] for step in alns_by_instance[instance]["steps"]],
        )
        for instance in set(random_by_instance) & set(alns_by_instance)
    )
    checks = {
        "matrix": bool(matrix["passed"]),
        "opportunity": all(opportunity_checks.values()),
        "coverage_1s": random1 - alns1 <= alns_gate["paired_coverage_group_deficit_max_at_1s"],
        "coverage_3s": random3 - alns3 <= alns_gate["paired_coverage_group_deficit_max_at_3s"],
        "normalized_primal_integral": npi_excess <= alns_gate["paired_mean_normalized_primal_integral_excess_max"],
        "completed_ratio": statistics.median(completed_alns) / max(statistics.median(completed_random), 1) >= alns_gate["median_completed_neighborhood_ratio_vs_random_min"],
        "challenge_count": len(challenge_groups) >= config["gates"]["challenge_recovery"]["eligible_groups_min"],
        "challenge_recovery": len(recovered) >= config["gates"]["challenge_recovery"]["ordinary_method_recovered_groups_min"] and len(recovered) / max(len(challenge_groups), 1) >= config["gates"]["challenge_recovery"]["ordinary_method_recovery_rate_min"],
        "alns_recovery_deficit": len(random_recovered) - len(alns_recovered) <= alns_gate["eligible_recovery_group_deficit_vs_random_max"],
        "operator_count": len(operator_choices) >= config["gates"]["operator_dependence"]["selected_operators_min"],
        "operator_weight_response": (not unequal_reward_seen) or nonuniform_weights_seen,
        "cell_outcome_dependence": all(len(values) >= 2 for values in cell_outcomes.values()),
        "not_identitywise_equal_to_random": identitywise_difference,
    }
    gate = conjunctive_gate(checks)
    payload = {
        "version": "a4b-protocol-r-train-gate-v1",
        "passed": gate["passed"],
        "gates": gate,
        "opportunity_checks": opportunity_checks,
        "matrix": matrix,
        "selected_operator": selected_operator,
        "selected_update_scheme": selected_scheme,
        "challenge_groups": sorted(challenge_groups),
        "recovered_challenge_groups": sorted(recovered),
        "random_recovered_challenge_groups": sorted(random_recovered),
        "alns_recovered_challenge_groups": sorted(alns_recovered),
        "paired_mean_normalized_primal_integral_excess": npi_excess,
        "operator_choices": sorted(operator_choices),
        "coverage": {"random_1s": random1, "alns_1s": alns1, "random_3s": random3, "alns_3s": alns3, "groups_1s": groups1, "groups_3s": groups3},
        "merge": merge,
        "record": _record(config),
    }
    payload["gate_sha256"] = canonical_hash(payload)
    _write_jsonl(output / "train_calibration_traces.jsonl", traces)
    _write_jsonl(output / "train_calibration_metrics.jsonl", sorted(rows, key=lambda row: (row["cell_id"], row["instance_id"], row["method"], row["search_seed"], row["view"], row["budget"])))
    _write_json(output / "train_gate.json", payload)
    if not payload["passed"]:
        raise RuntimeError("Protocol-R train gate failed")
    return payload


def smoke(config, context, output):
    gate = json.loads((output / "train_gate.json").read_text())
    if not gate["passed"]:
        raise RuntimeError("Protocol-R smoke blocked by train gate")
    items, manifest_hash = load_protocol_r_items(output / "corpus", "development", context)
    record, instance = items[0]
    traces = []
    for method in DEVELOPMENT_METHODS:
        traces.append(_run_trace(config, context, record, instance, method, config["search"]["random_seeds"][0], "fixed_iterations", selected_operator=gate["selected_operator"], selected_scheme=gate["selected_update_scheme"], iterations=1))
    payload = {"version": "a4b-protocol-r-smoke-v1", "passed": len(traces) == 4, "traces": traces, "manifest_sha256": manifest_hash, "record": _record(config)}
    payload["smoke_sha256"] = canonical_hash(payload)
    _write_json(output / "smoke.json", payload)
    return payload


def development_cell(config, context, output, cell):
    gate = json.loads((output / "train_gate.json").read_text())
    smoke_record = json.loads((output / "smoke.json").read_text())
    if not gate["passed"] or not smoke_record["passed"]:
        raise RuntimeError("Protocol-R development blocked by train/smoke gate")
    items, manifest_hash = load_protocol_r_items(output / "corpus", "development", context)
    traces, rows = [], []
    for record, instance in items:
        if record["cell_id"] != cell:
            continue
        for seed in config["search"]["random_seeds"]:
            for method in DEVELOPMENT_METHODS:
                for budget_mode in ("fixed_iterations", "fixed_time"):
                    trace = _run_trace(config, context, record, instance, method, seed, budget_mode, selected_operator=gate["selected_operator"], selected_scheme=gate["selected_update_scheme"])
                    traces.append(trace)
                    rows.extend(_metric_rows(record, trace))
    payload = {"version": "a4b-protocol-r-development-cell-v1", "cell": cell, "passed": True, "traces": traces, "rows": rows, "config_sha256": config["config_sha256"], "manifest_sha256": manifest_hash, "source_sha256": _source_sha256(), "protocol_document_sha256": sha256_file(PROTOCOL_PATH), "record": _record(config, cell=cell)}
    payload["shard_sha256"] = canonical_hash(payload)
    _write_json(output / "development_shards" / f"{cell}.json", payload)
    return payload


def finalize(config, output):
    shards = [json.loads((output / "development_shards" / f"{cell}.json").read_text()) for cell in CELLS]
    traces, merge = deterministic_merge_shards(shards, cells=CELLS, methods=DEVELOPMENT_METHODS)
    expected = config["expected_matrices"]["development"]
    matrix = validate_trace_matrix(traces, fixed_iteration_expected=expected["fixed_iteration_traces"], fixed_time_expected=expected["fixed_time_traces"], required_iterations=30)
    rows = [row for shard in shards for row in shard["rows"]]
    replay_ok = all(trace["trace_sha256"] == canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"}) for trace in traces)
    passed = matrix["passed"] and len(rows) == expected["metric_rows_total"] and replay_ok
    payload = {"version": "a4b-protocol-r-finalize-candidate-v1", "status": "IMPLEMENTED_PROTOCOL_R_DEVELOPMENT_COMPLETE" if passed else "PROTOCOL_R_FINALIZE_FAILED", "passed": passed, "formal_action": "HOLD_A4B_LEARNED_DESTROY_TRAINING", "matrix": matrix, "metric_rows": len(rows), "replay_passed": replay_ok, "merge": merge, "record": _record(config)}
    payload["finalize_sha256"] = canonical_hash(payload)
    _write_jsonl(output / "development_traces.jsonl", traces)
    _write_jsonl(output / "development_metrics.jsonl", sorted(rows, key=lambda row: (row["cell_id"], row["instance_id"], row["method"], row["search_seed"], row["view"], row["budget"])))
    _write_json(output / "finalize.json", payload)
    if not passed:
        raise RuntimeError("Protocol-R replay/aggregate failed")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "audit-draft",
            "fixture-parity",
            "preflight",
            "generate",
            "profile-cell",
            "merge-profile",
            "calibrate-cell",
            "merge-train-gate",
            "smoke",
            "development-cell",
            "finalize",
        ),
    )
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    review_only = args.command in {"audit-draft", "fixture-parity"}
    config = load_protocol_r_config(CONFIG_PATH, allow_draft=review_only)
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    if review_only:
        payload = audit_draft(config, context)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(1)
        return
    _require_runtime(config)
    output = args.output.resolve()
    if args.command == "preflight":
        payload = {"version": "a4b-protocol-r-preflight-v1", "passed": True, "record": _record(config)}
        payload["preflight_sha256"] = canonical_hash(payload)
        _write_json(output / "preflight.json", payload)
    elif args.command == "generate":
        generate_protocol_r_data(config, output / "corpus", context, execution_authorized=True)
    elif args.command == "profile-cell":
        profile_cell(config, context, output, args.cell)
    elif args.command == "merge-profile":
        merge_profile(config, output)
    elif args.command == "calibrate-cell":
        calibrate_cell(config, context, output, args.cell)
    elif args.command == "merge-train-gate":
        merge_train_gate(config, output)
    elif args.command == "smoke":
        smoke(config, context, output)
    elif args.command == "development-cell":
        development_cell(config, context, output, args.cell)
    elif args.command == "finalize":
        finalize(config, output)


if __name__ == "__main__":
    main()
