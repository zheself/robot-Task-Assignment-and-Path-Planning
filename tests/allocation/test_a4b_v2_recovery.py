from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import InitializerState
from safe_residual_rl.allocation.schema import HandoffPolicy, TimeWindow, allocation_instance_from_dict
from safe_residual_rl.allocation.search.anytime import build_hybrid_load_balanced_initializer
from safe_residual_rl.allocation.search.alns_v2 import (
    AlnsV2Config,
    _accept,
    repair_destroyed_state_v2,
    run_search_v2,
    update_segmented_weights,
)
from safe_residual_rl.allocation.search.data_v2 import guard_a4b_v2_path, load_a4b_v2_config
from safe_residual_rl.allocation.search.diagnostics import analyze_state, evaluate_state_timed
from safe_residual_rl.allocation.search.metrics import normalized_primal_integral
from safe_residual_rl.allocation.search.operators_v2 import select_destroy_set_v2
from safe_residual_rl.allocation.search.trace import state_from_dict, state_hash

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data/fixtures/allocation"
CONFIG = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2.json"
AMENDMENT = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_runtime_amendment.json"
WATCHDOG_AMENDMENT = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_watchdog_amendment.json"
PARALLEL_AMENDMENT = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_parallel_amendment.json"
RECOVERY_AMENDMENT = ROOT / "configs/allocation/a4b_ordinary_lns_dev_v2_metadata_recovery_amendment.json"
WEIGHTS = {"makespan": 1.0, "load_variance": 0.05, "travel_setup_time": 0.1, "priority_tardiness": 1.0}


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


def fixture(name):
    return allocation_instance_from_dict(load_auditable_fixture(FIXTURES / name)["instance"])


def two_unit_instance(*, resource=False, tight_second=False):
    base = fixture("01_valid_minimal.json")
    template = base.segments[0]
    resource_value = fixture("04_valid_shared_zone.json").resources[0] if resource else None
    segments = []
    for index, offset in enumerate((0.0, 0.15)):
        start, end = (offset, 0.0, 0.0), (offset + 0.1, 0.0, 0.0)
        segments.append(
            replace(
                template,
                id=f"seg-{index}", parent_curve_id=f"curve-{index}", segment_index=0,
                sampled_curve_m=(start, end),
                start_pose=replace(template.start_pose, position_m=start),
                end_pose=replace(template.end_pose, position_m=end),
                predecessor_ids=(), handoff_policy=HandoffPolicy.FREE,
                time_window=TimeWindow(0.0, 8.0 if tight_second and index == 1 else 100.0),
                shared_resource_ids=(() if resource_value is None else (resource_value.id,)),
            )
        )
    robot_0 = base.robots[0]
    robot_1 = replace(
        robot_0, id="robot-1",
        base_pose=replace(robot_0.base_pose, position_m=(0.55, 0.0, 0.0)),
    )
    return replace(
        base, instance_id="fixture-two-unit", segments=tuple(segments),
        robots=(robot_0, robot_1), resources=(() if resource_value is None else (resource_value,)),
    )


def cfg(*, iterations=4, mode="fixed_iterations", deadline_s=None):
    return AlnsV2Config(
        protocol_id="fixture-v2",
        budget_mode=mode,
        iterations=iterations,
        end_to_end_time_s=deadline_s,
        safety_watchdog_s=20.0,
        destroy_ratios=(0.1, 0.25, 0.4),
        repair_candidate_evaluation_budget=256,
        random_seed=17,
        objective_weights=WEIGHTS,
        update_scheme="segmented",
        segment_length=2,
    )


def test_fixed_iteration_completes_exact_k(context):
    instance = fixture("03_valid_explicit_boundary.json")
    outcome = run_search_v2(instance, context, build_hybrid_load_balanced_initializer(instance, context, WEIGHTS), cfg(iterations=5), mode="adaptive_alns", task_group_id="g", difficulty="fixture", split="train")
    assert outcome.trace["iterations_completed"] == 5
    assert len(outcome.trace["steps"]) == 5
    assert outcome.trace["fixed_iteration_complete"]


def test_time_budget_never_claims_fixed_iteration_complete(context):
    instance = fixture("03_valid_explicit_boundary.json")
    outcome = run_search_v2(instance, context, build_hybrid_load_balanced_initializer(instance, context, WEIGHTS), cfg(iterations=30, mode="fixed_time", deadline_s=1e-12), mode="random_lns", task_group_id="g", difficulty="fixture", split="train")
    assert not outcome.trace["fixed_iteration_complete"]
    assert outcome.trace["termination_reason"] == "end_to_end_time_budget"


def test_detailed_evaluation_matches_existing_verifier_semantics(context):
    instance = fixture("04_valid_shared_zone.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    timed = evaluate_state_timed(instance, context, initial.state, WEIGHTS)
    assert timed.evaluation.verified == initial.provenance.verifier_feasible
    assert timed.timing.scheduler_s >= 0 and timed.timing.verifier_s >= 0


def test_structured_diagnostics_identify_window_failure(context):
    instance = fixture("01_valid_minimal.json")
    impossible = replace(instance, segments=(replace(instance.segments[0], time_window=TimeWindow(0.0, 0.001)),))
    assigned = InitializerState(("robot-0",), (("robot-0", (0,)),))
    diagnostic = analyze_state(impossible, context, assigned)
    assert diagnostic.vector.missing_units == 0
    assert diagnostic.vector.time_window_lateness_s > 0


def test_failure_aware_destroy_is_deterministic_and_atomic(context):
    instance = fixture("03_valid_explicit_boundary.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    detailed = evaluate_state_timed(instance, context, initial.state, WEIGHTS)
    left = select_destroy_set_v2("precedence_chain_destroy", instance, context, initial.state, detailed, 0.4, np.random.default_rng(3))
    right = select_destroy_set_v2("precedence_chain_destroy", instance, context, initial.state, detailed, 0.4, np.random.default_rng(3))
    assert left == right and len(left) == len(set(left))


def test_segmented_weight_update_uses_score_per_use():
    weights = {"good": 1.0, "bad": 1.0}
    updated = update_segmented_weights(weights, {"good": 8.0, "bad": 0.0}, {"good": 2, "bad": 2}, 0.2)
    assert updated["good"] > updated["bad"]
    assert updated["bad"] > 0


def test_normalized_primal_integral_penalizes_no_incumbent():
    assert normalized_primal_integral([], 3.0, target=10.0, reference=20.0) == pytest.approx(1.0)
    events = [{"elapsed_s": 1.0, "objective": 20.0}, {"elapsed_s": 2.0, "objective": 10.0}]
    assert normalized_primal_integral(events, 3.0, target=10.0, reference=20.0) == pytest.approx(2.0 / 3.0)


def runner_module():
    spec = importlib.util.spec_from_file_location(
        "a4b_v2_runner", ROOT / "scripts/run_a4b_ordinary_lns_development.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_truncated_trace_cannot_emit_fixed_iteration_rows():
    module = runner_module()
    config = load_a4b_v2_config(CONFIG)
    record = {
        "split": "train", "cell_id": "fixture", "task_group_id": "g",
        "variant_index": 0, "instance_id": "i",
    }
    trace = {
        "fixed_iteration_complete": False, "iterations_completed": 2,
        "incumbents": [], "steps": [], "trace_sha256": "x",
        "initializer": {"actual_initializer": "hybrid_load_balanced", "fallback_used": False},
    }
    assert module._fixed_iteration_rows(config, record, "random_lns", 1, trace) == []


def test_post_cutoff_incumbent_is_not_credited_by_v2_snapshot():
    module = runner_module()
    events = [
        {"iteration": 1, "elapsed_s": 0.4, "objective": 10.0},
        {"iteration": 2, "elapsed_s": 0.6, "objective": 1.0},
    ]
    assert module._best(events, cutoff=0.5)["objective"] == 10.0


def test_v2_paths_and_splits_are_isolated(tmp_path):
    config = load_a4b_v2_config(CONFIG)
    assert config["data"]["id_prefix"] == "a4blnsd2"
    assert config["data"]["splits"] == ["train", "development"]
    for token in ("frozen_test", "stress", "a35f1", "a4_warm_start_pilot_v1", "a4b_neural_lns_dev_v1"):
        with pytest.raises(PermissionError):
            guard_a4b_v2_path(tmp_path / token)


def test_repair_records_assignment_and_order_edits(context):
    instance = fixture("03_valid_explicit_boundary.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    repaired = repair_destroyed_state_v2(
        instance, context, initial.state, (0,), weights=WEIGHTS,
        candidate_evaluation_budget=256,
    )
    assert repaired.assignment_edits >= 0 and repaired.order_edits >= 0
    assert repaired.total_modified_units >= max(repaired.assignment_edits, repaired.order_edits)


def test_shared_repair_recovers_precedence_counterexample(context):
    instance = fixture("03_valid_explicit_boundary.json")
    bad = InitializerState(("robot-0", "robot-0"), (("robot-0", (1, 0)),))
    before = evaluate_state_timed(instance, context, bad, WEIGHTS)
    repaired = repair_destroyed_state_v2(
        instance, context, bad, (0,), weights=WEIGHTS,
        candidate_evaluation_budget=256,
    )
    assert not before.evaluation.verified and before.evaluation.failure_reason == "precedence_failure"
    assert repaired.evaluation.evaluation.verified


def test_shared_repair_recovers_tight_window_counterexample(context):
    instance = two_unit_instance(tight_second=True)
    bad = InitializerState(
        ("robot-0", "robot-0"), (("robot-0", (0, 1)), ("robot-1", ()))
    )
    before = evaluate_state_timed(instance, context, bad, WEIGHTS)
    repaired = repair_destroyed_state_v2(
        instance, context, bad, (1,), weights=WEIGHTS,
        candidate_evaluation_budget=256,
    )
    assert not before.evaluation.verified and before.evaluation.failure_reason == "time_window_failure"
    assert repaired.evaluation.evaluation.verified


def test_shared_repair_recovers_resource_window_counterexample(context):
    instance = two_unit_instance(resource=True, tight_second=True)
    bad = InitializerState(
        ("robot-0", "robot-0"), (("robot-0", (0, 1)), ("robot-1", ()))
    )
    before = evaluate_state_timed(instance, context, bad, WEIGHTS)
    repaired = repair_destroyed_state_v2(
        instance, context, bad, (1,), weights=WEIGHTS,
        candidate_evaluation_budget=256,
    )
    assert not before.evaluation.verified
    assert repaired.evaluation.evaluation.verified
    assert instance.segments[1].shared_resource_ids


def test_v2_acceptance_never_replaces_verified_with_infeasible(context):
    feasible_instance = fixture("01_valid_minimal.json")
    feasible_state = build_hybrid_load_balanced_initializer(feasible_instance, context, WEIGHTS).state
    feasible = evaluate_state_timed(feasible_instance, context, feasible_state, WEIGHTS)
    impossible = replace(
        feasible_instance,
        segments=(replace(feasible_instance.segments[0], time_window=TimeWindow(0.0, 0.001)),),
    )
    assigned = InitializerState(("robot-0",), (("robot-0", (0,)),))
    infeasible = evaluate_state_timed(impossible, context, assigned, WEIGHTS)
    assert feasible.evaluation.verified and not infeasible.evaluation.verified
    assert not _accept(feasible, infeasible, 1, np.random.default_rng(1), cfg())


def test_group_aggregation_retains_failed_variant():
    module = runner_module()
    rows = [
        {"task_group_id": "g", "cell_id": "c", "view": "fixed_iterations", "budget": 10, "method": "m", "verified": True, "objective": 1.0},
        {"task_group_id": "g", "cell_id": "c", "view": "fixed_iterations", "budget": 10, "method": "m", "verified": False, "objective": None},
    ]
    grouped = module._group_rows(rows)
    assert len(grouped) == 1 and grouped[0]["coverage"] == 0.5 and grouped[0]["failure_rows"] == 1


def test_source_hash_matrix_is_complete_and_no_neural_method():
    module = runner_module()
    config = load_a4b_v2_config(CONFIG)
    required = {
        "src/safe_residual_rl/allocation/search/diagnostics.py",
        "src/safe_residual_rl/allocation/search/operators_v2.py",
        "src/safe_residual_rl/allocation/search/alns_v2.py",
        "src/safe_residual_rl/allocation/search/data_v2.py",
        "scripts/run_a4b_ordinary_lns_development.py",
        "scripts/run_a4b_cpu_worker.sh",
        "scripts/submit_a4b_v2_parallel_chain.sh",
        "scripts/submit_a4b_v2_metadata_recovery_chain.sh",
        "slurm/a4b_v2_calibration_array.sbatch",
        "slurm/a4b_v2_calibration_packed.sbatch",
        "slurm/a4b_v2_merge_gate_labels.sbatch",
        "slurm/a4b_v2_recover_merge_gate_labels.sbatch",
        "slurm/a4b_v2_development_packed.sbatch",
        "slurm/a4b_v2_development_array.sbatch",
        "slurm/a4b_v2_finalize.sbatch",
    }
    assert required <= set(module.SOURCE_FILES)
    assert all(len(module.sha256_file(ROOT / path)) == 64 for path in module.SOURCE_FILES)
    assert "neural" not in json.dumps(config["methods"]).lower()


def test_runtime_amendment_is_cpu_only_and_matches_base_config():
    config = load_a4b_v2_config(CONFIG)
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["status"] == "FROZEN_BEFORE_FIRST_SEARCH_EXECUTION"
    assert amendment["base_config_sha256"] == config["config_sha256"]
    hardware = amendment["runtime_hardware"]
    assert hardware["partition"] == "normal" and hardware["account"] == "v-chengwy"
    assert hardware["nodelist"] == "sist_gpu59"
    assert not hardware["gpu_requested"] and not hardware["gres_requested"]
    for name in (
        "a4b_v2_train_gate.sbatch", "a4b_v2_development_smoke.sbatch",
        "a4b_v2_development_array.sbatch", "a4b_v2_finalize.sbatch",
    ):
        source = (ROOT / "slurm" / name).read_text()
        assert "--partition=normal" in source
        assert "--account=v-chengwy" in source
        assert "--nodelist=sist_gpu59" in source
        assert "--gres" not in source
    wrapper = (ROOT / "scripts/run_a4b_cpu_worker.sh").read_text()
    assert 'export CUDA_VISIBLE_DEVICES=""' in wrapper
    assert "taskset --cpu-list" in wrapper


def test_failed_train_watchdog_amendment_is_narrow_and_frozen():
    config = load_a4b_v2_config(CONFIG)
    amendment = json.loads(WATCHDOG_AMENDMENT.read_text())
    assert amendment["status"] == "FROZEN_AFTER_FAILED_TRAIN_GATE_BEFORE_RESTART"
    assert amendment["base_config_sha256"] == config["config_sha256"]
    assert amendment["scope"] == "operational_safety_watchdog_only"
    assert amendment["failed_job_id"] == "981071"
    assert amendment["observed_failure"]["development_accessed"] is False
    assert amendment["replacement"]["search.safety_watchdog_s"] == 1800.0
    assert config["search"]["fixed_end_to_end_time_s"] == [0.5, 1.0, 3.0]


def test_runner_applies_watchdog_amendment_without_changing_base_protocol():
    source = (ROOT / "scripts/run_a4b_ordinary_lns_development.py").read_text()
    assert "WATCHDOG_AMENDMENT_PATH" in source
    assert 'config["search"]["safety_watchdog_s"] = float(' in source
    config = load_a4b_v2_config(CONFIG)
    assert config["search"]["safety_watchdog_s"] == 60.0


def parallel_config():
    config = load_a4b_v2_config(CONFIG)
    amendment = json.loads(PARALLEL_AMENDMENT.read_text())
    config["expected_calibration_matrix"] = amendment["expected_calibration_matrix"]
    config["parallel_execution"] = amendment["parallel_execution"]
    return config


def recovery_config(module):
    config = parallel_config()
    runtime = json.loads(AMENDMENT.read_text())
    config["runtime_amendment_sha256"] = module.sha256_file(AMENDMENT)
    config["watchdog_amendment_sha256"] = module.sha256_file(WATCHDOG_AMENDMENT)
    config["parallel_amendment_sha256"] = module.sha256_file(PARALLEL_AMENDMENT)
    config["runtime_hardware"] = runtime["runtime_hardware"]
    recovery = json.loads(RECOVERY_AMENDMENT.read_text())
    module._validate_recovery_amendment(config, recovery)
    config["recovery_amendment_sha256"] = module.sha256_file(RECOVERY_AMENDMENT)
    config["metadata_recovery"] = recovery
    return config


def fake_calibration_traces(module):
    config = parallel_config()
    methods = list(module._calibration_method_order(config))
    timed_methods = ["adaptive_alns_online", "adaptive_alns_segmented", "random_lns"]
    traces = []
    for cell in config["data"]["cells"]:
        for instance_index in range(4):
            instance = f"{cell}-{instance_index}"
            for method in methods:
                traces.append({
                    "instance_id": instance, "method_id": method, "random_seed": 2203,
                    "budget_mode": "fixed_iterations", "difficulty": cell,
                    "fixed_iteration_complete": True, "iterations_completed": 30,
                })
            for method in timed_methods:
                traces.append({
                    "instance_id": instance, "method_id": method, "random_seed": 2203,
                    "budget_mode": "fixed_time", "difficulty": cell,
                    "fixed_iteration_complete": False, "iterations_completed": 0,
                })
    return traces


def test_parallel_amendment_freezes_six_single_cpu_cell_workers():
    config = load_a4b_v2_config(CONFIG)
    amendment = json.loads(PARALLEL_AMENDMENT.read_text())
    parallel = amendment["parallel_execution"]
    assert amendment["status"] == "FROZEN_DURING_SERIAL_CALIBRATION_BEFORE_PARALLEL_SUBMISSION"
    assert amendment["base_config_sha256"] == config["config_sha256"]
    assert parallel["worker_count"] == 6
    assert parallel["cpus_per_worker"] == 1
    assert parallel["cells"] == config["data"]["cells"]
    assert parallel["train_and_development_corresponding_concurrency"] == 6


def test_calibration_cell_and_global_matrix_are_exact_and_complete():
    module = runner_module()
    config = parallel_config()
    traces = fake_calibration_traces(module)
    assert module._calibration_matrix_status(config, traces) == {
        "traces": 312, "fixed_iterations": 240, "fixed_time": 72,
        "fixed_iteration_complete": 240, "unique_identities": 312, "passed": True,
    }
    for cell in config["data"]["cells"]:
        selected = [trace for trace in traces if trace["difficulty"] == cell]
        status = module._calibration_matrix_status(config, selected, cell=cell)
        assert status["traces"] == 52 and status["fixed_iterations"] == 40
        assert status["fixed_time"] == 12 and status["passed"]


def test_calibration_matrix_rejects_duplicate_and_incomplete_trace():
    module = runner_module()
    config = parallel_config()
    traces = fake_calibration_traces(module)
    duplicate = list(traces)
    duplicate[-1] = dict(duplicate[0])
    assert not module._calibration_matrix_status(config, duplicate)["passed"]
    incomplete = [dict(trace) for trace in traces]
    fixed = next(trace for trace in incomplete if trace["budget_mode"] == "fixed_iterations")
    fixed["fixed_iteration_complete"] = False
    fixed["iterations_completed"] = 29
    assert not module._calibration_matrix_status(config, incomplete)["passed"]


def test_merge_worker_affinity_gate_rejects_shared_cpu():
    module = runner_module()
    distinct = [{"cpu_affinity": [cpu]} for cpu in (0, 1, 2, 28, 29, 30)]
    status = module._worker_affinity_matrix_status(distinct, 6)
    assert status["passed"] and status["unique_worker_cpus"] == 6
    duplicated = distinct[:-1] + [{"cpu_affinity": [0]}]
    assert not module._worker_affinity_matrix_status(duplicated, 6)["passed"]


def test_merge_rejects_missing_shards_and_trace_order_is_deterministic(tmp_path):
    module = runner_module()
    config = parallel_config()
    with pytest.raises(RuntimeError, match="missing duplicate or foreign"):
        module.merge_calibration(config, None, tmp_path)
    traces = fake_calibration_traces(module)
    left = sorted(reversed(traces), key=lambda trace: module._calibration_trace_sort_key(config, trace))
    right = sorted(traces, key=lambda trace: module._calibration_trace_sort_key(config, trace))
    assert [module._trace_identity(trace) for trace in left] == [module._trace_identity(trace) for trace in right]


def test_serial_and_sharded_execution_have_same_transition_signature(context):
    module = runner_module()
    instance = fixture("03_valid_explicit_boundary.json")
    initializer = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    outcomes = [
        run_search_v2(
            instance, context, initializer, cfg(iterations=4), mode="random_lns",
            task_group_id="g", difficulty="fixture", split="train",
        ).trace
        for _ in range(2)
    ]
    assert module._trace_transition_signature(outcomes[0]) == module._trace_transition_signature(outcomes[1])


def test_cpu_worker_enforces_single_thread_environment_and_affinity():
    command = [
        str(ROOT / "scripts/run_a4b_cpu_worker.sh"), str(ROOT / ".venv/bin/python"),
        "-c", "import json,os; print(json.dumps([len(os.sched_getaffinity(0)),os.environ['OMP_NUM_THREADS'],os.environ['MKL_NUM_THREADS'],os.environ['OPENBLAS_NUM_THREADS'],os.environ['NUMEXPR_NUM_THREADS'],os.environ['CUDA_VISIBLE_DEVICES']]))",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    assert json.loads(completed.stdout) == [1, "1", "1", "1", "1", ""]
    assert "worker_bound_cpu=" in completed.stderr and "worker_loadavg=" in completed.stderr


def test_cpu_worker_index_selects_distinct_cpus_from_allocation():
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("test process has fewer than two CPUs")
    selected = []
    for index in (0, 1):
        environment = dict(os.environ, A4B_WORKER_INDEX=str(index))
        completed = subprocess.run(
            [
                str(ROOT / "scripts/run_a4b_cpu_worker.sh"),
                str(ROOT / ".venv/bin/python"), "-c",
                "import os; print(next(iter(os.sched_getaffinity(0))))",
            ],
            cwd=ROOT, check=True, text=True, capture_output=True, env=environment,
        )
        selected.append(int(completed.stdout.strip()))
    assert len(set(selected)) == 2


def test_parallel_afterok_chain_and_cell_arrays_are_fail_closed():
    submit = (ROOT / "scripts/submit_a4b_v2_parallel_chain.sh").read_text()
    assert submit.count("--dependency=\"afterok:") == 4
    assert "A4B_CONFIRM_PARALLEL_SUBMIT" in submit
    assert "already queued or running" in submit
    calibration = (ROOT / "slurm/a4b_v2_calibration_array.sbatch").read_text()
    development = (ROOT / "slurm/a4b_v2_development_array.sbatch").read_text()
    for source in (calibration, development):
        assert "--array=0-5" in source
        assert "--cpus-per-task=1" in source
        assert "run_a4b_cpu_worker.sh" in source
        assert "--gres" not in source
    packed_calibration = (ROOT / "slurm/a4b_v2_calibration_packed.sbatch").read_text()
    packed_development = (ROOT / "slurm/a4b_v2_development_packed.sbatch").read_text()
    for source in (packed_calibration, packed_development):
        assert "--cpus-per-task=6" in source
        assert "srun" not in source
        assert "A4B_WORKER_INDEX" in source
        assert "ulimit -v 4194304" in source
        assert 'for index in "${!CELLS[@]}"' in source
        assert "run_a4b_cpu_worker.sh" in source
        assert "--gres" not in source
    assert "a4b_v2_calibration_packed.sbatch" in submit
    assert "a4b_v2_development_packed.sbatch" in submit


def test_parallel_provenance_hashes_match_base_amendments_and_manifest():
    module = runner_module()
    config = load_a4b_v2_config(CONFIG)
    parallel = json.loads(PARALLEL_AMENDMENT.read_text())
    manifest = json.loads((ROOT / "outputs/phase1_allocation/a4b_ordinary_lns_dev_v2/corpus/manifest.json").read_text())
    assert parallel["base_config_sha256"] == config["config_sha256"]
    assert parallel["runtime_amendment_sha256"] == module.sha256_file(AMENDMENT)
    assert parallel["watchdog_amendment_sha256"] == module.sha256_file(WATCHDOG_AMENDMENT)
    assert len(manifest["manifest_sha256"]) == 64


def test_metadata_recovery_amendment_freezes_failed_artifacts_and_no_search():
    module = runner_module()
    config = recovery_config(module)
    recovery = config["metadata_recovery"]
    assert recovery["status"] == "FROZEN_AFTER_JOB_984111_METADATA_ONLY_FAILURE_BEFORE_RECOVERY"
    assert recovery["scope"] == "metadata_reconstruction_only_no_search"
    assert recovery["failed_job"]["job_id"] == "984111"
    assert recovery["failed_job"]["runner_sha256"] == "8850981378a0b53d8516d0957ec2ba7bdadf07659285f2352bb284c84ff4614e"
    assert not recovery["recovery_rules"]["search_execution_allowed"]
    assert not recovery["recovery_rules"]["trace_rewrite_allowed"]
    assert set(recovery["artifacts"]) == set(config["data"]["cells"])
    for artifact in recovery["artifacts"].values():
        assert module.sha256_file(ROOT / artifact["trace_path"]) == artifact["trace_sha256"]
        assert module.sha256_file(ROOT / artifact["log_path"]) == artifact["log_sha256"]


def test_record_uses_real_numpy_version_attribute():
    module = runner_module()
    config = recovery_config(module)
    record = module._record(config)
    assert record["dependencies"]["numpy"] == np.__version__
    source = (ROOT / "scripts/run_a4b_ordinary_lns_development.py").read_text()
    assert "np.__version()" not in source
    assert "np.__version__" in source


def test_metadata_recovery_dry_run_reconstructs_complete_matrix_without_search(monkeypatch):
    module = runner_module()
    config = recovery_config(module)
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    output = ROOT / "outputs/phase1_allocation/a4b_ordinary_lns_dev_v2"
    assert not list((output / "train_calibration_shards").glob("*.json"))
    writes = {}

    def capture(path, value):
        writes[Path(path).name] = value

    def forbidden_search(*args, **kwargs):
        raise AssertionError("metadata recovery called search")

    monkeypatch.setattr(module, "_write_json", capture)
    monkeypatch.setattr(module, "_run", forbidden_search)
    recovered = module.recover_calibration_metadata(config, context, output)
    assert recovered["status"] == "RECOVERED_WITHOUT_SEARCH"
    assert recovered["trace_count"] == 312 and recovered["row_count"] == 792
    assert recovered["matrix"]["fixed_iteration_complete"] == 240
    assert recovered["transition_signature_sha256"] == config["metadata_recovery"]["expected_matrix"]["transition_signature_sha256"]
    assert set(writes) == {
        "iid_small.json", "iid_medium.json", "dense_precedence.json",
        "resource_bottleneck.json", "tight_windows.json", "scale.json",
        "calibration_metadata_recovery_record.json",
    }
    for cell in config["data"]["cells"]:
        payload = writes[f"{cell}.json"]
        assert payload["counts"]["traces"] == 52
        assert payload["counts"]["fixed_iteration_complete"] == 40
        assert payload["record"]["recovery_status"] == "RECONSTRUCTED_NOT_NATIVE"
        assert payload["record"]["worker_ended_unix_s"] is None
        traces = module._read_jsonl(
            output / "train_calibration_shards" / f"{cell}.jsonl"
        )
        module._validate_calibration_shard(
            config, context, output, cell, payload, traces
        )


def test_metadata_recovery_rejects_changed_trace_hash(monkeypatch):
    module = runner_module()
    config = recovery_config(module)
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    output = ROOT / "outputs/phase1_allocation/a4b_ordinary_lns_dev_v2"
    config["metadata_recovery"] = json.loads(json.dumps(config["metadata_recovery"]))
    config["metadata_recovery"]["artifacts"]["iid_small"]["trace_sha256"] = "0" * 64
    monkeypatch.setattr(module, "_write_json", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="artifact hash mismatch: iid_small"):
        module.recover_calibration_metadata(config, context, output)


def test_metadata_recovery_submission_chain_is_fail_closed_and_skips_calibration():
    submit = (ROOT / "scripts/submit_a4b_v2_metadata_recovery_chain.sh").read_text()
    assert submit.count("--dependency=\"afterok:") == 3
    assert "A4B_CONFIRM_METADATA_RECOVERY_SUBMIT" in submit
    assert "a4b_v2_recover_merge_gate_labels.sbatch" in submit
    assert "a4b_v2_calibration_packed.sbatch" not in submit
    recovery_job = (ROOT / "slurm/a4b_v2_recover_merge_gate_labels.sbatch").read_text()
    commands = [
        "recover-calibration-metadata", "merge-calibration", "train-gate", "labels"
    ]
    positions = [recovery_job.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "calibrate-cell" not in recovery_job and " calibrate" not in recovery_job


def test_candidate_budget_exhaustion_is_explicit(context):
    instance = fixture("03_valid_explicit_boundary.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    repaired = repair_destroyed_state_v2(
        instance, context, initial.state, (0, 1), weights=WEIGHTS,
        candidate_evaluation_budget=1,
    )
    assert repaired.budget_exhausted
    assert repaired.candidate_evaluations == 1


def test_all_controlled_modes_share_repair_config(context):
    instance = fixture("03_valid_explicit_boundary.json")
    hashes = set()
    for mode in ("random_lns", "handcrafted_round_robin", "adaptive_alns"):
        outcome = run_search_v2(
            instance, context,
            build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
            cfg(iterations=3), mode=mode, task_group_id="g", difficulty="fixture", split="train",
        )
        hashes.add(outcome.trace["config_sha256"])
    assert len(hashes) == 1


def test_fixed_iteration_seed_reproduces_transitions(context):
    instance = fixture("03_valid_explicit_boundary.json")
    outcomes = [
        run_search_v2(
            instance, context,
            build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
            cfg(iterations=4), mode="adaptive_alns", task_group_id="g",
            difficulty="fixture", split="train",
        )
        for _ in range(2)
    ]
    signature = lambda outcome: [
        (step["operator"], step["destroy_set"], step["candidate_state_sha256"], step["accepted"])
        for step in outcome.trace["steps"]
    ]
    assert signature(outcomes[0]) == signature(outcomes[1])


def test_exact_trace_replays_shared_repair_transitions(context):
    instance = fixture("03_valid_explicit_boundary.json")
    configuration = cfg(iterations=4)
    outcome = run_search_v2(
        instance, context,
        build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
        configuration, mode="random_lns", task_group_id="g",
        difficulty="fixture", split="train",
    )
    for step in outcome.trace["steps"]:
        before = state_from_dict(step["before_state"])
        replayed = repair_destroyed_state_v2(
            instance, context, before, step["destroy_set"], weights=WEIGHTS,
            candidate_evaluation_budget=configuration.repair_candidate_evaluation_budget,
        )
        assert state_hash(replayed.state) == step["candidate_state_sha256"]
        assert replayed.evaluation.evaluation.verified == step["after_verified"]


def test_infeasible_current_policy_can_fail_closed(context):
    instance = fixture("01_valid_minimal.json")
    impossible = replace(
        instance,
        segments=(replace(instance.segments[0], time_window=TimeWindow(0.0, 0.001)),),
    )
    initial = build_hybrid_load_balanced_initializer(impossible, context, WEIGHTS)
    restricted = replace(cfg(iterations=1), allow_infeasible_current=False)
    with pytest.raises(ValueError, match="infeasible initializer"):
        run_search_v2(
            impossible, context, initial, restricted, mode="random_lns",
            task_group_id="g", difficulty="fixture", split="train",
        )
