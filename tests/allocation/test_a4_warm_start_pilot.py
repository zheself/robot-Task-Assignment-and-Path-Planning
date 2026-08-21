from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import InitializerState, evaluate_state, identical_repair, state_from_plan
from safe_residual_rl.allocation.schema import allocation_instance_from_dict
from safe_residual_rl.allocation.solvers import solve_hybrid_load_balanced
from safe_residual_rl.allocation.solvers.common import allocation_units, edge_mask_and_costs
from safe_residual_rl.allocation.warm_start import canonical_digest, load_a4_config, load_a4_items, load_locked_preprocessor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data/fixtures/allocation"
CONFIG = ROOT / "configs/allocation/a4_warm_start_pilot_v1.json"
WEIGHTS = {"makespan": 1.0, "load_variance": 0.05, "travel_setup_time": 0.1, "priority_tardiness": 1.0}


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


def fixture(name):
    return allocation_instance_from_dict(load_auditable_fixture(FIXTURES / name)["instance"])


def feasible_state(instance, context):
    result = solve_hybrid_load_balanced(instance, context)
    assert result.plan is not None
    return state_from_plan(instance, result.plan)


def test_protocol_is_development_only_and_exactly_grouped():
    cfg = load_a4_config(CONFIG)
    assert cfg["data"]["splits"] == ["train", "validation"]
    assert cfg["data"]["validation_groups_total"] == 24
    assert cfg["statistics"]["overall_one_group_equivalent"] == pytest.approx(1 / 24)
    assert cfg["repair"]["fixed_iterations"] == [10, 50, 100]
    assert cfg["repair"]["fixed_end_to_end_time_s"] == [0.5, 1.0, 3.0]


@pytest.mark.parametrize("bad", ["frozen_test", "stress", "a35f1", "benchmark_v4"])
def test_loader_rejects_old_or_frozen_paths(tmp_path, context, bad):
    path = tmp_path / bad; path.mkdir()
    with pytest.raises(PermissionError):
        load_a4_items(path, "validation", context)


def test_cold_start_is_legal_and_incomplete(context):
    instance = fixture("01_valid_minimal.json")
    robots = tuple(sorted(x.id for x in instance.robots))
    state = InitializerState(tuple(None for _ in allocation_units(instance)), tuple((r, ()) for r in robots))
    result = evaluate_state(instance, context, state, WEIGHTS)
    assert not result.verified and result.failure_reason == "initializer_incomplete"


def test_identical_repair_constructs_from_cold_start(context):
    instance = fixture("01_valid_minimal.json")
    robots = tuple(sorted(x.id for x in instance.robots))
    state = InitializerState(tuple(None for _ in allocation_units(instance)), tuple((r, ()) for r in robots))
    result = identical_repair(instance, context, state, iterations=10, random_seed=401, weights=WEIGHTS)
    assert result.final_evaluation.verified
    assert result.first_feasible_iteration == 1


def test_fixed_iteration_is_deterministic(context):
    instance = fixture("03_valid_explicit_boundary.json")
    state = feasible_state(instance, context)
    left = identical_repair(instance, context, state, iterations=10, random_seed=907, weights=WEIGHTS)
    right = identical_repair(instance, context, state, iterations=10, random_seed=907, weights=WEIGHTS)
    assert left.final_state == right.final_state
    assert [(x.operator, x.accepted, x.verified) for x in left.trace] == [(x.operator, x.accepted, x.verified) for x in right.trace]


def test_timeout_stops_without_dropping_row(context):
    instance = fixture("03_valid_explicit_boundary.json")
    result = identical_repair(instance, context, feasible_state(instance, context), iterations=100, random_seed=401, weights=WEIGHTS, time_limit_s=0.0)
    assert result.timed_out and result.iterations_completed == 0
    assert result.final_evaluation.verified


def test_trace_is_complete(context):
    instance = fixture("03_valid_explicit_boundary.json")
    result = identical_repair(instance, context, feasible_state(instance, context), iterations=10, random_seed=401, weights=WEIGHTS)
    assert len(result.trace) == 10
    assert [x.iteration for x in result.trace] == list(range(1, 11))
    assert all(x.operator and x.elapsed_s >= 0 for x in result.trace)


def test_atomic_unit_never_splits(context):
    instance = fixture("02_valid_same_robot_segments.json")
    assert len(allocation_units(instance)) == 1
    result = identical_repair(instance, context, feasible_state(instance, context), iterations=20, random_seed=401, weights=WEIGHTS)
    assert len(result.final_state.assignments) == 1
    assert result.final_state.assignments[0] is not None


def test_hard_infeasible_robot_is_never_selected(context):
    instance = fixture("01_valid_minimal.json")
    bad = replace(instance.robots[0], id="bad", capabilities=())
    instance = replace(instance, robots=(instance.robots[0], bad))
    robots = tuple(sorted(x.id for x in instance.robots))
    state = InitializerState((None,), tuple((r, ()) for r in robots))
    result = identical_repair(instance, context, state, iterations=20, random_seed=401, weights=WEIGHTS)
    assert result.final_state.assignments[0] != "bad"


def test_all_operator_names_are_initializer_agnostic(context):
    instance = fixture("03_valid_explicit_boundary.json")
    result = identical_repair(instance, context, feasible_state(instance, context), iterations=12, random_seed=401, weights=WEIGHTS)
    assert {x.operator for x in result.trace} == {"random_destroy", "worst_load_destroy", "atomic_reassign", "robot_local_relocate", "robot_local_swap", "precedence_safe_reorder"}


def test_state_from_plan_round_trip_is_verified(context):
    instance = fixture("04_valid_shared_zone.json")
    state = feasible_state(instance, context)
    assert evaluate_state(instance, context, state, WEIGHTS).verified


def test_assignment_and_order_edit_metrics_are_bounded(context):
    instance = fixture("03_valid_explicit_boundary.json")
    result = identical_repair(instance, context, feasible_state(instance, context), iterations=10, random_seed=401, weights=WEIGHTS)
    assert 0 <= result.initializer_assignment_retention <= 1
    assert result.assignment_modifications >= 0 and result.order_modifications >= 0


def test_target_time_is_recorded_from_initial_feasible(context):
    instance = fixture("01_valid_minimal.json")
    state = feasible_state(instance, context)
    initial = evaluate_state(instance, context, state, WEIGHTS)
    result = identical_repair(instance, context, state, iterations=1, random_seed=401, weights=WEIGHTS, target_score=float(initial.objective) + 1)
    assert result.time_to_target_s == 0.0


def test_preprocessor_artifact_hash_contract(tmp_path):
    payload = {"version": "a3-5-immutable-inference-preprocessing-v1", "accessed_splits": ["train"], "forbidden_splits_accessed": [], "vocabulary": {}, "normalizer": {}}
    payload["artifact_sha256"] = canonical_digest(payload)
    path = tmp_path / "meta.json"; path.write_text(json.dumps(payload))
    payload["artifact_sha256"] = "bad"; path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        load_locked_preprocessor(path)


def test_config_has_matched_three_seed_pairing():
    cfg = load_a4_config(CONFIG)
    assert cfg["immutable_initializers"]["seed_pairing"] == [[101, 101], [211, 211], [307, 307]]


def test_repair_config_has_no_initializer_specific_branch():
    repair = load_a4_config(CONFIG)["repair"]
    text = json.dumps(repair, sort_keys=True)
    assert "pair_pointer" not in text and "load_balanced" not in text and "static" not in text


def test_group_level_gate_classification_helpers():
    spec = importlib.util.spec_from_file_location("a4runner", ROOT / "scripts/run_a4_warm_start_pilot.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = [{"initializer": "pair_pointer_seed_101", "verified": True}, {"initializer": "pair_pointer_seed_101", "verified": False}]
    assert module._coverage(rows, "pair_pointer_seed_101") == 0.5


def test_same_seed_same_input_repair_digest(context):
    instance = fixture("03_valid_explicit_boundary.json")
    state = feasible_state(instance, context)
    first = identical_repair(instance, context, state, iterations=10, random_seed=401, weights=WEIGHTS).to_dict()
    second = identical_repair(instance, context, state, iterations=10, random_seed=401, weights=WEIGHTS).to_dict()
    for value in (first, second):
        value.pop("repair_runtime_s"); [item.pop("elapsed_s") for item in value["trace"]]
    assert canonical_digest(first) == canonical_digest(second)
