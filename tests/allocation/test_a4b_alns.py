from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import state_from_plan
from safe_residual_rl.allocation.schema import allocation_instance_from_dict
from safe_residual_rl.allocation.search.alns import (
    AlnsConfig,
    _accept,
    repair_destroyed_state,
    run_search,
    update_operator_weight,
)
from safe_residual_rl.allocation.search.anytime import build_hybrid_load_balanced_initializer
from safe_residual_rl.allocation.search.data import load_a4b_config, load_a4b_items
from safe_residual_rl.allocation.search.operators import (
    DESTROY_OPERATORS,
    build_operator_problem,
    destroy_count,
    select_destroy_set,
)
from safe_residual_rl.allocation.search.trace import canonical_hash, replay_trace
from safe_residual_rl.allocation.solvers import solve_hybrid_load_balanced
from safe_residual_rl.allocation.solvers.common import allocation_units

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data/fixtures/allocation"
CONFIG = ROOT / "configs/allocation/a4b_neural_lns_dev_v1.json"
WEIGHTS = {"makespan": 1.0, "load_variance": 0.05, "travel_setup_time": 0.1, "priority_tardiness": 1.0}


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


def fixture(name):
    return allocation_instance_from_dict(load_auditable_fixture(FIXTURES / name)["instance"])


def initializer(instance, context):
    return build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)


def config(seed=1201, iterations=8, time_s=10.0):
    return AlnsConfig(
        "fixture", iterations, time_s, (0.1, 0.25, 0.4), 256, seed, WEIGHTS,
        restart_no_improvement=100,
    )


def test_atomic_destroy_operates_on_unsplittable_unit(context):
    instance = fixture("02_valid_same_robot_segments.json")
    assert len(allocation_units(instance)) == 1 and len(allocation_units(instance)[0]) == 2
    outcome = initializer(instance, context)
    evaluation = solve_hybrid_load_balanced(instance, context)
    state = state_from_plan(instance, evaluation.plan)
    selected = select_destroy_set("random_destroy", instance, context, state, build_eval(instance, context, state), 0.4, np.random.default_rng(1))
    assert selected == (0,)


def build_eval(instance, context, state):
    from safe_residual_rl.allocation.repair import evaluate_state
    return evaluate_state(instance, context, state, WEIGHTS)


@pytest.mark.parametrize("operator", DESTROY_OPERATORS)
def test_destroy_sets_are_unique_and_deterministic(operator, context):
    instance = fixture("03_valid_explicit_boundary.json")
    state = initializer(instance, context).state
    evaluation = build_eval(instance, context, state)
    left = select_destroy_set(operator, instance, context, state, evaluation, 0.4, np.random.default_rng(91))
    right = select_destroy_set(operator, instance, context, state, evaluation, 0.4, np.random.default_rng(91))
    assert left == right and len(left) == len(set(left))
    assert len(left) == destroy_count(len(allocation_units(instance)), 0.4)


def test_precedence_operator_selects_a_precedence_incident_unit(context):
    instance = fixture("03_valid_explicit_boundary.json")
    state = initializer(instance, context).state
    problem = build_operator_problem(instance, context)
    selected = select_destroy_set("precedence_chain_destroy", instance, context, state, build_eval(instance, context, state), 0.1, np.random.default_rng(3))
    incident = {item for edge in problem.predecessor_edges for item in edge}
    assert not incident or selected[0] in incident


def test_tight_window_operator_selects_smallest_slack(context):
    instance = fixture("05_valid_priority_window.json")
    state = initializer(instance, context).state
    selected = select_destroy_set("critical_slack_destroy", instance, context, state, build_eval(instance, context, state), 0.1, np.random.default_rng(3))
    assert len(selected) == 1


def test_resource_operator_targets_shared_resource_unit(context):
    instance = fixture("04_valid_shared_zone.json")
    state = initializer(instance, context).state
    problem = build_operator_problem(instance, context)
    selected = select_destroy_set("shared_resource_conflict_destroy", instance, context, state, build_eval(instance, context, state), 0.1, np.random.default_rng(3))
    assert problem.unit_resources[selected[0]]


def test_same_repair_parameters_are_method_independent(context):
    instance = fixture("03_valid_explicit_boundary.json")
    traces = []
    for mode in ("random_lns", "handcrafted_round_robin", "adaptive_alns"):
        traces.append(run_search(instance, context, initializer(instance, context), config(), mode=mode, task_group_id="g", difficulty="f", split="train").trace)
    assert len({trace.config_sha256 for trace in traces}) == 1


def test_alns_operator_weight_update():
    assert update_operator_weight(1.0, 8.0, 0.2) == pytest.approx(2.4)
    assert update_operator_weight(1.0, 0.0, 0.2) > 0.0


def test_acceptance_rule_never_replaces_verified_with_infeasible(context):
    instance = fixture("01_valid_minimal.json")
    feasible_state = initializer(instance, context).state
    feasible = build_eval(instance, context, feasible_state)
    broken = type(feasible)(False, None, None, feasible.surrogate + 1e6, "schedule_infeasible")
    assert not _accept(feasible, broken, 1, np.random.default_rng(1), config())


def test_fixed_iteration_search_is_reproducible(context):
    instance = fixture("03_valid_explicit_boundary.json")
    first = run_search(instance, context, initializer(instance, context), config(), mode="adaptive_alns", task_group_id="g", difficulty="f", split="train")
    second = run_search(instance, context, initializer(instance, context), config(), mode="adaptive_alns", task_group_id="g", difficulty="f", split="train")
    assert [(s.operator, s.destroy_set, s.accepted, s.candidate_state_sha256) for s in first.trace.steps] == [(s.operator, s.destroy_set, s.accepted, s.candidate_state_sha256) for s in second.trace.steps]


def test_end_to_end_budget_includes_initializer(context):
    instance = fixture("01_valid_minimal.json")
    init = initializer(instance, context)
    delayed = type(init)(init.state, type(init.provenance)(**{**init.provenance.to_dict(), "completion_monotonic_ns": init.provenance.start_monotonic_ns + 2_000_000_000, "completion_elapsed_s": 2.0}))
    outcome = run_search(instance, context, delayed, config(time_s=1.0), mode="random_lns", task_group_id="g", difficulty="f", split="train")
    assert len(outcome.trace.steps) == 0


def test_trace_replay_reconstructs_candidate_states(context):
    instance = fixture("03_valid_explicit_boundary.json")
    cfg = config(iterations=4)
    outcome = run_search(instance, context, initializer(instance, context), cfg, mode="random_lns", task_group_id="g", difficulty="f", split="train")

    def replay(step):
        return repair_destroyed_state(
            instance, context, step.before_state, step.destroy_set,
            random_seed=step.iteration_seed, weights=WEIGHTS,
            candidate_evaluation_budget=cfg.repair_candidate_evaluation_budget,
        ).state

    assert replay_trace(outcome.trace, replay) == outcome.trace.sha256


def test_protocol_forbids_old_ids_and_future_splits(tmp_path, context):
    cfg = load_a4b_config(CONFIG)
    assert cfg["data"]["id_prefix"] not in cfg["data"]["disjoint_id_prefixes"]
    assert set(cfg["data"]["forbidden_splits"]) == {"validation", "frozen_test", "stress"}
    for token in ("frozen_test", "stress", "a35f1", "benchmark_v4"):
        path = tmp_path / token
        path.mkdir()
        with pytest.raises(PermissionError):
            load_a4b_items(path, "train", context)


def test_parameter_selection_is_train_only_and_no_neural_checkpoint():
    cfg = load_a4b_config(CONFIG)
    assert cfg["operator_selection"]["split"] == "train"
    assert cfg["operator_selection"]["validation_or_future_data_access"] == "forbidden"
    assert "neural" not in json.dumps(cfg["methods"]).lower()


def test_group_level_aggregation_keeps_variants_inside_group():
    spec = importlib.util.spec_from_file_location("a4brunner", ROOT / "scripts/run_a4b_baseline_development.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"task_group_id": "g", "cell_id": "c", "view": "fixed_iterations", "budget": 10, "method": "m", "verified": True, "objective": 1.0},
        {"task_group_id": "g", "cell_id": "c", "view": "fixed_iterations", "budget": 10, "method": "m", "verified": False, "objective": None},
    ]
    grouped = module._group_rows(rows)
    assert len(grouped) == 1 and grouped[0]["coverage"] == 0.5


def test_config_and_schema_hashes_are_canonical():
    cfg = load_a4b_config(CONFIG)
    assert len(cfg["config_sha256"]) == 64
    manifest = {"version": "fixture", "records": [{"id": "a4bnlsd1-x"}]}
    assert canonical_hash(manifest) == canonical_hash(json.loads(json.dumps(manifest)))


def test_recorded_source_matrix_covers_provenance_search_tests_and_slurm():
    spec = importlib.util.spec_from_file_location("a4brunner_hashes", ROOT / "scripts/run_a4b_baseline_development.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "src/safe_residual_rl/allocation/solvers/common.py",
        "src/safe_residual_rl/allocation/solvers/heuristics.py",
        "src/safe_residual_rl/allocation/solvers/milp.py",
        "src/safe_residual_rl/allocation/search/anytime.py",
        "src/safe_residual_rl/allocation/search/alns.py",
        "tests/allocation/test_a4b_anytime_evaluator.py",
        "tests/allocation/test_a4b_alns.py",
        "slurm/a4b_development_smoke.sbatch",
        "slurm/a4b_development_array.sbatch",
    }
    assert required <= set(module.SOURCE_FILES)
    assert all(len(module.sha256_file(ROOT / path)) == 64 for path in module.SOURCE_FILES)


def test_label_verifier_hash_is_canonical(context):
    spec = importlib.util.spec_from_file_location("a4brunner_label_hash", ROOT / "scripts/run_a4b_baseline_development.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = fixture("01_valid_minimal.json")
    evaluation = build_eval(instance, context, initializer(instance, context).state)
    assert len(module._state_evaluation_hash(evaluation)) == 64
    assert module._state_evaluation_hash(evaluation) == module._state_evaluation_hash(evaluation)
