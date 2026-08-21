from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import InitializerState
from safe_residual_rl.allocation.schema import TimeWindow, allocation_instance_from_dict
from safe_residual_rl.allocation.search.anytime import (
    InitializerOutcome,
    InitializerProvenance,
    adapt_solver_initializer,
)
from safe_residual_rl.allocation.search.trace import (
    IncumbentEvent,
    SearchTrace,
    best_at_budget,
    canonical_hash,
    state_hash,
)
from safe_residual_rl.allocation.solvers.common import SolverResult, allocation_units
from safe_residual_rl.allocation.solvers import solve_hybrid_load_balanced

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data/fixtures/allocation"
WEIGHTS = {"makespan": 1.0, "load_variance": 0.05, "travel_setup_time": 0.1, "priority_tardiness": 1.0}


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


@pytest.fixture
def instance():
    return allocation_instance_from_dict(load_auditable_fixture(FIXTURES / "01_valid_minimal.json")["instance"])


def _state(instance):
    robot = instance.robots[0].id
    return InitializerState(tuple(robot for _ in allocation_units(instance)), ((robot, tuple(range(len(allocation_units(instance))))),))


def _trace(instance, events, *, completion=0.05, returned=2.0):
    state = _state(instance)
    provenance = InitializerProvenance(
        "hybrid_load_balanced", "hybrid_load_balanced", "feasible", True, False, None,
        state_hash(state), True, None, 1_000_000_000,
        1_000_000_000 + int(completion * 1e9), completion,
    )
    incumbents = tuple(
        IncumbentEvent(
            iteration, 1_000_000_000 + int(elapsed * 1e9), elapsed, objective,
            state, state_hash(state), canonical_hash([objective]), canonical_hash([True, objective]),
        )
        for iteration, elapsed, objective in events
    )
    return SearchTrace(
        "a4b-search-trace-v1", "fixture", instance.instance_id, "group", "fixture", "train",
        "random_lns", 7, "time.monotonic_ns", 1_000_000_000,
        1_000_000_000 + int(returned * 1e9), provenance.to_dict(), state, (), "cfg", (), incumbents,
        "fixture_return",
    )


def test_incumbent_before_cutoff_survives_late_function_return(instance):
    trace = _trace(instance, [(1, 0.40, 10.0)], returned=1.25)
    snapshot = best_at_budget(trace, 0.5)
    assert snapshot.verified and snapshot.objective == 10.0


def test_incumbent_first_found_after_cutoff_is_not_backdated(instance):
    trace = _trace(instance, [(1, 0.60, 10.0)], returned=0.75)
    assert not best_at_budget(trace, 0.5).verified


def test_multiple_incumbents_return_best_known_at_each_budget(instance):
    trace = _trace(instance, [(1, 0.2, 10.0), (2, 0.4, 8.0), (3, 0.8, 7.0)])
    assert best_at_budget(trace, 0.3).objective == 10.0
    assert best_at_budget(trace, 0.5).objective == 8.0
    assert best_at_budget(trace, 1.0).objective == 7.0


def test_initializer_consuming_budget_is_an_explicit_failure(instance):
    trace = _trace(instance, [], completion=0.75)
    snapshot = best_at_budget(trace, 0.5)
    assert not snapshot.verified and snapshot.failure_reason == "initializer_timeout"


def test_milp_without_incumbent_has_truthful_provenance(instance, context):
    result = SolverResult("assignment-milp", "limit", None, 1.0, None, None, None, ("NO_INCUMBENT",))
    outcome = adapt_solver_initializer(instance, context, "hybrid_assignment_milp", result, WEIGHTS, start_monotonic_ns=10, completion_monotonic_ns=20)
    assert not outcome.provenance.has_true_incumbent
    assert outcome.provenance.actual_initializer == "hybrid_assignment_milp"
    assert not outcome.provenance.fallback_used


def test_milp_assignment_incumbent_survives_scheduler_infeasibility(instance, context):
    impossible = replace(
        instance,
        segments=(replace(instance.segments[0], time_window=TimeWindow(0.0, 0.01)),),
    )
    result = SolverResult(
        "assignment-milp", "schedule_infeasible", None, 0.2, 1.0, 1.0, 0.0,
        ("SCHEDULER_REJECTED",), ((0, impossible.robots[0].id),),
    )
    outcome = adapt_solver_initializer(impossible, context, "hybrid_assignment_milp", result, WEIGHTS, start_monotonic_ns=10, completion_monotonic_ns=20)
    assert outcome.provenance.has_true_incumbent
    assert outcome.provenance.actual_initializer.endswith("_assignment_incumbent")
    assert not outcome.provenance.verifier_feasible
    assert outcome.provenance.initializer_plan_hash == state_hash(outcome.state)


def test_hybrid_assignment_is_not_erased_when_scheduler_rejects(context):
    """Regression for the post-smoke ordinary-LNS provenance audit.

    Destroy/repair needs the real complete assignment even when neither
    deterministic scheduling policy can produce a verified plan.  Replacing
    it with an all-None state makes neighborhood search unable to repair units
    outside the current destroy set.
    """
    base = allocation_instance_from_dict(
        load_auditable_fixture(FIXTURES / "03_valid_explicit_boundary.json")["instance"]
    )
    impossible = replace(
        base,
        segments=(base.segments[0], replace(base.segments[1], time_window=TimeWindow(0.0, 3.0))),
        robots=(replace(base.robots[0], nominal_cartesian_speed_m_s=1.0),),
    )
    result = solve_hybrid_load_balanced(impossible, context)
    assert result.status == "schedule_infeasible"
    assert result.plan is None
    assert result.assignment_incumbent == ((0, impossible.robots[0].id), (1, impossible.robots[0].id))
    outcome = adapt_solver_initializer(
        impossible,
        context,
        "hybrid_load_balanced",
        result,
        WEIGHTS,
        start_monotonic_ns=10,
        completion_monotonic_ns=20,
    )
    assert outcome.provenance.has_true_incumbent
    assert outcome.provenance.actual_initializer == "hybrid_load_balanced_assignment_incumbent"
    assert all(robot is not None for robot in outcome.state.assignments)


def test_fallback_label_never_claims_milp(instance, context):
    fallback_result = SolverResult("load", "feasible", None, 0.1, None, None, None, ())
    fallback = InitializerOutcome(
        _state(instance),
        InitializerProvenance(
            "hybrid_load_balanced", "hybrid_load_balanced", "feasible", True, False, None,
            state_hash(_state(instance)), True, None, 10, 15, 5e-9,
        ),
    )
    failed = SolverResult("milp", "limit", None, 1.0, None, None, None, ())
    outcome = adapt_solver_initializer(instance, context, "hybrid_assignment_milp", failed, WEIGHTS, start_monotonic_ns=10, completion_monotonic_ns=20, fallback=fallback)
    assert outcome.provenance.fallback_used
    assert outcome.provenance.actual_initializer == "hybrid_load_balanced"
    assert "hybrid_assignment_milp" in outcome.provenance.fallback_reason


def test_monotonic_clock_regression_is_rejected(instance, context):
    result = SolverResult("milp", "limit", None, 1.0, None, None, None, ())
    with pytest.raises(ValueError, match="backwards"):
        adapt_solver_initializer(instance, context, "milp", result, WEIGHTS, start_monotonic_ns=20, completion_monotonic_ns=10)


def test_timeout_failure_remains_a_snapshot_denominator_row(instance):
    snapshots = [best_at_budget(_trace(instance, [(1, 0.2, 1.0)]), 0.5), best_at_budget(_trace(instance, []), 0.5)]
    assert len(snapshots) == 2
    assert sum(item.verified for item in snapshots) / len(snapshots) == 0.5
