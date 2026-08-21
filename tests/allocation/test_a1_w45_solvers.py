from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.schema import (
    AllocationPlan,
    HandoffPolicy,
    PLAN_SCHEMA_VERSION,
    ScheduledSegment,
    allocation_instance_from_dict,
)
from safe_residual_rl.allocation.scheduling import build_schedule
from safe_residual_rl.allocation.solvers import (
    load_solver_protocol,
    solve_assignment_milp,
    solve_greedy,
    solve_hungarian,
    solve_load_balanced,
)
from safe_residual_rl.allocation.solvers.common import allocation_units, topological_segment_order
from safe_residual_rl.allocation.verifier import verify_plan

TARGET_ROOT = Path("/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL")
STAGING_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = TARGET_ROOT / "data" / "fixtures" / "allocation"


@pytest.fixture
def context():
    return load_oracle_context(STAGING_ROOT / "configs" / "allocation" / "oracle_proxy_v1.json")


@pytest.fixture
def multi_instance():
    raw = load_auditable_fixture(FIXTURES / "01_valid_minimal.json")
    base = allocation_instance_from_dict(raw["instance"])
    template = base.segments[0]
    segments = []
    for index, offset in enumerate((0.00, 0.15, 0.30, 0.45)):
        start = (offset, 0.0, 0.0)
        end = (offset + 0.1, 0.0, 0.0)
        segments.append(
            replace(
                template,
                id=f"seg-{index}",
                parent_curve_id=f"curve-{index}",
                segment_index=0,
                sampled_curve_m=(start, end),
                start_pose=replace(template.start_pose, position_m=start),
                end_pose=replace(template.end_pose, position_m=end),
                predecessor_ids=(),
                handoff_policy=HandoffPolicy.FREE,
                priority=4 - index,
            )
        )
    robot_0 = base.robots[0]
    robot_1 = replace(
        robot_0,
        id="robot-1",
        base_pose=replace(robot_0.base_pose, position_m=(0.55, 0.0, 0.0)),
    )
    return replace(base, instance_id="a1-multi", segments=tuple(segments), robots=(robot_0, robot_1))


def _codes(verification):
    return {item.code for item in verification.violations}


def test_topological_order_is_stable(multi_instance) -> None:
    assert topological_segment_order(multi_instance) == ("seg-0", "seg-1", "seg-2", "seg-3")


def test_same_robot_segments_form_one_assignment_unit(context) -> None:
    raw = load_auditable_fixture(FIXTURES / "02_valid_same_robot_segments.json")
    instance = allocation_instance_from_dict(raw["instance"])
    assert len(allocation_units(instance)) == 1
    assert len(allocation_units(instance)[0]) == 2


def test_verifier_rejects_same_robot_handoff_split(context) -> None:
    raw = load_auditable_fixture(FIXTURES / "02_valid_same_robot_segments.json")
    instance = allocation_instance_from_dict(raw["instance"])
    robot_1 = replace(instance.robots[0], id="robot-1")
    instance = replace(instance, robots=(instance.robots[0], robot_1))
    schedule = (
        ScheduledSegment(instance.segments[0].id, "robot-0", 0, 0.0, 5.0),
        ScheduledSegment(instance.segments[1].id, "robot-1", 0, 5.0, 10.0),
    )
    plan = AllocationPlan(PLAN_SCHEMA_VERSION, instance.instance_id, "bad", schedule, "feasible", (), ())
    assert "SAME_ROBOT_HANDOFF" in _codes(verify_plan(instance, plan, context))


def test_scheduler_and_verifier_accept_valid_plan(multi_instance, context) -> None:
    built = build_schedule(
        multi_instance,
        {"robot-0": ("seg-0", "seg-1"), "robot-1": ("seg-2", "seg-3")},
        context,
        "test",
    )
    assert built.status == "feasible"
    checked = verify_plan(multi_instance, built.plan, context)
    assert checked.feasible
    assert dict(checked.objective_terms)["makespan"] > 0


def test_scheduler_rejects_incomplete_assignment(multi_instance, context) -> None:
    built = build_schedule(multi_instance, {"robot-0": ("seg-0",)}, context, "test")
    assert built.status == "invalid_assignment"
    assert built.plan is None


def test_verifier_detects_robot_overlap(multi_instance, context) -> None:
    plan = AllocationPlan(
        PLAN_SCHEMA_VERSION,
        multi_instance.instance_id,
        "bad",
        tuple(ScheduledSegment(item.id, "robot-0", index, 0.0, item.process_duration_s) for index, item in enumerate(multi_instance.segments)),
        "feasible",
        (),
        (),
    )
    checked = verify_plan(multi_instance, plan, context)
    assert "ROBOT_OVERLAP" in _codes(checked)


def test_verifier_rejects_missing_transition_time(multi_instance, context) -> None:
    segment = multi_instance.segments[3]
    plan = AllocationPlan(
        PLAN_SCHEMA_VERSION,
        multi_instance.instance_id,
        "bad-transition",
        (
            ScheduledSegment("seg-0", "robot-0", 0, 0.0, 5.0),
            ScheduledSegment(segment.id, "robot-0", 1, 5.0, 10.0),
            ScheduledSegment("seg-1", "robot-1", 0, 4.0, 9.0),
            ScheduledSegment("seg-2", "robot-1", 1, 10.0, 15.0),
        ),
        "feasible",
        (),
        (),
    )
    assert "ROBOT_TRANSITION_TIME" in _codes(verify_plan(multi_instance, plan, context))


def test_verifier_detects_segment_time_window(multi_instance, context) -> None:
    built = build_schedule(multi_instance, {"robot-0": tuple(x.id for x in multi_instance.segments), "robot-1": ()}, context, "test")
    first = built.plan.schedule[0]
    bad = replace(built.plan, schedule=(replace(first, planned_start_s=-1.0),) + built.plan.schedule[1:])
    assert "SEGMENT_TIME_WINDOW" in _codes(verify_plan(multi_instance, bad, context))


def test_verifier_detects_edge_infeasibility(multi_instance, context) -> None:
    incapable = replace(multi_instance.robots[0], capabilities=())
    instance = replace(multi_instance, robots=(incapable, multi_instance.robots[1]))
    built = build_schedule(instance, {"robot-0": tuple(x.id for x in instance.segments), "robot-1": ()}, context, "test")
    assert "EDGE_INFEASIBLE" in _codes(verify_plan(instance, built.plan, context))


def test_capacity_one_shared_zone_serializes_different_robots(multi_instance, context) -> None:
    raw = load_auditable_fixture(FIXTURES / "04_valid_shared_zone.json")
    resource = allocation_instance_from_dict(raw["instance"]).resources[0]
    segments = tuple(replace(item, shared_resource_ids=(resource.id,)) for item in multi_instance.segments[:2])
    instance = replace(multi_instance, segments=segments, resources=(resource,))
    built = build_schedule(instance, {"robot-0": ("seg-0",), "robot-1": ("seg-1",)}, context, "test")
    ordered = sorted(built.plan.schedule, key=lambda item: item.planned_start_s)
    assert ordered[0].planned_end_s <= ordered[1].planned_start_s
    assert verify_plan(instance, built.plan, context).feasible


def test_verifier_detects_resource_capacity_conflict(multi_instance, context) -> None:
    raw = load_auditable_fixture(FIXTURES / "04_valid_shared_zone.json")
    resource = allocation_instance_from_dict(raw["instance"]).resources[0]
    segments = tuple(replace(item, shared_resource_ids=(resource.id,)) for item in multi_instance.segments[:2])
    instance = replace(multi_instance, segments=segments, resources=(resource,))
    schedule = (
        ScheduledSegment("seg-0", "robot-0", 0, 1.0, 11.0),
        ScheduledSegment("seg-1", "robot-1", 0, 1.0, 11.0),
    )
    plan = AllocationPlan(PLAN_SCHEMA_VERSION, instance.instance_id, "bad", schedule, "feasible", (), ())
    checked = verify_plan(instance, plan, context)
    assert "RESOURCE_CAPACITY" in _codes(checked)
    assert checked.proxy_conflict_count > 0


@pytest.mark.parametrize("solver", [solve_greedy, solve_load_balanced, solve_hungarian])
def test_deterministic_heuristics_return_verified_plans(solver, multi_instance, context) -> None:
    result = solver(multi_instance, context)
    assert result.status == "feasible"
    assert result.plan is not None
    assert verify_plan(multi_instance, result.plan, context).feasible
    second = solver(multi_instance, context)
    assert result.plan.schedule == second.plan.schedule


def test_assignment_milp_records_optimality_fields(multi_instance, context) -> None:
    protocol = load_solver_protocol(STAGING_ROOT / "configs" / "allocation" / "solver_protocol_v1.json")
    result = solve_assignment_milp(multi_instance, context, protocol)
    assert result.status == "optimal"
    assert result.plan is not None
    assert result.objective_value is not None
    assert result.best_bound is not None
    assert result.mip_gap == pytest.approx(0.0)
    assert verify_plan(multi_instance, result.plan, context).feasible


def test_assignment_milp_preserves_time_limit_status(monkeypatch, multi_instance, context) -> None:
    module = importlib.import_module("safe_residual_rl.allocation.solvers.milp")
    monkeypatch.setattr(
        module,
        "milp",
        lambda **_: SimpleNamespace(
            status=1,
            message="Time limit reached",
            x=None,
            fun=None,
            mip_dual_bound=3.0,
            mip_gap=None,
        ),
    )
    protocol = load_solver_protocol(STAGING_ROOT / "configs" / "allocation" / "solver_protocol_v1.json")
    result = module.solve_assignment_milp(multi_instance, context, protocol)
    assert result.status == "limit"
    assert result.plan is None
    assert result.best_bound == pytest.approx(3.0)
    assert any(item.startswith("TIME_LIMIT_S=") for item in result.diagnostics)


def test_all_methods_report_infeasible_edge_mask(multi_instance, context) -> None:
    robots = tuple(replace(item, capabilities=()) for item in multi_instance.robots)
    instance = replace(multi_instance, robots=robots)
    protocol = load_solver_protocol(STAGING_ROOT / "configs" / "allocation" / "solver_protocol_v1.json")
    results = [
        solve_greedy(instance, context),
        solve_load_balanced(instance, context),
        solve_hungarian(instance, context),
        solve_assignment_milp(instance, context, protocol),
    ]
    assert {item.status for item in results} == {"infeasible"}
    assert all(item.plan is None for item in results)


def test_solver_protocol_rejects_invalid_gap(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"version":"a1-solver-protocol-v1","time_limit_s":1,"relative_gap":2,"deterministic_seed":0}')
    with pytest.raises(ValueError, match="invalid solver limits"):
        load_solver_protocol(path)
