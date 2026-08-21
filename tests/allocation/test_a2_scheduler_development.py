from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation import (
    build_deadline_aware_schedule,
    build_schedule,
    load_oracle_context,
    solve_hybrid_load_balanced,
    solve_order_aware_lns,
    verify_plan,
)
from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.schema import HandoffPolicy, TimeWindow, allocation_instance_from_dict

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data" / "fixtures" / "allocation"


@pytest.fixture
def context():
    return load_oracle_context(ROOT / "configs" / "allocation" / "oracle_proxy_v1.json")


@pytest.fixture
def deadline_instance():
    raw = load_auditable_fixture(FIXTURES / "01_valid_minimal.json")
    base = allocation_instance_from_dict(raw["instance"])
    robot = base.robots[0]
    pose = replace(base.segments[0].start_pose, position_m=robot.base_pose.position_m)
    loose = replace(
        base.segments[0],
        id="seg-loose",
        parent_curve_id="curve-loose",
        segment_index=0,
        sampled_curve_m=(robot.base_pose.position_m, robot.base_pose.position_m),
        start_pose=pose,
        end_pose=pose,
        length_m=0.0,
        process_duration_s=5.0,
        time_window=TimeWindow(0.0, 100.0),
        predecessor_ids=(),
        handoff_policy=HandoffPolicy.FREE,
    )
    tight = replace(
        loose,
        id="seg-tight",
        parent_curve_id="curve-tight",
        time_window=TimeWindow(0.0, 6.0),
        priority=loose.priority + 1,
    )
    incapable = replace(robot, id="robot-incapable", capabilities=(), tool_ids=())
    return replace(
        base,
        instance_id="deadline-order-counterexample",
        segments=(loose, tight),
        robots=(robot, incapable),
        resources=(),
    )


def test_deadline_dispatch_repairs_fixed_lexical_order(deadline_instance, context) -> None:
    assignment = {"seg-loose": "robot-0", "seg-tight": "robot-0"}
    fixed = build_schedule(
        deadline_instance,
        {"robot-0": ("seg-loose", "seg-tight"), "robot-incapable": ()},
        context,
        "fixed",
    )
    adaptive = build_deadline_aware_schedule(
        deadline_instance, assignment, context, "deadline-aware"
    )
    assert fixed.status == "infeasible"
    assert adaptive.status == "feasible"
    assert [item.segment_id for item in adaptive.plan.schedule] == ["seg-tight", "seg-loose"]
    assert verify_plan(deadline_instance, adaptive.plan, context).feasible


def test_deadline_dispatch_is_deterministic(deadline_instance, context) -> None:
    assignment = {"seg-loose": "robot-0", "seg-tight": "robot-0"}
    first = build_deadline_aware_schedule(deadline_instance, assignment, context, "test")
    second = build_deadline_aware_schedule(deadline_instance, assignment, context, "test")
    assert first.plan.schedule == second.plan.schedule
    assert first.diagnostics == second.diagnostics


def test_order_aware_lns_does_not_require_feasible_initial_schedule(
    deadline_instance, context
) -> None:
    result = solve_order_aware_lns(deadline_instance, context, iterations=10, seed=3)
    assert result.status == "feasible"
    assert result.plan is not None
    assert verify_plan(deadline_instance, result.plan, context).feasible
    assert "NO_FEASIBLE_INITIAL_SCHEDULE_REQUIRED" in result.diagnostics


def test_hybrid_solver_uses_deadline_fallback(deadline_instance, context) -> None:
    result = solve_hybrid_load_balanced(deadline_instance, context)
    assert result.status == "feasible"
    assert result.plan is not None
    assert verify_plan(deadline_instance, result.plan, context).feasible
    assert any("BEST_OF_FIXED_AND_DEADLINE_V2" in item for item in result.diagnostics)


def test_deadline_dispatch_rejects_incomplete_assignment(deadline_instance, context) -> None:
    result = build_deadline_aware_schedule(
        deadline_instance, {"seg-loose": "robot-0"}, context, "bad"
    )
    assert result.status == "invalid_assignment"
    assert result.plan is None
