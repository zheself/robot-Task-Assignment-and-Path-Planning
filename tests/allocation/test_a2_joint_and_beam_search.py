from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from safe_residual_rl.allocation import (
    load_oracle_context,
    search_fixed_assignment_schedule,
    solve_beam_alns,
    solve_joint_assignment_sequence_reference,
    verify_plan,
)
from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.schema import HandoffPolicy, TimeWindow, allocation_instance_from_dict

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context():
    return load_oracle_context(ROOT / "configs" / "allocation" / "oracle_proxy_v1.json")


@pytest.fixture
def small_search_instance():
    raw = load_auditable_fixture(
        ROOT / "data" / "fixtures" / "allocation" / "01_valid_minimal.json"
    )
    base = allocation_instance_from_dict(raw["instance"])
    robot = base.robots[0]
    second_robot = replace(
        robot,
        id="robot-1",
        base_pose=replace(robot.base_pose, position_m=(0.25, 0.0, 0.0)),
    )
    template = base.segments[0]
    segments = []
    for index, (position, due) in enumerate(((0.0, 30.0), (0.12, 12.0), (0.24, 30.0))):
        point = (position, 0.0, 0.0)
        pose = replace(template.start_pose, position_m=point)
        segments.append(
            replace(
                template,
                id=f"seg-{index}",
                parent_curve_id=f"curve-{index}",
                segment_index=0,
                sampled_curve_m=(point, point),
                start_pose=pose,
                end_pose=pose,
                length_m=0.0,
                process_duration_s=4.0,
                time_window=TimeWindow(0.0, due),
                predecessor_ids=(),
                handoff_policy=HandoffPolicy.FREE,
            )
        )
    return replace(
        base,
        instance_id="small-joint-search",
        segments=tuple(segments),
        robots=(robot, second_robot),
        resources=(),
    )


def test_beam_search_returns_verified_deterministic_plan(small_search_instance, context) -> None:
    assignment = {item.id: "robot-0" for item in small_search_instance.segments}
    first = search_fixed_assignment_schedule(
        small_search_instance,
        assignment,
        context,
        "beam-test",
        mode="beam",
        objective_weights={"makespan": 1.0, "travel_setup_time": 0.1},
        beam_width=4,
        node_limit=100,
    )
    second = search_fixed_assignment_schedule(
        small_search_instance,
        assignment,
        context,
        "beam-test",
        mode="beam",
        objective_weights={"makespan": 1.0, "travel_setup_time": 0.1},
        beam_width=4,
        node_limit=100,
    )
    assert first.status == "feasible_limit"
    assert first.plan is not None
    assert first.plan.schedule == second.plan.schedule
    assert verify_plan(small_search_instance, first.plan, context).feasible
    assert any(item == "BRANCHES_ROBOT_AND_SHARED_RESOURCE_ORDER" for item in first.diagnostics)


def test_joint_reference_completes_and_reports_proxy_optimum(
    small_search_instance, context
) -> None:
    result = solve_joint_assignment_sequence_reference(
        small_search_instance,
        context,
        max_segments=4,
        max_assignment_combinations=100,
        node_limit=20_000,
        time_limit_s=5.0,
    )
    assert result.status == "optimal"
    assert result.plan is not None
    assert result.objective_value == pytest.approx(result.best_bound)
    assert verify_plan(small_search_instance, result.plan, context).feasible
    assert "SEARCH_COMPLETE=True" in result.diagnostics
    assert "OPTIMAL_ONLY_WITHIN_A1_PROXY_IF_SEARCH_COMPLETE" in result.diagnostics


def test_joint_reference_distinguishes_limit_from_infeasible(
    small_search_instance, context
) -> None:
    result = solve_joint_assignment_sequence_reference(
        small_search_instance,
        context,
        max_segments=4,
        max_assignment_combinations=100,
        node_limit=1,
        time_limit_s=5.0,
    )
    assert result.status == "limit"
    assert result.plan is None
    assert "SEARCH_COMPLETE=False" in result.diagnostics


def test_joint_reference_rejects_unsupported_scale(small_search_instance, context) -> None:
    result = solve_joint_assignment_sequence_reference(
        small_search_instance, context, max_segments=2
    )
    assert result.status == "unsupported_scale"
    assert result.plan is None


def test_beam_alns_returns_verified_plan(small_search_instance, context) -> None:
    first = solve_beam_alns(
        small_search_instance,
        context,
        iterations=5,
        seed=7,
        beam_width=4,
        beam_node_limit=100,
    )
    second = solve_beam_alns(
        small_search_instance,
        context,
        iterations=5,
        seed=7,
        beam_width=4,
        beam_node_limit=100,
    )
    assert first.status == "feasible"
    assert first.plan is not None
    assert first.plan.schedule == second.plan.schedule
    assert verify_plan(small_search_instance, first.plan, context).feasible
    assert "BRANCHES_ROBOT_AND_SHARED_RESOURCE_ORDER" in first.diagnostics


def test_beam_search_branches_shared_resource_order(small_search_instance, context) -> None:
    raw = load_auditable_fixture(
        ROOT / "data" / "fixtures" / "allocation" / "04_valid_shared_zone.json"
    )
    resource = allocation_instance_from_dict(raw["instance"]).resources[0]
    segments = tuple(
        replace(item, shared_resource_ids=(resource.id,))
        for item in small_search_instance.segments[:2]
    )
    instance = replace(
        small_search_instance,
        instance_id="shared-resource-branch",
        segments=segments,
        resources=(resource,),
    )
    assignment = {segments[0].id: "robot-0", segments[1].id: "robot-1"}
    result = search_fixed_assignment_schedule(
        instance,
        assignment,
        context,
        "resource-beam",
        mode="beam",
        objective_weights={"makespan": 1.0},
        beam_width=4,
        node_limit=100,
    )
    assert result.plan is not None
    assert result.nodes_expanded >= 3
    assert verify_plan(instance, result.plan, context).feasible


def test_joint_development_configs_forbid_frozen_access() -> None:
    development = json.loads(
        (ROOT / "configs/allocation/a2_joint_search_development_v1.json").read_text()
    )
    reference = json.loads(
        (ROOT / "configs/allocation/a2_joint_reference_protocol_v1.json").read_text()
    )
    assert development["allowed_splits"] == ["train", "validation"]
    assert set(development["forbidden_splits"]) == {"frozen_test", "stress"}
    assert reference["allowed_benchmark_split"] == "train"
    assert set(reference["forbidden_splits"]) == {
        "validation",
        "frozen_test",
        "stress",
    }
