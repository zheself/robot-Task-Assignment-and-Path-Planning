from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation.features import extract_curve_features
from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.masks import build_edge_mask
from safe_residual_rl.allocation.oracle import (
    EdgeReason,
    OracleContext,
    SphereProxy,
    estimate_edge,
    load_oracle_context,
    oracle_context_from_dict,
)
from safe_residual_rl.allocation.schema import (
    ProcessDirection,
    allocation_instance_from_dict,
)

TARGET_ROOT = Path("/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL")
FIXTURES = TARGET_ROOT / "data" / "fixtures" / "allocation"
STAGING_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def instance():
    payload = load_auditable_fixture(FIXTURES / "01_valid_minimal.json")
    return allocation_instance_from_dict(payload["instance"])


@pytest.fixture
def context() -> OracleContext:
    return load_oracle_context(STAGING_ROOT / "configs" / "allocation" / "oracle_proxy_v1.json")


def test_straight_curve_features_are_analytical(instance) -> None:
    features = extract_curve_features(instance.segments[0])
    assert features.polyline_length_m == pytest.approx(0.1)
    assert features.chord_length_m == pytest.approx(0.1)
    assert features.tortuosity == pytest.approx(1.0)
    assert features.direction_unit == pytest.approx((1.0, 0.0, 0.0))
    assert features.max_turn_angle_rad == 0.0


def test_compatible_edge_is_feasible_and_deterministic(instance, context) -> None:
    first = estimate_edge(instance.robots[0], instance.segments[0], context)
    second = estimate_edge(instance.robots[0], instance.segments[0], context)
    assert first == second
    assert first.feasible
    assert first.reason_codes == ()
    assert first.path_length_m == pytest.approx(0.1)
    assert first.confidence == pytest.approx(0.25)
    assert "SIMPLIFIED_ANALYTICAL_PROXY" in first.diagnostics


def test_missing_capability_has_stable_reason(instance, context) -> None:
    robot = replace(instance.robots[0], capabilities=())
    edge = estimate_edge(robot, instance.segments[0], context)
    assert not edge.feasible
    assert EdgeReason.MISSING_CAPABILITY.value in edge.reason_codes


def test_missing_tool_has_stable_reason(instance, context) -> None:
    robot = replace(instance.robots[0], tool_ids=("other-tool",))
    edge = estimate_edge(robot, instance.segments[0], context)
    assert not edge.feasible
    assert EdgeReason.MISSING_TOOL.value in edge.reason_codes


def test_outside_reach_has_stable_reason(instance, context) -> None:
    short_reach = replace(context, default_max_reach_m=0.1, reach_by_kinematic_model_m=())
    edge = estimate_edge(instance.robots[0], instance.segments[0], short_reach)
    assert not edge.feasible
    assert EdgeReason.OUTSIDE_REACH_PROXY.value in edge.reason_codes
    assert edge.kinematic_risk > 1.0


def test_no_common_time_window_is_rejected(instance, context) -> None:
    robot = replace(instance.robots[0], availability=replace(instance.robots[0].availability, end_s=1.0))
    edge = estimate_edge(robot, instance.segments[0], context)
    assert EdgeReason.NO_COMMON_TIME_WINDOW.value in edge.reason_codes


def test_spherical_no_go_is_only_a_proxy_but_masks_edge(instance, context) -> None:
    sphere = SphereProxy(id="proxy-0", center_m=(0.05, 0.0, 0.0), radius_m=0.01)
    blocked_context = replace(context, no_go_spheres=(sphere,))
    edge = estimate_edge(instance.robots[0], instance.segments[0], blocked_context)
    assert EdgeReason.INTERSECTS_NO_GO_PROXY.value in edge.reason_codes
    assert "SIMPLIFIED_ANALYTICAL_PROXY" in edge.diagnostics


def test_either_direction_uses_nearest_endpoint(instance, context) -> None:
    segment = replace(instance.segments[0], process_direction=ProcessDirection.EITHER)
    robot = replace(instance.robots[0], base_pose=replace(instance.robots[0].base_pose, position_m=(0.2, 0.0, 0.0)))
    edge = estimate_edge(robot, segment, context)
    assert edge.travel_time_s == pytest.approx(1.0)


def test_mask_shape_order_and_reasons_are_stable(instance, context) -> None:
    bad_robot = replace(instance.robots[0], id="robot-bad", capabilities=())
    expanded = replace(instance, robots=(instance.robots[0], bad_robot))
    mask = build_edge_mask(expanded, context)
    assert mask.segment_ids == ("seg-0",)
    assert mask.robot_ids == ("robot-0", "robot-bad")
    assert mask.allowed == ((True, False),)
    assert mask.reason_codes[0][1] == (EdgeReason.MISSING_CAPABILITY.value,)
    assert mask.is_allowed("seg-0", "robot-0")


def test_oracle_config_rejects_unbounded_confidence() -> None:
    raw = json.loads((STAGING_ROOT / "configs" / "allocation" / "oracle_proxy_v1.json").read_text())
    raw["confidence"] = 1.5
    with pytest.raises(ValueError, match="out of range"):
        oracle_context_from_dict(raw)
