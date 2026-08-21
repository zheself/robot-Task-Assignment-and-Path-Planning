"""Reason-coded analytical edge oracle for A1 W3.

The oracle is deliberately a conservative geometry/semantics proxy. It is not
IK, motion planning, full collision checking, or physical process modelling.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .features import extract_curve_features, point_distance, point_to_segment_distance
from .schema import ProcessDirection, ProcessSegment, RobotSpec

ORACLE_VERSION = "analytical-edge-oracle-v1"


class EdgeReason(str, Enum):
    MISSING_CAPABILITY = "missing_capability"
    MISSING_TOOL = "missing_tool"
    OUTSIDE_REACH_PROXY = "outside_reach_proxy"
    INTERSECTS_NO_GO_PROXY = "intersects_no_go_proxy"
    NO_COMMON_TIME_WINDOW = "no_common_time_window"


@dataclass(frozen=True)
class SphereProxy:
    id: str
    center_m: tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True)
class OracleContext:
    version: str
    default_max_reach_m: float
    reach_by_kinematic_model_m: tuple[tuple[str, float], ...]
    no_go_spheres: tuple[SphereProxy, ...]
    confidence: float

    def max_reach_m(self, robot: RobotSpec) -> tuple[float, bool]:
        mapping = dict(self.reach_by_kinematic_model_m)
        if robot.kinematic_model_id in mapping:
            return mapping[robot.kinematic_model_id], False
        return self.default_max_reach_m, True


@dataclass(frozen=True)
class EdgeEstimate:
    robot_id: str
    segment_id: str
    feasible: bool
    reason_codes: tuple[str, ...]
    travel_time_s: float
    process_time_s: float
    path_length_m: float
    kinematic_risk: float
    conflict_proxy: float
    confidence: float
    diagnostics: tuple[str, ...]


def estimate_edge(
    robot: RobotSpec, segment: ProcessSegment, context: OracleContext
) -> EdgeEstimate:
    features = extract_curve_features(segment)
    max_reach_m, used_default = context.max_reach_m(robot)
    base = robot.base_pose.position_m
    maximum_distance = max(point_distance(base, point) for point in segment.sampled_curve_m)
    reach_ratio = maximum_distance / max_reach_m
    reasons: list[EdgeReason] = []
    diagnostics = ["SIMPLIFIED_ANALYTICAL_PROXY", f"oracle_version={context.version}"]
    if used_default:
        diagnostics.append("DEFAULT_REACH_USED")
    missing_capabilities = sorted(set(segment.required_capabilities) - set(robot.capabilities))
    if missing_capabilities:
        reasons.append(EdgeReason.MISSING_CAPABILITY)
        diagnostics.append(f"missing_capabilities={','.join(missing_capabilities)}")
    if segment.required_tool_id is not None and segment.required_tool_id not in robot.tool_ids:
        reasons.append(EdgeReason.MISSING_TOOL)
        diagnostics.append(f"missing_tool={segment.required_tool_id}")
    if maximum_distance > max_reach_m + 1e-12:
        reasons.append(EdgeReason.OUTSIDE_REACH_PROXY)
        diagnostics.append(f"max_distance_m={maximum_distance:.9g}>reach_m={max_reach_m:.9g}")
    for proxy in context.no_go_spheres:
        if _curve_intersects_sphere(segment, proxy):
            reasons.append(EdgeReason.INTERSECTS_NO_GO_PROXY)
            diagnostics.append(f"no_go_proxy={proxy.id}")
    earliest_start = max(segment.time_window.start_s, robot.availability.start_s)
    latest_end = min(segment.time_window.end_s, robot.availability.end_s)
    if earliest_start + segment.process_duration_s > latest_end + 1e-12:
        reasons.append(EdgeReason.NO_COMMON_TIME_WINDOW)
        diagnostics.append("insufficient_common_time_window")

    entry_points = [segment.start_pose.position_m]
    if segment.process_direction in {ProcessDirection.REVERSE, ProcessDirection.EITHER}:
        entry_points = [segment.end_pose.position_m]
    if segment.process_direction is ProcessDirection.EITHER:
        entry_points.append(segment.start_pose.position_m)
    approach_distance = min(point_distance(base, point) for point in entry_points)
    unique_reasons = tuple(sorted({reason.value for reason in reasons}))
    return EdgeEstimate(
        robot_id=robot.id,
        segment_id=segment.id,
        feasible=not unique_reasons,
        reason_codes=unique_reasons,
        travel_time_s=approach_distance / robot.nominal_cartesian_speed_m_s,
        process_time_s=segment.process_duration_s,
        path_length_m=features.polyline_length_m,
        kinematic_risk=reach_ratio,
        conflict_proxy=float(len(segment.shared_resource_ids)),
        confidence=context.confidence,
        diagnostics=tuple(diagnostics),
    )


def oracle_context_from_dict(data: Mapping[str, Any]) -> OracleContext:
    if data.get("version") != ORACLE_VERSION:
        raise ValueError(f"expected oracle version {ORACLE_VERSION}")
    default_reach = float(data["default_max_reach_m"])
    confidence = float(data["confidence"])
    reach_items = tuple(
        sorted((str(key), float(value)) for key, value in data["reach_by_kinematic_model_m"].items())
    )
    spheres = tuple(
        SphereProxy(
            id=str(item["id"]),
            center_m=_finite_point(item["center_m"]),
            radius_m=float(item["radius_m"]),
        )
        for item in data["no_go_spheres"]
    )
    all_positive = default_reach > 0 and all(value > 0 for _, value in reach_items)
    spheres_valid = all(item.id and item.radius_m > 0 for item in spheres)
    if not all_positive or not spheres_valid or not 0.0 <= confidence <= 1.0:
        raise ValueError("oracle reach, sphere radii and confidence are out of range")
    return OracleContext(
        version=ORACLE_VERSION,
        default_max_reach_m=default_reach,
        reach_by_kinematic_model_m=reach_items,
        no_go_spheres=spheres,
        confidence=confidence,
    )


def load_oracle_context(path: str | Path) -> OracleContext:
    with Path(path).open("r", encoding="utf-8") as handle:
        return oracle_context_from_dict(json.load(handle))


def _curve_intersects_sphere(segment: ProcessSegment, sphere: SphereProxy) -> bool:
    return any(
        point_to_segment_distance(sphere.center_m, start, end) <= sphere.radius_m
        for start, end in zip(segment.sampled_curve_m, segment.sampled_curve_m[1:])
    )


def _finite_point(values: Any) -> tuple[float, float, float]:
    point = tuple(float(value) for value in values)
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError("sphere center must contain three finite SI coordinates")
    return point  # type: ignore[return-value]
