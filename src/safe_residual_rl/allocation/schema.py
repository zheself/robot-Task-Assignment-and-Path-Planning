"""Versioned A0 schema for continuous-process allocation instances.

This module validates semantics only. Reachability, scheduling, assignment and
collision-proxy evaluation belong to A1 and later gates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "allocation-instance-v1"
PLAN_SCHEMA_VERSION = "allocation-plan-v1"
_POINT_TOLERANCE_M = 1e-6
_QUATERNION_TOLERANCE = 1e-4


class EvidenceLabel(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    SIM_GEOMETRIC = "SIM_GEOMETRIC"


class ProcessDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"
    EITHER = "either"


class HandoffPolicy(str, Enum):
    FREE = "free"
    SAME_ROBOT = "same_robot"
    EXPLICIT_BOUNDARY = "explicit_boundary"
    NOT_SPLITTABLE = "not_splittable"


class ResourceType(str, Enum):
    SHARED_ZONE = "shared_zone"
    FIXTURE = "fixture"
    NO_GO = "no_go"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


class SchemaValidationError(ValueError):
    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        detail = "; ".join(f"{i.code}@{i.path}: {i.message}" for i in issues)
        super().__init__(detail)


@dataclass(frozen=True)
class Pose:
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class ProcessSegment:
    id: str
    parent_curve_id: str
    segment_index: int
    sampled_curve_m: tuple[tuple[float, float, float], ...]
    start_pose: Pose
    end_pose: Pose
    process_direction: ProcessDirection
    length_m: float
    process_duration_s: float
    required_capabilities: tuple[str, ...]
    required_tool_id: str | None
    priority: int
    time_window: TimeWindow
    predecessor_ids: tuple[str, ...]
    handoff_policy: HandoffPolicy
    shared_resource_ids: tuple[str, ...]


@dataclass(frozen=True)
class RobotSpec:
    id: str
    base_pose: Pose
    capabilities: tuple[str, ...]
    tool_ids: tuple[str, ...]
    availability: TimeWindow
    kinematic_model_id: str
    initial_joint_state_rad: tuple[float, ...]
    nominal_cartesian_speed_m_s: float


@dataclass(frozen=True)
class ResourceSpec:
    id: str
    resource_type: ResourceType
    capacity: int
    availability: TimeWindow


@dataclass(frozen=True)
class AllocationInstance:
    schema_version: str
    evidence_label: EvidenceLabel
    instance_id: str
    workpiece_id: str
    layout_id: str
    coordinate_frame: str
    segments: tuple[ProcessSegment, ...]
    robots: tuple[RobotSpec, ...]
    resources: tuple[ResourceSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_label": self.evidence_label.value,
            "instance_id": self.instance_id,
            "workpiece_id": self.workpiece_id,
            "layout_id": self.layout_id,
            "coordinate_frame": self.coordinate_frame,
            "segments": [_segment_to_dict(x) for x in self.segments],
            "robots": [_robot_to_dict(x) for x in self.robots],
            "resources": [_resource_to_dict(x) for x in self.resources],
        }


@dataclass(frozen=True)
class ScheduledSegment:
    segment_id: str
    robot_id: str
    order_index: int
    planned_start_s: float
    planned_end_s: float


@dataclass(frozen=True)
class AllocationPlan:
    schema_version: str
    instance_id: str
    method_id: str
    schedule: tuple[ScheduledSegment, ...]
    solver_status: str
    objective_terms: tuple[tuple[str, float], ...]
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "method_id": self.method_id,
            "schedule": [
                {
                    "segment_id": x.segment_id,
                    "robot_id": x.robot_id,
                    "order_index": x.order_index,
                    "planned_start_s": x.planned_start_s,
                    "planned_end_s": x.planned_end_s,
                }
                for x in self.schedule
            ],
            "solver_status": self.solver_status,
            "objective_terms": dict(self.objective_terms),
            "diagnostics": list(self.diagnostics),
        }


def allocation_plan_from_dict(
    data: Mapping[str, Any], instance: AllocationInstance | None = None
) -> AllocationPlan:
    """Parse the stable plan exchange schema; A1 adds constraint verification."""
    try:
        plan = AllocationPlan(
            schema_version=str(data["schema_version"]),
            instance_id=str(data["instance_id"]),
            method_id=str(data["method_id"]),
            schedule=tuple(
                ScheduledSegment(
                    segment_id=str(x["segment_id"]),
                    robot_id=str(x["robot_id"]),
                    order_index=int(x["order_index"]),
                    planned_start_s=float(x["planned_start_s"]),
                    planned_end_s=float(x["planned_end_s"]),
                )
                for x in data["schedule"]
            ),
            solver_status=str(data["solver_status"]),
            objective_terms=tuple(
                sorted((str(key), float(value)) for key, value in data["objective_terms"].items())
            ),
            diagnostics=tuple(str(x) for x in data["diagnostics"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError([ValidationIssue("MALFORMED_PLAN", "$", str(exc))]) from exc
    issues: list[ValidationIssue] = []
    if plan.schema_version != PLAN_SCHEMA_VERSION:
        issues.append(ValidationIssue("PLAN_SCHEMA_VERSION", "schema_version", PLAN_SCHEMA_VERSION))
    if not plan.instance_id or not plan.method_id or not plan.solver_status:
        issues.append(ValidationIssue("EMPTY_PLAN_FIELD", "$", "IDs and solver_status are required"))
    segment_ids = [x.segment_id for x in plan.schedule]
    slots = [(x.robot_id, x.order_index) for x in plan.schedule]
    if len(segment_ids) != len(set(segment_ids)):
        issues.append(ValidationIssue("DUPLICATE_PLAN_SEGMENT", "schedule", "segments must be unique"))
    if len(slots) != len(set(slots)):
        issues.append(ValidationIssue("DUPLICATE_ROBOT_ORDER", "schedule", "robot order slots must be unique"))
    for index, item in enumerate(plan.schedule):
        if (
            not item.segment_id
            or not item.robot_id
            or item.order_index < 0
            or not math.isfinite(item.planned_start_s)
            or not math.isfinite(item.planned_end_s)
            or item.planned_end_s < item.planned_start_s
        ):
            issues.append(ValidationIssue("INVALID_SCHEDULE_ITEM", f"schedule[{index}]", "invalid IDs/order/time"))
    if instance is not None:
        if plan.instance_id != instance.instance_id:
            issues.append(ValidationIssue("PLAN_INSTANCE_MISMATCH", "instance_id", instance.instance_id))
        expected_segments = {x.id for x in instance.segments}
        if set(segment_ids) != expected_segments:
            issues.append(ValidationIssue("PLAN_SEGMENT_COVERAGE", "schedule", "must cover every segment once"))
        robot_ids = {x.id for x in instance.robots}
        if any(x.robot_id not in robot_ids for x in plan.schedule):
            issues.append(ValidationIssue("UNKNOWN_PLAN_ROBOT", "schedule", "robot reference is unknown"))
    if issues:
        raise SchemaValidationError(issues)
    return plan


def _finite_tuple(value: Iterable[Any], length: int, path: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in value)
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{path} must contain {length} finite numbers")
    return result


def _pose(data: Mapping[str, Any], path: str) -> Pose:
    return Pose(
        position_m=_finite_tuple(data["position_m"], 3, f"{path}.position_m"),  # type: ignore[arg-type]
        quaternion_xyzw=_finite_tuple(
            data["quaternion_xyzw"], 4, f"{path}.quaternion_xyzw"  # type: ignore[arg-type]
        ),
    )


def _window(data: Mapping[str, Any]) -> TimeWindow:
    return TimeWindow(float(data["start_s"]), float(data["end_s"]))


def allocation_instance_from_dict(data: Mapping[str, Any]) -> AllocationInstance:
    """Parse a dictionary and raise a stable validation error on any problem."""
    try:
        segments = tuple(
            ProcessSegment(
                id=str(s["id"]),
                parent_curve_id=str(s["parent_curve_id"]),
                segment_index=int(s["segment_index"]),
                sampled_curve_m=tuple(
                    _finite_tuple(p, 3, f"segments[{i}].sampled_curve_m")  # type: ignore[arg-type]
                    for p in s["sampled_curve_m"]
                ),
                start_pose=_pose(s["start_pose"], f"segments[{i}].start_pose"),
                end_pose=_pose(s["end_pose"], f"segments[{i}].end_pose"),
                process_direction=ProcessDirection(s["process_direction"]),
                length_m=float(s["length_m"]),
                process_duration_s=float(s["process_duration_s"]),
                required_capabilities=tuple(str(x) for x in s["required_capabilities"]),
                required_tool_id=(
                    None if s.get("required_tool_id") is None else str(s["required_tool_id"])
                ),
                priority=int(s["priority"]),
                time_window=_window(s["time_window"]),
                predecessor_ids=tuple(str(x) for x in s["predecessor_ids"]),
                handoff_policy=HandoffPolicy(s["handoff_policy"]),
                shared_resource_ids=tuple(str(x) for x in s["shared_resource_ids"]),
            )
            for i, s in enumerate(data["segments"])
        )
        robots = tuple(
            RobotSpec(
                id=str(r["id"]),
                base_pose=_pose(r["base_pose"], f"robots[{i}].base_pose"),
                capabilities=tuple(str(x) for x in r["capabilities"]),
                tool_ids=tuple(str(x) for x in r["tool_ids"]),
                availability=_window(r["availability"]),
                kinematic_model_id=str(r["kinematic_model_id"]),
                initial_joint_state_rad=tuple(float(x) for x in r["initial_joint_state_rad"]),
                nominal_cartesian_speed_m_s=float(r["nominal_cartesian_speed_m_s"]),
            )
            for i, r in enumerate(data["robots"])
        )
        resources = tuple(
            ResourceSpec(
                id=str(z["id"]),
                resource_type=ResourceType(z["resource_type"]),
                capacity=int(z["capacity"]),
                availability=_window(z["availability"]),
            )
            for z in data["resources"]
        )
        instance = AllocationInstance(
            schema_version=str(data["schema_version"]),
            evidence_label=EvidenceLabel(data["evidence_label"]),
            instance_id=str(data["instance_id"]),
            workpiece_id=str(data["workpiece_id"]),
            layout_id=str(data["layout_id"]),
            coordinate_frame=str(data["coordinate_frame"]),
            segments=segments,
            robots=robots,
            resources=resources,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            [ValidationIssue("MALFORMED_FIELD", "$", str(exc))]
        ) from exc
    issues = validate_instance(instance)
    if issues:
        raise SchemaValidationError(issues)
    return instance


def validate_instance(instance: AllocationInstance) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    def add(code: str, path: str, message: str) -> None:
        issues.append(ValidationIssue(code, path, message))

    if instance.schema_version != SCHEMA_VERSION:
        add("SCHEMA_VERSION", "schema_version", f"expected {SCHEMA_VERSION}")
    for name in ("instance_id", "workpiece_id", "layout_id", "coordinate_frame"):
        if not getattr(instance, name).strip():
            add("EMPTY_ID", name, "must be non-empty")
    if not instance.segments:
        add("EMPTY_SEGMENTS", "segments", "at least one segment is required")
    if not instance.robots:
        add("EMPTY_ROBOTS", "robots", "at least one robot is required")

    _check_unique((x.id for x in instance.segments), "segments", "DUPLICATE_SEGMENT_ID", add)
    _check_unique((x.id for x in instance.robots), "robots", "DUPLICATE_ROBOT_ID", add)
    _check_unique((x.id for x in instance.resources), "resources", "DUPLICATE_RESOURCE_ID", add)

    segment_ids = {x.id for x in instance.segments}
    resource_by_id = {x.id: x for x in instance.resources}
    parent_groups: dict[str, list[ProcessSegment]] = {}
    for i, segment in enumerate(instance.segments):
        path = f"segments[{i}]"
        parent_groups.setdefault(segment.parent_curve_id, []).append(segment)
        if segment.segment_index < 0:
            add("NEGATIVE_SEGMENT_INDEX", f"{path}.segment_index", "must be >= 0")
        if len(segment.sampled_curve_m) < 2:
            add("CURVE_TOO_SHORT", f"{path}.sampled_curve_m", "needs at least two points")
        else:
            if _distance(segment.start_pose.position_m, segment.sampled_curve_m[0]) > _POINT_TOLERANCE_M:
                add("START_POSE_MISMATCH", f"{path}.start_pose", "must match first curve point")
            if _distance(segment.end_pose.position_m, segment.sampled_curve_m[-1]) > _POINT_TOLERANCE_M:
                add("END_POSE_MISMATCH", f"{path}.end_pose", "must match last curve point")
        _check_pose(segment.start_pose, f"{path}.start_pose", add)
        _check_pose(segment.end_pose, f"{path}.end_pose", add)
        if not math.isfinite(segment.length_m) or segment.length_m <= 0:
            add("INVALID_LENGTH", f"{path}.length_m", "must be finite and > 0")
        if not math.isfinite(segment.process_duration_s) or segment.process_duration_s <= 0:
            add("INVALID_DURATION", f"{path}.process_duration_s", "must be finite and > 0")
        if segment.priority < 0:
            add("INVALID_PRIORITY", f"{path}.priority", "must be >= 0")
        _check_window(segment.time_window, f"{path}.time_window", add)
        if len(set(segment.predecessor_ids)) != len(segment.predecessor_ids):
            add("DUPLICATE_PREDECESSOR", f"{path}.predecessor_ids", "must be unique")
        for predecessor_id in segment.predecessor_ids:
            if predecessor_id == segment.id:
                add("SELF_PREDECESSOR", f"{path}.predecessor_ids", "cannot reference itself")
            elif predecessor_id not in segment_ids:
                add("UNKNOWN_PREDECESSOR", f"{path}.predecessor_ids", predecessor_id)
        for resource_id in segment.shared_resource_ids:
            if resource_id not in resource_by_id:
                add("UNKNOWN_RESOURCE", f"{path}.shared_resource_ids", resource_id)

    for i, robot in enumerate(instance.robots):
        path = f"robots[{i}]"
        _check_pose(robot.base_pose, f"{path}.base_pose", add)
        _check_window(robot.availability, f"{path}.availability", add)
        if not robot.kinematic_model_id.strip():
            add("EMPTY_KINEMATIC_MODEL", f"{path}.kinematic_model_id", "must be non-empty")
        if not robot.initial_joint_state_rad or not all(
            math.isfinite(x) for x in robot.initial_joint_state_rad
        ):
            add("INVALID_JOINT_STATE", f"{path}.initial_joint_state_rad", "must be finite and non-empty")
        if not math.isfinite(robot.nominal_cartesian_speed_m_s) or robot.nominal_cartesian_speed_m_s <= 0:
            add("INVALID_SPEED", f"{path}.nominal_cartesian_speed_m_s", "must be finite and > 0")

    for i, resource in enumerate(instance.resources):
        _check_window(resource.availability, f"resources[{i}].availability", add)
        if resource.capacity < 1:
            add("INVALID_RESOURCE_CAPACITY", f"resources[{i}].capacity", "must be >= 1")

    for parent_id, group in parent_groups.items():
        ordered = sorted(group, key=lambda x: x.segment_index)
        indices = [x.segment_index for x in ordered]
        if indices != list(range(len(ordered))):
            add("NONCONTIGUOUS_SEGMENT_INDEX", f"parent_curve[{parent_id}]", str(indices))
        if len(ordered) > 1 and any(x.handoff_policy is HandoffPolicy.NOT_SPLITTABLE for x in ordered):
            add("NOT_SPLITTABLE_PARENT", f"parent_curve[{parent_id}]", "contains multiple segments")
        for left, right in zip(ordered, ordered[1:]):
            if _distance(left.end_pose.position_m, right.start_pose.position_m) > _POINT_TOLERANCE_M:
                add("CURVE_DISCONTINUITY", f"parent_curve[{parent_id}]", f"{left.id}->{right.id}")

    if _has_precedence_cycle(instance.segments):
        add("PRECEDENCE_CYCLE", "segments.predecessor_ids", "precedence graph must be acyclic")
    return tuple(issues)


def load_instance(path: str | Path) -> AllocationInstance:
    with Path(path).open("r", encoding="utf-8") as handle:
        return allocation_instance_from_dict(json.load(handle))


def load_fixture(path: str | Path) -> tuple[AllocationInstance | None, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = dict(payload["expected"])
    try:
        return allocation_instance_from_dict(payload["instance"]), expected
    except SchemaValidationError as exc:
        expected["observed_issue_codes"] = sorted({x.code for x in exc.issues})
        return None, expected


def _check_unique(values: Iterable[str], path: str, code: str, add: Any) -> None:
    seen: set[str] = set()
    for value in values:
        if not value.strip():
            add("EMPTY_ID", path, "IDs must be non-empty")
        if value in seen:
            add(code, path, value)
        seen.add(value)


def _check_window(window: TimeWindow, path: str, add: Any) -> None:
    if not math.isfinite(window.start_s) or not math.isfinite(window.end_s) or window.end_s < window.start_s:
        add("INVALID_TIME_WINDOW", path, "requires finite end_s >= start_s")


def _check_pose(pose: Pose, path: str, add: Any) -> None:
    norm = math.sqrt(sum(x * x for x in pose.quaternion_xyzw))
    if abs(norm - 1.0) > _QUATERNION_TOLERANCE:
        add("NON_UNIT_QUATERNION", f"{path}.quaternion_xyzw", f"norm={norm:.6g}")


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _has_precedence_cycle(segments: Sequence[ProcessSegment]) -> bool:
    predecessors = {x.id: set(x.predecessor_ids) for x in segments}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(pred in predecessors and visit(pred) for pred in predecessors[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in predecessors)


def _pose_to_dict(pose: Pose) -> dict[str, list[float]]:
    return {"position_m": list(pose.position_m), "quaternion_xyzw": list(pose.quaternion_xyzw)}


def _window_to_dict(window: TimeWindow) -> dict[str, float]:
    return {"start_s": window.start_s, "end_s": window.end_s}


def _segment_to_dict(segment: ProcessSegment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "parent_curve_id": segment.parent_curve_id,
        "segment_index": segment.segment_index,
        "sampled_curve_m": [list(x) for x in segment.sampled_curve_m],
        "start_pose": _pose_to_dict(segment.start_pose),
        "end_pose": _pose_to_dict(segment.end_pose),
        "process_direction": segment.process_direction.value,
        "length_m": segment.length_m,
        "process_duration_s": segment.process_duration_s,
        "required_capabilities": list(segment.required_capabilities),
        "required_tool_id": segment.required_tool_id,
        "priority": segment.priority,
        "time_window": _window_to_dict(segment.time_window),
        "predecessor_ids": list(segment.predecessor_ids),
        "handoff_policy": segment.handoff_policy.value,
        "shared_resource_ids": list(segment.shared_resource_ids),
    }


def _robot_to_dict(robot: RobotSpec) -> dict[str, Any]:
    return {
        "id": robot.id,
        "base_pose": _pose_to_dict(robot.base_pose),
        "capabilities": list(robot.capabilities),
        "tool_ids": list(robot.tool_ids),
        "availability": _window_to_dict(robot.availability),
        "kinematic_model_id": robot.kinematic_model_id,
        "initial_joint_state_rad": list(robot.initial_joint_state_rad),
        "nominal_cartesian_speed_m_s": robot.nominal_cartesian_speed_m_s,
    }


def _resource_to_dict(resource: ResourceSpec) -> dict[str, Any]:
    return {
        "id": resource.id,
        "resource_type": resource.resource_type.value,
        "capacity": resource.capacity,
        "availability": _window_to_dict(resource.availability),
    }
