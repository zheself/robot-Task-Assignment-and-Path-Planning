"""Deterministic post-hoc verification for allocation-plan-v1.

This verifier checks allocation and proxy scheduling constraints.  It does not
certify IK feasibility, continuous collision freedom, or process physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .oracle import OracleContext, estimate_edge
from .features import point_distance
from .schema import AllocationInstance, AllocationPlan, HandoffPolicy, ProcessSegment, ScheduledSegment
from .scheduling import compute_proxy_objectives, segment_entry_exit

_TOL = 1e-9


@dataclass(frozen=True)
class PlanViolation:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PlanVerification:
    feasible: bool
    violations: tuple[PlanViolation, ...]
    objective_terms: tuple[tuple[str, float], ...]
    proxy_conflict_count: int
    diagnostics: tuple[str, ...]


def verify_plan(
    instance: AllocationInstance, plan: AllocationPlan, context: OracleContext
) -> PlanVerification:
    """Verify a plan against all A1 constraints with stable reason codes."""
    violations: list[PlanViolation] = []
    segment_by_id = {item.id: item for item in instance.segments}
    robot_by_id = {item.id: item for item in instance.robots}
    resource_by_id = {item.id: item for item in instance.resources}
    schedule_by_id: dict[str, ScheduledSegment] = {}

    def add(code: str, path: str, message: str) -> None:
        violations.append(PlanViolation(code, path, message))

    if plan.instance_id != instance.instance_id:
        add("INSTANCE_MISMATCH", "instance_id", "plan and instance IDs differ")
    for index, item in enumerate(plan.schedule):
        path = f"schedule[{index}]"
        if item.segment_id in schedule_by_id:
            add("DUPLICATE_SEGMENT", path, item.segment_id)
        schedule_by_id[item.segment_id] = item
        if item.segment_id not in segment_by_id:
            add("UNKNOWN_SEGMENT", path, item.segment_id)
        if item.robot_id not in robot_by_id:
            add("UNKNOWN_ROBOT", path, item.robot_id)
        if item.planned_end_s + _TOL < item.planned_start_s:
            add("NEGATIVE_DURATION", path, "end precedes start")
    missing = sorted(set(segment_by_id) - set(schedule_by_id))
    if missing:
        add("SEGMENT_COVERAGE", "schedule", f"missing={','.join(missing)}")

    valid_items = [
        item
        for item in plan.schedule
        if item.segment_id in segment_by_id and item.robot_id in robot_by_id
    ]
    for item in valid_items:
        segment = segment_by_id[item.segment_id]
        robot = robot_by_id[item.robot_id]
        path = f"schedule.{item.segment_id}"
        estimate = estimate_edge(robot, segment, context)
        if not estimate.feasible:
            add("EDGE_INFEASIBLE", path, ",".join(estimate.reason_codes))
        duration = item.planned_end_s - item.planned_start_s
        if duration + _TOL < segment.process_duration_s:
            add("PROCESS_DURATION", path, "scheduled duration is below process duration")
        if item.planned_start_s + _TOL < segment.time_window.start_s or item.planned_end_s > segment.time_window.end_s + _TOL:
            add("SEGMENT_TIME_WINDOW", path, "outside segment time window")
        if item.planned_start_s + _TOL < robot.availability.start_s or item.planned_end_s > robot.availability.end_s + _TOL:
            add("ROBOT_AVAILABILITY", path, "outside robot availability")
        for resource_id in segment.shared_resource_ids:
            resource = resource_by_id[resource_id]
            if item.planned_start_s + _TOL < resource.availability.start_s or item.planned_end_s > resource.availability.end_s + _TOL:
                add("RESOURCE_AVAILABILITY", path, resource_id)

    orders = _robot_orders(valid_items)
    for robot_id, ordered in orders.items():
        robot = robot_by_id[robot_id]
        indices = [item.order_index for item in ordered]
        if indices != list(range(len(ordered))):
            add("ROBOT_ORDER_INDEX", f"robots.{robot_id}", "indices must be contiguous from zero")
        chronological = sorted(ordered, key=lambda item: (item.planned_start_s, item.planned_end_s, item.segment_id))
        if [item.segment_id for item in chronological] != [item.segment_id for item in ordered]:
            add("ROBOT_ORDER_TIME", f"robots.{robot_id}", "order index conflicts with start time")
        for left, right in zip(ordered, ordered[1:]):
            if left.planned_end_s > right.planned_start_s + _TOL:
                add("ROBOT_OVERLAP", f"robots.{robot_id}", f"{left.segment_id},{right.segment_id}")
        source = robot.base_pose.position_m
        previous_end = robot.availability.start_s
        for item in ordered:
            entry, exit_point = segment_entry_exit(segment_by_id[item.segment_id], source)
            required_start = previous_end + point_distance(source, entry) / robot.nominal_cartesian_speed_m_s
            if item.planned_start_s + _TOL < required_start:
                add("ROBOT_TRANSITION_TIME", f"schedule.{item.segment_id}", f"required_start_s={required_start:.9g}")
            source = exit_point
            previous_end = item.planned_end_s

    for segment in instance.segments:
        current = schedule_by_id.get(segment.id)
        if current is None:
            continue
        for predecessor_id in segment.predecessor_ids:
            predecessor = schedule_by_id.get(predecessor_id)
            if predecessor is not None and predecessor.planned_end_s > current.planned_start_s + _TOL:
                add("PRECEDENCE", f"segments.{segment.id}", predecessor_id)

    for group in _parent_groups(instance):
        for left, right in zip(group, group[1:]):
            left_item, right_item = schedule_by_id.get(left.id), schedule_by_id.get(right.id)
            if left_item is not None and right_item is not None and left_item.planned_end_s > right_item.planned_start_s + _TOL:
                add("PARENT_SEGMENT_ORDER", f"segments.{right.id}", left.id)
        if any(item.handoff_policy in {HandoffPolicy.SAME_ROBOT, HandoffPolicy.NOT_SPLITTABLE} for item in group):
            assigned = {schedule_by_id[item.id].robot_id for item in group if item.id in schedule_by_id}
            if len(assigned) > 1:
                add("SAME_ROBOT_HANDOFF", f"curves.{group[0].parent_curve_id}", ",".join(sorted(assigned)))

    conflict_count = 0
    for resource_id, resource in resource_by_id.items():
        intervals = [
            item
            for item in valid_items
            if resource_id in segment_by_id[item.segment_id].shared_resource_ids
        ]
        for time in sorted({x.planned_start_s for x in intervals} | {x.planned_end_s for x in intervals}):
            active = [x for x in intervals if x.planned_start_s <= time + _TOL and time < x.planned_end_s - _TOL]
            if len(active) > resource.capacity:
                conflict_count += 1
                add("RESOURCE_CAPACITY", f"resources.{resource_id}", f"active={len(active)},capacity={resource.capacity}")

    objective_terms: tuple[tuple[str, float], ...] = ()
    if len(schedule_by_id) == len(segment_by_id) and len(valid_items) == len(instance.segments):
        robot_orders = {key: tuple(item.segment_id for item in value) for key, value in orders.items()}
        objective_terms = tuple(sorted(compute_proxy_objectives(instance, valid_items, robot_orders).items()))
    return PlanVerification(
        feasible=not violations,
        violations=tuple(_deduplicate(violations)),
        objective_terms=objective_terms,
        proxy_conflict_count=conflict_count,
        diagnostics=("A1_PROXY_CONSTRAINT_VERIFIER", "NOT_A_CONTINUOUS_COLLISION_CERTIFICATE"),
    )


def _robot_orders(items: Iterable[ScheduledSegment]) -> dict[str, list[ScheduledSegment]]:
    result: dict[str, list[ScheduledSegment]] = {}
    for item in items:
        result.setdefault(item.robot_id, []).append(item)
    for values in result.values():
        values.sort(key=lambda item: (item.order_index, item.segment_id))
    return result


def _parent_groups(instance: AllocationInstance) -> list[list[ProcessSegment]]:
    groups: dict[str, list[ProcessSegment]] = {}
    for segment in instance.segments:
        groups.setdefault(segment.parent_curve_id, []).append(segment)
    return [sorted(group, key=lambda item: item.segment_index) for group in groups.values()]


def _deduplicate(items: Iterable[PlanViolation]) -> list[PlanViolation]:
    result: list[PlanViolation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.code, item.path, item.message)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result
