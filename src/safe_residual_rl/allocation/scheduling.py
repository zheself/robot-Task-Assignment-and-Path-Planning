"""Deterministic list scheduling for fixed robot assignments and orders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .features import point_distance
from .oracle import OracleContext
from .schema import (
    PLAN_SCHEMA_VERSION,
    AllocationInstance,
    AllocationPlan,
    ProcessDirection,
    ProcessSegment,
    ScheduledSegment,
)


@dataclass(frozen=True)
class ScheduleBuildResult:
    status: str
    plan: AllocationPlan | None
    diagnostics: tuple[str, ...]


def build_schedule(
    instance: AllocationInstance,
    robot_orders: Mapping[str, Sequence[str]],
    context: OracleContext,
    method_id: str,
) -> ScheduleBuildResult:
    """Build a deterministic proxy schedule; it is not motion planning."""
    segment_by_id = {segment.id: segment for segment in instance.segments}
    robot_by_id = {robot.id: robot for robot in instance.robots}
    assigned = [segment_id for order in robot_orders.values() for segment_id in order]
    if sorted(assigned) != sorted(segment_by_id) or len(assigned) != len(set(assigned)):
        return ScheduleBuildResult("invalid_assignment", None, ("SEGMENT_COVERAGE",))
    if any(robot_id not in robot_by_id for robot_id in robot_orders):
        return ScheduleBuildResult("invalid_assignment", None, ("UNKNOWN_ROBOT",))

    assignment = {
        segment_id: robot_id
        for robot_id, order in robot_orders.items()
        for segment_id in order
    }
    parent_previous = _parent_previous_map(instance.segments)
    resource_by_id = {resource.id: resource for resource in instance.resources}
    scheduled: dict[str, ScheduledSegment] = {}
    selected_exit: dict[str, tuple[float, float, float]] = {}
    resource_intervals: dict[str, list[tuple[float, float]]] = {
        resource.id: [] for resource in instance.resources
    }
    next_index = {robot_id: 0 for robot_id in robot_orders}
    diagnostics = ["DETERMINISTIC_LIST_SCHEDULER", "PROXY_TIMING_ONLY"]

    while len(scheduled) < len(segment_by_id):
        candidates: list[
            tuple[float, int, str, str, tuple[float, float, float], tuple[float, float, float]]
        ] = []
        for robot_id, order in robot_orders.items():
            cursor = next_index[robot_id]
            if cursor >= len(order):
                continue
            segment_id = order[cursor]
            segment = segment_by_id[segment_id]
            dependencies = set(segment.predecessor_ids)
            if segment_id in parent_previous:
                dependencies.add(parent_previous[segment_id])
            if not dependencies.issubset(scheduled):
                continue
            robot = robot_by_id[robot_id]
            prior_id = order[cursor - 1] if cursor > 0 else None
            source = selected_exit[prior_id] if prior_id is not None else robot.base_pose.position_m
            entry, exit_point = segment_entry_exit(segment, source)
            transition_s = point_distance(source, entry) / robot.nominal_cartesian_speed_m_s
            earliest = max(segment.time_window.start_s, robot.availability.start_s)
            if prior_id is not None:
                earliest = max(earliest, scheduled[prior_id].planned_end_s + transition_s)
            else:
                earliest = max(earliest, robot.availability.start_s + transition_s)
            if dependencies:
                earliest = max(earliest, max(scheduled[item].planned_end_s for item in dependencies))
            earliest = _resource_feasible_start(
                earliest,
                segment.process_duration_s,
                segment.shared_resource_ids,
                resource_intervals,
                resource_by_id,
            )
            candidates.append((earliest, -segment.priority, segment_id, robot_id, entry, exit_point))
        if not candidates:
            remaining = sorted(set(segment_by_id) - set(scheduled))
            return ScheduleBuildResult(
                "infeasible",
                None,
                tuple(diagnostics + [f"DEPENDENCY_OR_ORDER_DEADLOCK={','.join(remaining)}"]),
            )
        start_s, _, segment_id, robot_id, _, exit_point = min(candidates)
        segment = segment_by_id[segment_id]
        robot = robot_by_id[robot_id]
        end_s = start_s + segment.process_duration_s
        resource_latest = min(
            [resource_by_id[item].availability.end_s for item in segment.shared_resource_ids]
            or [math.inf]
        )
        if end_s > min(segment.time_window.end_s, robot.availability.end_s, resource_latest) + 1e-12:
            return ScheduleBuildResult(
                "infeasible",
                None,
                tuple(diagnostics + [f"TIME_OR_RESOURCE_WINDOW={segment_id}"]),
            )
        cursor = next_index[robot_id]
        item = ScheduledSegment(segment_id, robot_id, cursor, start_s, end_s)
        scheduled[segment_id] = item
        selected_exit[segment_id] = exit_point
        next_index[robot_id] += 1
        for resource_id in segment.shared_resource_ids:
            resource_intervals[resource_id].append((start_s, end_s))

    schedule = tuple(
        sorted(scheduled.values(), key=lambda item: (item.robot_id, item.order_index))
    )
    objective_terms = compute_proxy_objectives(instance, schedule, robot_orders)
    return ScheduleBuildResult(
        "feasible",
        AllocationPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            instance_id=instance.instance_id,
            method_id=method_id,
            schedule=schedule,
            solver_status="feasible",
            objective_terms=tuple(sorted(objective_terms.items())),
            diagnostics=tuple(diagnostics),
        ),
        tuple(diagnostics),
    )


def build_deadline_aware_schedule(
    instance: AllocationInstance,
    assignment: Mapping[str, str],
    context: OracleContext,
    method_id: str,
) -> ScheduleBuildResult:
    """Schedule a fixed assignment with precedence-aware minimum-slack dispatch.

    Unlike :func:`build_schedule`, this routine does not freeze a robot order
    before timing is known.  At each dispatch step it considers every currently
    precedence-ready segment, computes its earliest resource-feasible start and
    selects minimum remaining slack.  It is still a deterministic geometric
    proxy scheduler, not a collision-safe or globally optimal scheduler.
    """
    segment_by_id = {segment.id: segment for segment in instance.segments}
    robot_by_id = {robot.id: robot for robot in instance.robots}
    if set(assignment) != set(segment_by_id):
        return ScheduleBuildResult("invalid_assignment", None, ("SEGMENT_COVERAGE",))
    if any(robot_id not in robot_by_id for robot_id in assignment.values()):
        return ScheduleBuildResult("invalid_assignment", None, ("UNKNOWN_ROBOT",))

    parent_previous = _parent_previous_map(instance.segments)
    resource_by_id = {resource.id: resource for resource in instance.resources}
    resource_intervals: dict[str, list[tuple[float, float]]] = {
        resource.id: [] for resource in instance.resources
    }
    scheduled: dict[str, ScheduledSegment] = {}
    robot_last_end = {
        robot.id: robot.availability.start_s for robot in instance.robots
    }
    robot_last_exit = {
        robot.id: robot.base_pose.position_m for robot in instance.robots
    }
    robot_order_index = {robot.id: 0 for robot in instance.robots}
    robot_orders: dict[str, list[str]] = {robot.id: [] for robot in instance.robots}
    diagnostics = [
        "DEADLINE_AWARE_MINIMUM_SLACK_SCHEDULER_V2",
        "PRECEDENCE_AND_RESOURCE_AWARE_DISPATCH",
        "PROXY_TIMING_ONLY",
    ]

    while len(scheduled) < len(segment_by_id):
        candidates: list[
            tuple[
                float,
                float,
                float,
                int,
                str,
                str,
                tuple[float, float, float],
            ]
        ] = []
        for segment_id, segment in segment_by_id.items():
            if segment_id in scheduled:
                continue
            dependencies = set(segment.predecessor_ids)
            if segment_id in parent_previous:
                dependencies.add(parent_previous[segment_id])
            if not dependencies.issubset(scheduled):
                continue
            robot_id = assignment[segment_id]
            robot = robot_by_id[robot_id]
            source = robot_last_exit[robot_id]
            entry, exit_point = segment_entry_exit(segment, source)
            transition_s = point_distance(source, entry) / robot.nominal_cartesian_speed_m_s
            earliest = max(
                segment.time_window.start_s,
                robot.availability.start_s,
                robot_last_end[robot_id] + transition_s,
            )
            if dependencies:
                earliest = max(
                    earliest,
                    max(scheduled[item].planned_end_s for item in dependencies),
                )
            earliest = _resource_feasible_start(
                earliest,
                segment.process_duration_s,
                segment.shared_resource_ids,
                resource_intervals,
                resource_by_id,
            )
            latest_end = min(
                segment.time_window.end_s,
                robot.availability.end_s,
                *(
                    [resource_by_id[item].availability.end_s for item in segment.shared_resource_ids]
                    or [math.inf]
                ),
            )
            slack = latest_end - (earliest + segment.process_duration_s)
            candidates.append(
                (
                    slack,
                    latest_end,
                    earliest,
                    -segment.priority,
                    segment_id,
                    robot_id,
                    exit_point,
                )
            )
        if not candidates:
            remaining = sorted(set(segment_by_id) - set(scheduled))
            return ScheduleBuildResult(
                "infeasible",
                None,
                tuple(diagnostics + [f"DEPENDENCY_DEADLOCK={','.join(remaining)}"]),
            )

        slack, _, start_s, _, segment_id, robot_id, exit_point = min(candidates)
        segment = segment_by_id[segment_id]
        end_s = start_s + segment.process_duration_s
        if slack < -1e-12:
            return ScheduleBuildResult(
                "infeasible",
                None,
                tuple(diagnostics + [f"NEGATIVE_DISPATCH_SLACK={segment_id}:{slack:.9g}"]),
            )
        order_index = robot_order_index[robot_id]
        scheduled[segment_id] = ScheduledSegment(
            segment_id, robot_id, order_index, start_s, end_s
        )
        robot_order_index[robot_id] += 1
        robot_orders[robot_id].append(segment_id)
        robot_last_end[robot_id] = end_s
        robot_last_exit[robot_id] = exit_point
        for resource_id in segment.shared_resource_ids:
            resource_intervals[resource_id].append((start_s, end_s))

    schedule = tuple(
        sorted(scheduled.values(), key=lambda item: (item.robot_id, item.order_index))
    )
    objective_terms = compute_proxy_objectives(instance, schedule, robot_orders)
    return ScheduleBuildResult(
        "feasible",
        AllocationPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            instance_id=instance.instance_id,
            method_id=method_id,
            schedule=schedule,
            solver_status="feasible",
            objective_terms=tuple(sorted(objective_terms.items())),
            diagnostics=tuple(diagnostics),
        ),
        tuple(diagnostics),
    )


def transition_time_s(
    previous: ProcessSegment | None,
    current: ProcessSegment,
    robot_base: tuple[float, float, float],
    speed_m_s: float,
) -> float:
    source = robot_base if previous is None else segment_entry_exit(previous, robot_base)[1]
    entry, _ = segment_entry_exit(current, source)
    return point_distance(source, entry) / speed_m_s


def segment_entry_exit(
    segment: ProcessSegment, source: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    forward = (segment.start_pose.position_m, segment.end_pose.position_m)
    reverse = (segment.end_pose.position_m, segment.start_pose.position_m)
    if segment.process_direction is ProcessDirection.FORWARD:
        return forward
    if segment.process_direction is ProcessDirection.REVERSE:
        return reverse
    return min((forward, reverse), key=lambda pair: point_distance(source, pair[0]))


def _parent_previous_map(segments: Sequence[ProcessSegment]) -> dict[str, str]:
    groups: dict[str, list[ProcessSegment]] = {}
    for segment in segments:
        groups.setdefault(segment.parent_curve_id, []).append(segment)
    result: dict[str, str] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.segment_index)
        result.update({right.id: left.id for left, right in zip(ordered, ordered[1:])})
    return result


def _resource_feasible_start(
    candidate: float,
    duration: float,
    resource_ids: Sequence[str],
    intervals: Mapping[str, list[tuple[float, float]]],
    resource_by_id: Mapping[str, object],
) -> float:
    if not resource_ids:
        return candidate
    candidate = max(
        candidate,
        max(getattr(resource_by_id[item], "availability").start_s for item in resource_ids),
    )
    while True:
        end = candidate + duration
        blocked_ends: list[float] = []
        for resource_id in resource_ids:
            capacity = getattr(resource_by_id[resource_id], "capacity")
            active = [item for item in intervals[resource_id] if item[0] < end and item[1] > candidate]
            event_times = [candidate] + [item[0] for item in active if candidate < item[0] < end]
            if any(
                1 + sum(start <= time < stop for start, stop in active) > capacity
                for time in event_times
            ):
                blocked_ends.extend(stop for _, stop in active if stop > candidate)
        if not blocked_ends:
            return candidate
        candidate = min(blocked_ends)


def compute_proxy_objectives(
    instance: AllocationInstance,
    schedule: Sequence[ScheduledSegment],
    robot_orders: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    segment_by_id = {item.id: item for item in instance.segments}
    robot_by_id = {item.id: item for item in instance.robots}
    makespan = max((item.planned_end_s for item in schedule), default=0.0)
    loads = [sum(segment_by_id[item].process_duration_s for item in robot_orders.get(robot.id, ())) for robot in instance.robots]
    mean_load = sum(loads) / len(loads)
    load_variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
    travel = 0.0
    for robot_id, order in robot_orders.items():
        robot = robot_by_id[robot_id]
        previous: ProcessSegment | None = None
        for segment_id in order:
            current = segment_by_id[segment_id]
            travel += transition_time_s(
                previous, current, robot.base_pose.position_m, robot.nominal_cartesian_speed_m_s
            )
            previous = current
    tardiness = sum(
        segment_by_id[item.segment_id].priority
        * max(0.0, item.planned_end_s - segment_by_id[item.segment_id].time_window.end_s)
        for item in schedule
    )
    return {
        "makespan": makespan,
        "load_variance": load_variance,
        "travel_setup_time": travel,
        "priority_tardiness": tardiness,
    }
