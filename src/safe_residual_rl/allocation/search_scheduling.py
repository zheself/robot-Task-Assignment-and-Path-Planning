"""Bounded branch search over robot and shared-resource proxy sequences.

The search operates on the A1 geometric/timing proxy.  It is neither motion
planning nor a continuous-collision certificate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .features import point_distance
from .oracle import OracleContext, estimate_edge
from .schema import (
    PLAN_SCHEMA_VERSION,
    AllocationInstance,
    AllocationPlan,
    ScheduledSegment,
)
from .scheduling import (
    _parent_previous_map,
    _resource_feasible_start,
    compute_proxy_objectives,
    segment_entry_exit,
)


@dataclass(frozen=True)
class ScheduleSearchResult:
    status: str
    plan: AllocationPlan | None
    nodes_expanded: int
    search_complete: bool
    diagnostics: tuple[str, ...]


@dataclass
class _SearchState:
    scheduled: dict[str, ScheduledSegment]
    robot_orders: dict[str, tuple[str, ...]]
    robot_last_end: dict[str, float]
    robot_last_exit: dict[str, tuple[float, float, float]]
    resource_intervals: dict[str, tuple[tuple[float, float], ...]]
    total_travel_s: float
    dispatch_signature: tuple[str, ...]


def search_fixed_assignment_schedule(
    instance: AllocationInstance,
    assignment: Mapping[str, str],
    context: OracleContext,
    method_id: str,
    *,
    mode: str,
    objective_weights: Mapping[str, float],
    beam_width: int = 16,
    node_limit: int = 100_000,
    time_limit_s: float | None = None,
) -> ScheduleSearchResult:
    """Search feasible order/resource sequences for one fixed assignment.

    ``mode='exact'`` exhausts all deduplicated partial sequences unless a node
    or wall-time budget is reached. ``mode='beam'`` keeps a deterministic
    heuristic frontier and never claims completeness.
    """
    if mode not in {"exact", "beam"}:
        raise ValueError("mode must be exact or beam")
    if beam_width < 1 or node_limit < 1:
        raise ValueError("beam_width and node_limit must be positive")
    segment_ids = {item.id for item in instance.segments}
    robot_by_id = {item.id: item for item in instance.robots}
    if set(assignment) != segment_ids:
        return ScheduleSearchResult(
            "invalid_assignment", None, 0, True, ("SEGMENT_COVERAGE",)
        )
    if any(item not in robot_by_id for item in assignment.values()):
        return ScheduleSearchResult(
            "invalid_assignment", None, 0, True, ("UNKNOWN_ROBOT",)
        )
    segment_by_id = {item.id: item for item in instance.segments}
    if any(
        not estimate_edge(robot_by_id[robot_id], segment_by_id[segment_id], context).feasible
        for segment_id, robot_id in assignment.items()
    ):
        return ScheduleSearchResult(
            "invalid_assignment", None, 0, True, ("EDGE_INFEASIBLE",)
        )

    started = time.perf_counter()
    deadline = math.inf if time_limit_s is None else started + time_limit_s
    initial = _SearchState(
        scheduled={},
        robot_orders={item.id: () for item in instance.robots},
        robot_last_end={item.id: item.availability.start_s for item in instance.robots},
        robot_last_exit={item.id: item.base_pose.position_m for item in instance.robots},
        resource_intervals={item.id: () for item in instance.resources},
        total_travel_s=0.0,
        dispatch_signature=(),
    )
    nodes = 0
    budget_exhausted = False
    best_plan: AllocationPlan | None = None
    best_score = math.inf

    if mode == "exact":
        frontier = [initial]
        seen: set[tuple[object, ...]] = set()
        while frontier:
            if nodes >= node_limit or time.perf_counter() >= deadline:
                budget_exhausted = True
                break
            state = frontier.pop()
            key = _state_key(state)
            if key in seen:
                continue
            seen.add(key)
            if len(state.scheduled) == len(instance.segments):
                plan = _to_plan(instance, state, method_id)
                score = _score(plan, objective_weights)
                if score + 1e-12 < best_score:
                    best_plan, best_score = plan, score
                continue
            children = _expand(instance, assignment, state)
            nodes += len(children)
            frontier.extend(reversed(children))
    else:
        frontier = [initial]
        for _ in range(len(instance.segments)):
            candidates: list[_SearchState] = []
            for state in frontier:
                if nodes >= node_limit or time.perf_counter() >= deadline:
                    budget_exhausted = True
                    break
                children = _expand(instance, assignment, state)
                remaining = node_limit - nodes
                if len(children) > remaining:
                    children = children[: max(0, remaining)]
                    budget_exhausted = True
                nodes += len(children)
                candidates.extend(children)
            if not candidates:
                frontier = []
                break
            unique: dict[tuple[object, ...], _SearchState] = {}
            for state in candidates:
                unique.setdefault(_state_key(state), state)
            ranked = sorted(
                unique.values(),
                key=lambda item: _beam_rank(instance, assignment, item),
            )
            frontier = ranked[:beam_width]
            if len(ranked) > beam_width:
                budget_exhausted = True
        for state in frontier:
            if len(state.scheduled) != len(instance.segments):
                continue
            plan = _to_plan(instance, state, method_id)
            score = _score(plan, objective_weights)
            if score + 1e-12 < best_score:
                best_plan, best_score = plan, score

    search_complete = mode == "exact" and not budget_exhausted
    if best_plan is None:
        status = "infeasible" if search_complete else "limit"
    else:
        status = "optimal" if search_complete else "feasible_limit"
    return ScheduleSearchResult(
        status,
        best_plan,
        nodes,
        search_complete,
        (
            f"SEQUENCE_SEARCH_MODE={mode}",
            f"BEAM_WIDTH={beam_width}",
            f"NODE_LIMIT={node_limit}",
            f"NODES_EXPANDED={nodes}",
            f"SEARCH_COMPLETE={search_complete}",
            "BRANCHES_ROBOT_AND_SHARED_RESOURCE_ORDER",
            "PROXY_TIMING_ONLY",
            "NOT_MOTION_PLANNING_OR_COLLISION_CERTIFICATE",
        ),
    )


def _expand(
    instance: AllocationInstance,
    assignment: Mapping[str, str],
    state: _SearchState,
) -> list[_SearchState]:
    segment_by_id = {item.id: item for item in instance.segments}
    robot_by_id = {item.id: item for item in instance.robots}
    resource_by_id = {item.id: item for item in instance.resources}
    parent_previous = _parent_previous_map(instance.segments)
    children: list[_SearchState] = []
    for segment_id in sorted(segment_by_id):
        if segment_id in state.scheduled:
            continue
        segment = segment_by_id[segment_id]
        dependencies = set(segment.predecessor_ids)
        if segment_id in parent_previous:
            dependencies.add(parent_previous[segment_id])
        if not dependencies.issubset(state.scheduled):
            continue
        robot_id = assignment[segment_id]
        robot = robot_by_id[robot_id]
        source = state.robot_last_exit[robot_id]
        entry, exit_point = segment_entry_exit(segment, source)
        travel_s = point_distance(source, entry) / robot.nominal_cartesian_speed_m_s
        start_s = max(
            segment.time_window.start_s,
            robot.availability.start_s,
            state.robot_last_end[robot_id] + travel_s,
        )
        if dependencies:
            start_s = max(
                start_s,
                max(state.scheduled[item].planned_end_s for item in dependencies),
            )
        start_s = _resource_feasible_start(
            start_s,
            segment.process_duration_s,
            segment.shared_resource_ids,
            {key: list(value) for key, value in state.resource_intervals.items()},
            resource_by_id,
        )
        end_s = start_s + segment.process_duration_s
        latest_end = min(
            segment.time_window.end_s,
            robot.availability.end_s,
            *(
                [resource_by_id[item].availability.end_s for item in segment.shared_resource_ids]
                or [math.inf]
            ),
        )
        if end_s > latest_end + 1e-12:
            continue

        scheduled = dict(state.scheduled)
        scheduled[segment_id] = ScheduledSegment(
            segment_id,
            robot_id,
            len(state.robot_orders[robot_id]),
            start_s,
            end_s,
        )
        robot_orders = dict(state.robot_orders)
        robot_orders[robot_id] = robot_orders[robot_id] + (segment_id,)
        robot_last_end = dict(state.robot_last_end)
        robot_last_end[robot_id] = end_s
        robot_last_exit = dict(state.robot_last_exit)
        robot_last_exit[robot_id] = exit_point
        resource_intervals = dict(state.resource_intervals)
        for resource_id in segment.shared_resource_ids:
            resource_intervals[resource_id] = tuple(
                sorted(resource_intervals[resource_id] + ((start_s, end_s),))
            )
        children.append(
            _SearchState(
                scheduled,
                robot_orders,
                robot_last_end,
                robot_last_exit,
                resource_intervals,
                state.total_travel_s + travel_s,
                state.dispatch_signature + (segment_id,),
            )
        )
    return children


def _beam_rank(
    instance: AllocationInstance,
    assignment: Mapping[str, str],
    state: _SearchState,
) -> tuple[float | str, ...]:
    segment_by_id = {item.id: item for item in instance.segments}
    robot_by_id = {item.id: item for item in instance.robots}
    remaining_risk = 0.0
    negative_slack = 0
    for segment_id, segment in segment_by_id.items():
        if segment_id in state.scheduled:
            continue
        robot = robot_by_id[assignment[segment_id]]
        source = state.robot_last_exit[robot.id]
        entry, _ = segment_entry_exit(segment, source)
        earliest_end = max(
            segment.time_window.start_s,
            state.robot_last_end[robot.id]
            + point_distance(source, entry) / robot.nominal_cartesian_speed_m_s,
        ) + segment.process_duration_s
        slack = min(segment.time_window.end_s, robot.availability.end_s) - earliest_end
        if slack < 0:
            negative_slack += 1
            remaining_risk += -slack
        else:
            remaining_risk += 1.0 / (1.0 + slack)
    return (
        float(negative_slack),
        remaining_risk,
        max(state.robot_last_end.values(), default=0.0),
        state.total_travel_s,
        "|".join(state.dispatch_signature),
    )


def _state_key(state: _SearchState) -> tuple[object, ...]:
    return (
        tuple(sorted(state.robot_orders.items())),
        tuple(
            sorted(
                (
                    item.segment_id,
                    item.robot_id,
                    round(item.planned_start_s, 10),
                    round(item.planned_end_s, 10),
                )
                for item in state.scheduled.values()
            )
        ),
        tuple(sorted(state.resource_intervals.items())),
    )


def _to_plan(
    instance: AllocationInstance, state: _SearchState, method_id: str
) -> AllocationPlan:
    schedule = tuple(
        sorted(state.scheduled.values(), key=lambda item: (item.robot_id, item.order_index))
    )
    objectives = compute_proxy_objectives(instance, schedule, state.robot_orders)
    return AllocationPlan(
        PLAN_SCHEMA_VERSION,
        instance.instance_id,
        method_id,
        schedule,
        "feasible",
        tuple(sorted(objectives.items())),
        (
            "BRANCHED_SEQUENCE_PROXY_SCHEDULE",
            "NOT_MOTION_PLANNING_OR_COLLISION_CERTIFICATE",
        ),
    )


def _score(plan: AllocationPlan, weights: Mapping[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in plan.objective_terms)
