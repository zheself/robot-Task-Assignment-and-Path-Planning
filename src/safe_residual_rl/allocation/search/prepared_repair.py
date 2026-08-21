"""Prepared, semantics-preserving structural diagnostics for Protocol R.

This module removes repeated construction of immutable instance tables.  It
does not prune or reorder candidates and does not replace the A1 scheduler or
verifier.  The reference implementation remains :mod:`diagnostics`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from ..oracle import OracleContext
from ..repair import InitializerState
from ..repair.identical import StateEvaluation, canonicalize_state
from ..schema import AllocationInstance
from ..scheduling import build_schedule
from ..solvers.common import allocation_units, edge_mask_and_costs
from ..verifier import verify_plan
from .diagnostics import EvaluationTiming, StateDiagnostic, TimedEvaluation, ViolationVector
from .trace import canonical_hash


@dataclass(frozen=True)
class PreparedRepairProblem:
    """Immutable tables shared by every candidate in one search trace."""

    instance_id: str
    bound_object_id: int
    units: tuple[tuple[str, ...], ...]
    robots: tuple[str, ...]
    costs: tuple[tuple[float, ...], ...]
    segment_to_unit: Mapping[str, int]
    precedence_edges: frozenset[tuple[int, int]]
    unit_starts: tuple[float, ...]
    unit_ends: tuple[float, ...]
    unit_process_durations: tuple[float, ...]
    unit_resources: tuple[frozenset[str], ...]
    prepared_sha256: str


def prepare_repair_problem(
    instance: AllocationInstance, context: OracleContext
) -> PreparedRepairProblem:
    """Build immutable values once without changing their reference ordering."""
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    segment_by_id = {item.id: item for item in instance.segments}
    segment_to_unit = {
        segment_id: index for index, unit in enumerate(units) for segment_id in unit
    }
    precedence: set[tuple[int, int]] = set()
    for right, unit in enumerate(units):
        for segment_id in unit:
            for predecessor in segment_by_id[segment_id].predecessor_ids:
                left = segment_to_unit[predecessor]
                if left != right:
                    precedence.add((left, right))
    starts = tuple(
        min(segment_by_id[item].time_window.start_s for item in unit) for unit in units
    )
    ends = tuple(
        min(segment_by_id[item].time_window.end_s for item in unit) for unit in units
    )
    process_durations = tuple(
        sum(segment_by_id[item].process_duration_s for item in unit) for unit in units
    )
    resources = tuple(
        frozenset(
            resource
            for segment_id in unit
            for resource in segment_by_id[segment_id].shared_resource_ids
        )
        for unit in units
    )
    payload = {
        "instance_id": instance.instance_id,
        "units": [list(item) for item in units],
        "robots": list(robots),
        "costs": [
            [float(value) if math.isfinite(value) else None for value in row]
            for row in costs
        ],
        "segment_to_unit": dict(sorted(segment_to_unit.items())),
        "precedence_edges": [list(item) for item in sorted(precedence)],
        "unit_starts": list(starts),
        "unit_ends": list(ends),
        "unit_process_durations": list(process_durations),
        "unit_resources": [sorted(item) for item in resources],
    }
    return PreparedRepairProblem(
        instance.instance_id,
        id(instance),
        units,
        robots,
        costs,
        segment_to_unit,
        frozenset(precedence),
        starts,
        ends,
        process_durations,
        resources,
        canonical_hash(payload),
    )


def _failure_segments(diagnostics: tuple[str, ...]) -> tuple[str, ...]:
    result: set[str] = set()
    for item in diagnostics:
        if item.startswith("DEPENDENCY_OR_ORDER_DEADLOCK="):
            result.update(part for part in item.split("=", 1)[1].split(",") if part)
        elif item.startswith("TIME_OR_RESOURCE_WINDOW="):
            result.add(item.split("=", 1)[1])
        elif item.startswith("DEPENDENCY_DEADLOCK="):
            result.update(part for part in item.split("=", 1)[1].split(",") if part)
        elif item.startswith("NEGATIVE_DISPATCH_SLACK="):
            result.add(item.split("=", 1)[1].split(":", 1)[0])
    return tuple(sorted(result))


def analyze_state_prepared(
    instance: AllocationInstance,
    prepared: PreparedRepairProblem,
    state: InitializerState,
    *,
    scheduler_diagnostics: tuple[str, ...] = (),
) -> StateDiagnostic:
    """Exact structural equivalent of ``diagnostics.analyze_state``."""
    if instance.instance_id != prepared.instance_id or id(instance) != prepared.bound_object_id:
        raise ValueError("prepared repair problem belongs to another instance")
    state = canonicalize_state(instance, state)
    orders = state.order_map()
    positions = {
        unit_index: (robot, position)
        for robot, order in orders.items()
        for position, unit_index in enumerate(order)
    }
    missing = sum(robot is None for robot in state.assignments)
    invalid = 0
    loads = {robot: 0.0 for robot in prepared.robots}
    robot_indices = {robot: index for index, robot in enumerate(prepared.robots)}
    for index, robot in enumerate(state.assignments):
        if robot is None:
            continue
        if robot not in robot_indices:
            invalid += 1
            continue
        value = prepared.costs[index][robot_indices[robot]]
        if not math.isfinite(value):
            invalid += 1
        else:
            loads[robot] += value

    order_violations = 0
    for left, right in prepared.precedence_edges:
        if left in positions and right in positions:
            left_robot, left_position = positions[left]
            right_robot, right_position = positions[right]
            if left_robot == right_robot and left_position > right_position:
                order_violations += 1

    assigned = {index for index, robot in enumerate(state.assignments) if robot is not None}
    graph_edges = {
        (left, right)
        for left, right in prepared.precedence_edges
        if left in assigned and right in assigned
    }
    for order in orders.values():
        graph_edges.update((left, right) for left, right in zip(order, order[1:]))
    successors = {index: set() for index in assigned}
    indegree = {index: 0 for index in assigned}
    for left, right in graph_edges:
        if right not in successors[left]:
            successors[left].add(right)
            indegree[right] += 1
    ready = sorted(index for index, degree in indegree.items() if degree == 0)
    topological: list[int] = []
    while ready:
        current = ready.pop(0)
        topological.append(current)
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    deadlock = len(assigned) - len(topological)

    durations: dict[int, float] = {}
    for index in range(len(prepared.units)):
        robot = state.assignments[index]
        edge_duration = (
            prepared.costs[index][robot_indices[robot]]
            if robot in robot_indices
            else math.inf
        )
        durations[index] = (
            float(edge_duration)
            if math.isfinite(edge_duration)
            else prepared.unit_process_durations[index]
        )
    predecessors = {index: set() for index in assigned}
    for left, right in graph_edges:
        predecessors[right].add(left)
    earliest_end: dict[int, float] = {}
    lateness = 0.0
    intervals: dict[int, tuple[float, float]] = {}
    for index in topological:
        earliest = prepared.unit_starts[index]
        if predecessors[index]:
            earliest = max(earliest, max(earliest_end[item] for item in predecessors[index]))
        finish = earliest + durations[index]
        earliest_end[index] = finish
        intervals[index] = (earliest, finish)
        lateness += max(0.0, finish - prepared.unit_ends[index])

    resource_overlap = 0.0
    scheduled_indices = sorted(intervals)
    for left_offset, left in enumerate(scheduled_indices):
        for right in scheduled_indices[left_offset + 1 :]:
            if not prepared.unit_resources[left] & prepared.unit_resources[right]:
                continue
            left_start, left_end = intervals[left]
            right_start, right_end = intervals[right]
            resource_overlap += max(
                0.0, min(left_end, right_end) - max(left_start, right_start)
            )

    failure_segments = _failure_segments(scheduler_diagnostics)
    failure_units = tuple(
        sorted(
            {
                prepared.segment_to_unit[item]
                for item in failure_segments
                if item in prepared.segment_to_unit
            }
        )
    )
    return StateDiagnostic(
        ViolationVector(
            missing_units=missing,
            invalid_edges=invalid,
            precedence_deadlock_units=deadlock,
            precedence_order_violations=order_violations,
            time_window_lateness_s=lateness,
            shared_resource_overlap_s=resource_overlap,
            max_load_s=max(loads.values(), default=0.0),
        ),
        failure_segments,
        failure_units,
    )


def evaluate_state_timed_prepared(
    instance: AllocationInstance,
    context: OracleContext,
    prepared: PreparedRepairProblem,
    state: InitializerState,
    weights: Mapping[str, float],
    *,
    method_id: str = "a4b-v2-shared-repair",
    clock: Callable[[], int] = time.monotonic_ns,
) -> TimedEvaluation:
    """Run the unchanged scheduler/verifier around prepared diagnostics."""
    structural_started = clock()
    state = canonicalize_state(instance, state)
    diagnostic = analyze_state_prepared(instance, prepared, state)
    structural_s = (clock() - structural_started) / 1e9
    if diagnostic.vector.missing_units or diagnostic.vector.invalid_edges:
        reason = (
            "initializer_incomplete"
            if diagnostic.vector.missing_units
            else "mask_integrity_failure"
        )
        evaluation = StateEvaluation(False, None, None, diagnostic.vector.scalar(), reason)
        return TimedEvaluation(
            evaluation, diagnostic, EvaluationTiming(structural_s, 0.0, 0.0), ()
        )

    orders = {
        robot: tuple(
            segment for unit_index in order for segment in prepared.units[unit_index]
        )
        for robot, order in state.robot_orders
    }
    scheduler_started = clock()
    built = build_schedule(instance, orders, context, method_id)
    scheduler_s = (clock() - scheduler_started) / 1e9
    diagnostic = analyze_state_prepared(
        instance, prepared, state, scheduler_diagnostics=built.diagnostics
    )
    if built.plan is None:
        joined = " ".join(built.diagnostics)
        if "TIME_OR_RESOURCE" in joined or "NEGATIVE_DISPATCH_SLACK" in joined:
            reason = "time_window_failure"
        elif "DEPENDENCY" in joined:
            reason = "precedence_failure"
        else:
            reason = "schedule_infeasible"
        evaluation = StateEvaluation(
            False, None, None, 100_000.0 + diagnostic.vector.scalar(), reason
        )
        return TimedEvaluation(
            evaluation,
            diagnostic,
            EvaluationTiming(structural_s, scheduler_s, 0.0),
            built.diagnostics,
        )

    verifier_started = clock()
    checked = verify_plan(instance, built.plan, context)
    verifier_s = (clock() - verifier_started) / 1e9
    if not checked.feasible:
        codes = {item.code for item in checked.violations}
        if any("RESOURCE" in item for item in codes):
            reason = "shared_resource_failure"
        elif any("PRECEDENCE" in item or "ORDER" in item for item in codes):
            reason = "precedence_failure"
        elif any("TIME" in item for item in codes):
            reason = "time_window_failure"
        else:
            reason = "schedule_infeasible"
        evaluation = StateEvaluation(
            False, built.plan, None, 100_000.0 + diagnostic.vector.scalar(), reason
        )
    else:
        terms = dict(checked.objective_terms)
        objective = sum(
            float(weights.get(key, 0.0)) * float(value) for key, value in terms.items()
        )
        evaluation = StateEvaluation(True, built.plan, objective, objective, None)
    return TimedEvaluation(
        evaluation,
        diagnostic,
        EvaluationTiming(structural_s, scheduler_s, verifier_s),
        built.diagnostics,
    )
