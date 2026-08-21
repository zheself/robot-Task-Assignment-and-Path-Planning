"""Structured, search-side diagnostics for A4b ordinary-LNS recovery v2.

The A1 scheduler and verifier remain the sole feasibility authorities.  This
module adds deterministic partial-state signals for destroy/repair ranking and
times the unchanged scheduling and verification calls separately.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

from ..oracle import OracleContext
from ..repair import InitializerState
from ..repair.identical import StateEvaluation, canonicalize_state
from ..schema import AllocationInstance
from ..scheduling import build_schedule
from ..solvers.common import allocation_units, edge_mask_and_costs
from ..verifier import verify_plan


@dataclass(frozen=True)
class ViolationVector:
    missing_units: int = 0
    invalid_edges: int = 0
    precedence_deadlock_units: int = 0
    precedence_order_violations: int = 0
    time_window_lateness_s: float = 0.0
    shared_resource_overlap_s: float = 0.0
    max_load_s: float = 0.0

    def rank(self) -> tuple[float, ...]:
        return (
            float(self.missing_units),
            float(self.invalid_edges),
            float(self.precedence_deadlock_units),
            float(self.precedence_order_violations),
            float(self.time_window_lateness_s),
            float(self.shared_resource_overlap_s),
            float(self.max_load_s),
        )

    def scalar(self) -> float:
        return (
            1_000_000.0 * self.missing_units
            + 1_000_000.0 * self.invalid_edges
            + 100_000.0 * self.precedence_deadlock_units
            + 10_000.0 * self.precedence_order_violations
            + 100.0 * self.time_window_lateness_s
            + 10.0 * self.shared_resource_overlap_s
            + self.max_load_s
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class StateDiagnostic:
    vector: ViolationVector
    failure_segment_ids: tuple[str, ...]
    failure_unit_indices: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationTiming:
    structural_s: float
    scheduler_s: float
    verifier_s: float

    @property
    def total_s(self) -> float:
        return self.structural_s + self.scheduler_s + self.verifier_s

    def to_dict(self) -> dict[str, float]:
        return {
            "structural_s": self.structural_s,
            "scheduler_s": self.scheduler_s,
            "verifier_s": self.verifier_s,
            "total_s": self.total_s,
        }


@dataclass(frozen=True)
class TimedEvaluation:
    evaluation: StateEvaluation
    diagnostic: StateDiagnostic
    timing: EvaluationTiming
    scheduler_diagnostics: tuple[str, ...]


def _unit_problem(instance: AllocationInstance):
    units = allocation_units(instance)
    segment_by_id = {item.id: item for item in instance.segments}
    segment_to_unit = {
        segment_id: index for index, unit in enumerate(units) for segment_id in unit
    }
    return units, segment_by_id, segment_to_unit


def _precedence_edges(instance: AllocationInstance) -> set[tuple[int, int]]:
    units, segment_by_id, segment_to_unit = _unit_problem(instance)
    edges: set[tuple[int, int]] = set()
    for right, unit in enumerate(units):
        for segment_id in unit:
            for predecessor in segment_by_id[segment_id].predecessor_ids:
                left = segment_to_unit[predecessor]
                if left != right:
                    edges.add((left, right))
    return edges


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


def analyze_state(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    *,
    scheduler_diagnostics: tuple[str, ...] = (),
) -> StateDiagnostic:
    """Return a deterministic proxy violation vector for partial search states."""
    units, segment_by_id, segment_to_unit = _unit_problem(instance)
    _, _, robots, costs = edge_mask_and_costs(instance, context)
    state = canonicalize_state(instance, state)
    orders = state.order_map()
    positions = {
        unit_index: (robot, position)
        for robot, order in orders.items()
        for position, unit_index in enumerate(order)
    }
    missing = sum(robot is None for robot in state.assignments)
    invalid = 0
    loads = {robot: 0.0 for robot in robots}
    for index, robot in enumerate(state.assignments):
        if robot is None:
            continue
        if robot not in robots:
            invalid += 1
            continue
        value = costs[index][robots.index(robot)]
        if not math.isfinite(value):
            invalid += 1
        else:
            loads[robot] += value

    precedence = _precedence_edges(instance)
    order_violations = 0
    for left, right in precedence:
        if left in positions and right in positions:
            left_robot, left_position = positions[left]
            right_robot, right_position = positions[right]
            if left_robot == right_robot and left_position > right_position:
                order_violations += 1

    assigned = {index for index, robot in enumerate(state.assignments) if robot is not None}
    graph_edges = {(left, right) for left, right in precedence if left in assigned and right in assigned}
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

    # The edge oracle's unit cost includes process plus the geometric travel
    # proxy for the current robot. Using process duration alone made partial
    # scoring prefer assignments that the unchanged A1 scheduler later
    # rejected because their arrival/setup time exhausted a tight window.
    durations = {}
    for index, unit in enumerate(units):
        robot = state.assignments[index]
        if robot in robots:
            edge_duration = costs[index][robots.index(robot)]
        else:
            edge_duration = math.inf
        durations[index] = (
            float(edge_duration)
            if math.isfinite(edge_duration)
            else sum(segment_by_id[item].process_duration_s for item in unit)
        )
    starts = {
        index: min(segment_by_id[item].time_window.start_s for item in unit)
        for index, unit in enumerate(units)
    }
    ends = {
        index: min(segment_by_id[item].time_window.end_s for item in unit)
        for index, unit in enumerate(units)
    }
    predecessors = {index: set() for index in assigned}
    for left, right in graph_edges:
        predecessors[right].add(left)
    earliest_end: dict[int, float] = {}
    lateness = 0.0
    intervals: dict[int, tuple[float, float]] = {}
    for index in topological:
        earliest = starts[index]
        if predecessors[index]:
            earliest = max(earliest, max(earliest_end[item] for item in predecessors[index]))
        finish = earliest + durations[index]
        earliest_end[index] = finish
        intervals[index] = (earliest, finish)
        lateness += max(0.0, finish - ends[index])

    resources = {
        index: {
            resource
            for segment_id in unit
            for resource in segment_by_id[segment_id].shared_resource_ids
        }
        for index, unit in enumerate(units)
    }
    resource_overlap = 0.0
    scheduled_indices = sorted(intervals)
    for left_offset, left in enumerate(scheduled_indices):
        for right in scheduled_indices[left_offset + 1 :]:
            if not resources[left] & resources[right]:
                continue
            left_start, left_end = intervals[left]
            right_start, right_end = intervals[right]
            resource_overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))

    failure_segments = _failure_segments(scheduler_diagnostics)
    failure_units = tuple(
        sorted({segment_to_unit[item] for item in failure_segments if item in segment_to_unit})
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


def evaluate_state_timed(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    weights: Mapping[str, float],
    *,
    method_id: str = "a4b-v2-shared-repair",
    clock: Callable[[], int] = time.monotonic_ns,
) -> TimedEvaluation:
    """Evaluate with unchanged A1 semantics and auditable timing components."""
    structural_started = clock()
    state = canonicalize_state(instance, state)
    units = allocation_units(instance)
    diagnostic = analyze_state(instance, context, state)
    structural_s = (clock() - structural_started) / 1e9
    if diagnostic.vector.missing_units or diagnostic.vector.invalid_edges:
        reason = (
            "initializer_incomplete"
            if diagnostic.vector.missing_units
            else "mask_integrity_failure"
        )
        evaluation = StateEvaluation(False, None, None, diagnostic.vector.scalar(), reason)
        return TimedEvaluation(
            evaluation,
            diagnostic,
            EvaluationTiming(structural_s, 0.0, 0.0),
            (),
        )

    orders = {
        robot: tuple(segment for unit_index in order for segment in units[unit_index])
        for robot, order in state.robot_orders
    }
    scheduler_started = clock()
    built = build_schedule(instance, orders, context, method_id)
    scheduler_s = (clock() - scheduler_started) / 1e9
    diagnostic = analyze_state(
        instance, context, state, scheduler_diagnostics=built.diagnostics
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
