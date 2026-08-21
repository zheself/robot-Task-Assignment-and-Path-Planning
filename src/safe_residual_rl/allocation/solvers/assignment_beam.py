"""Deterministic assignment-beam plus branched sequence search."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

from ..oracle import OracleContext
from ..schema import AllocationInstance, AllocationPlan
from ..search_scheduling import search_fixed_assignment_schedule
from ..verifier import verify_plan
from .common import SolverResult, edge_mask_and_costs


@dataclass(frozen=True)
class _AssignmentState:
    assignment: tuple[tuple[int, str], ...]
    loads: tuple[tuple[str, float], ...]


def solve_assignment_beam_sequence(
    instance: AllocationInstance,
    context: OracleContext,
    *,
    assignment_beam_width: int = 24,
    sequence_beam_width: int = 8,
    sequence_node_limit: int = 30_000,
    objective_weights: Mapping[str, float] | None = None,
) -> SolverResult:
    """Branch over assignments before branching robot/resource sequences."""
    method_id = "assignment-beam-sequence-beam-v1"
    started = time.perf_counter()
    if min(assignment_beam_width, sequence_beam_width, sequence_node_limit) < 1:
        raise ValueError("beam budgets must be positive")
    weights = dict(
        objective_weights
        or {
            "makespan": 1.0,
            "load_variance": 0.05,
            "travel_setup_time": 0.1,
            "priority_tardiness": 1.0,
        }
    )
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    eligible = {
        index: tuple(
            (robot_id, value)
            for robot_id, value in zip(robots, row)
            if math.isfinite(value)
        )
        for index, row in enumerate(costs)
    }
    if any(not values for values in eligible.values()):
        return SolverResult(
            method_id,
            "infeasible",
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            ("UNIT_WITHOUT_FEASIBLE_ROBOT",),
        )

    segment_by_id = {item.id: item for item in instance.segments}
    unit_order = sorted(
        range(len(units)),
        key=lambda index: (
            len(eligible[index]),
            min(segment_by_id[item].time_window.end_s for item in units[index]),
            -sum(len(segment_by_id[item].shared_resource_ids) for item in units[index]),
            index,
        ),
    )
    frontier = [
        _AssignmentState((), tuple((robot_id, 0.0) for robot_id in robots))
    ]
    assignment_nodes = 0
    for unit_index in unit_order:
        expanded: list[_AssignmentState] = []
        for state in frontier:
            loads = dict(state.loads)
            for robot_id, cost in eligible[unit_index]:
                next_loads = dict(loads)
                next_loads[robot_id] += cost
                expanded.append(
                    _AssignmentState(
                        tuple(sorted(state.assignment + ((unit_index, robot_id),))),
                        tuple(sorted(next_loads.items())),
                    )
                )
        assignment_nodes += len(expanded)
        frontier = sorted(expanded, key=_assignment_rank)[:assignment_beam_width]

    best_plan: AllocationPlan | None = None
    best_score = math.inf
    sequence_candidates = 0
    sequence_nodes = 0
    for state in frontier:
        unit_assignment = dict(state.assignment)
        assignment = {
            segment_id: unit_assignment[index]
            for index, unit in enumerate(units)
            for segment_id in unit
        }
        searched = search_fixed_assignment_schedule(
            instance,
            assignment,
            context,
            method_id,
            mode="beam",
            objective_weights=weights,
            beam_width=sequence_beam_width,
            node_limit=sequence_node_limit,
        )
        sequence_nodes += searched.nodes_expanded
        if searched.plan is None:
            continue
        checked = verify_plan(instance, searched.plan, context)
        if not checked.feasible:
            continue
        sequence_candidates += 1
        score = _score(searched.plan, weights)
        if score + 1e-12 < best_score:
            best_plan, best_score = searched.plan, score

    diagnostics = (
        "ASSIGNMENT_BEAM_THEN_SEQUENCE_BEAM",
        "MOST_CONSTRAINED_UNIT_FIRST",
        f"ASSIGNMENT_BEAM_WIDTH={assignment_beam_width}",
        f"ASSIGNMENT_NODES={assignment_nodes}",
        f"SEQUENCE_BEAM_WIDTH={sequence_beam_width}",
        f"SEQUENCE_NODE_LIMIT={sequence_node_limit}",
        f"SEQUENCE_NODES={sequence_nodes}",
        f"VERIFIED_SEQUENCE_CANDIDATES={sequence_candidates}",
        "HEURISTIC_NOT_GLOBAL_OPTIMALITY",
        "NOT_MOTION_PLANNING_OR_COLLISION_CERTIFICATE",
    )
    if best_plan is None:
        return SolverResult(
            method_id,
            "schedule_infeasible",
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            diagnostics,
        )
    return SolverResult(
        method_id,
        "feasible",
        best_plan,
        time.perf_counter() - started,
        best_score,
        None,
        None,
        diagnostics,
    )


def _assignment_rank(state: _AssignmentState) -> tuple[float | str, ...]:
    loads = [value for _, value in state.loads]
    mean = sum(loads) / len(loads)
    variance = sum((item - mean) ** 2 for item in loads) / len(loads)
    return (
        max(loads, default=0.0),
        variance,
        sum(loads),
        "|".join(f"{index}:{robot}" for index, robot in state.assignment),
    )


def _score(plan: AllocationPlan, weights: Mapping[str, float]) -> float:
    return sum(float(weights.get(key, 0.0)) * value for key, value in plan.objective_terms)
