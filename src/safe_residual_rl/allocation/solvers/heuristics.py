"""Deterministic non-learning A1 allocation baselines."""

from __future__ import annotations

import math
import time

from scipy.optimize import linear_sum_assignment

from ..oracle import OracleContext
from ..schema import AllocationInstance
from .common import SolverResult, edge_mask_and_costs, finalize_assignment


def solve_greedy(instance: AllocationInstance, context: OracleContext) -> SolverResult:
    return _solve_incremental(instance, context, "greedy-cost-v1", balance=False)


def solve_load_balanced(instance: AllocationInstance, context: OracleContext) -> SolverResult:
    return _solve_incremental(instance, context, "load-balanced-greedy-v1", balance=True)


def solve_deadline_aware_greedy(
    instance: AllocationInstance, context: OracleContext
) -> SolverResult:
    """Minimum-edge assignment followed by the A2 minimum-slack scheduler."""
    return _solve_incremental(
        instance,
        context,
        "greedy-cost-deadline-schedule-v2",
        balance=False,
        scheduling_policy="deadline_aware_v2",
    )


def solve_deadline_aware_load_balanced(
    instance: AllocationInstance, context: OracleContext
) -> SolverResult:
    """Load-aware assignment followed by the A2 minimum-slack scheduler."""
    return _solve_incremental(
        instance,
        context,
        "load-balanced-deadline-schedule-v2",
        balance=True,
        scheduling_policy="deadline_aware_v2",
    )


def solve_hybrid_load_balanced(
    instance: AllocationInstance, context: OracleContext
) -> SolverResult:
    """Load-aware assignment with best verified fixed/deadline order."""
    return _solve_incremental(
        instance,
        context,
        "load-balanced-hybrid-schedule-v2",
        balance=True,
        scheduling_policy="best_of_fixed_and_deadline_v2",
    )


def _solve_incremental(
    instance: AllocationInstance,
    context: OracleContext,
    method_id: str,
    balance: bool,
    scheduling_policy: str = "fixed_topological_v1",
) -> SolverResult:
    started = time.perf_counter()
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    loads = {robot_id: 0.0 for robot_id in robots}
    assignment: dict[int, str] = {}
    for unit_index, row in enumerate(costs):
        candidates = [
            ((loads[robot_id] if balance else 0.0) + cost, robot_id, cost)
            for robot_id, cost in zip(robots, row)
            if math.isfinite(cost)
        ]
        if not candidates:
            return SolverResult(method_id, "infeasible", None, time.perf_counter() - started, None, None, None, (f"NO_FEASIBLE_ROBOT_FOR_UNIT={unit_index}",))
        _, robot_id, raw_cost = min(candidates)
        assignment[unit_index] = robot_id
        loads[robot_id] += raw_cost
    return finalize_assignment(
        instance,
        context,
        method_id,
        assignment,
        units,
        started,
        ("DETERMINISTIC", "LOAD_AWARE" if balance else "MIN_EDGE_COST"),
        objective_value=max(loads.values(), default=0.0),
        scheduling_policy=scheduling_policy,
        assignment_incumbent=assignment,
    )


def solve_hungarian(instance: AllocationInstance, context: OracleContext) -> SolverResult:
    """Assign units to repeated robot slots, then apply deterministic ordering."""
    started = time.perf_counter()
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    if not units:
        return SolverResult("hungarian-order-v1", "infeasible", None, time.perf_counter() - started, None, None, None, ("NO_UNITS",))
    columns = [(robot_id, slot) for robot_id in robots for slot in range(len(units))]
    finite = [value for row in costs for value in row if math.isfinite(value)]
    if not finite or any(not any(math.isfinite(value) for value in row) for row in costs):
        return SolverResult("hungarian-order-v1", "infeasible", None, time.perf_counter() - started, None, None, None, ("UNIT_WITHOUT_FEASIBLE_ROBOT",))
    large = max(finite) * (len(units) + 1) * 1_000.0 + 1.0
    matrix = []
    for row in costs:
        matrix.append([
            large if not math.isfinite(row[robots.index(robot_id)]) else row[robots.index(robot_id)] * (1.0 + slot)
            for robot_id, slot in columns
        ])
    row_indices, column_indices = linear_sum_assignment(matrix)
    assignment: dict[int, str] = {}
    objective = 0.0
    for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
        if matrix[row_index][column_index] >= large:
            return SolverResult("hungarian-order-v1", "infeasible", None, time.perf_counter() - started, None, None, None, ("FORBIDDEN_EDGE_SELECTED",))
        assignment[row_index] = columns[column_index][0]
        objective += matrix[row_index][column_index]
    return finalize_assignment(
        instance,
        context,
        "hungarian-order-v1",
        assignment,
        units,
        started,
        ("HUNGARIAN_REPEATED_ROBOT_SLOTS", "DETERMINISTIC_TOPOLOGICAL_ORDER"),
        objective_value=objective,
        assignment_incumbent=assignment,
    )
