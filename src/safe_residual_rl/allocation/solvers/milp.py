"""Small-instance assignment MILP plus deterministic proxy scheduling."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..oracle import OracleContext
from ..schema import AllocationInstance
from .common import SolverProtocol, SolverResult, edge_mask_and_costs, finalize_assignment


def solve_assignment_milp(
    instance: AllocationInstance,
    context: OracleContext,
    protocol: SolverProtocol,
    scheduling_policy: str = "fixed_topological_v1",
    method_id: str = "assignment-milp-v1",
) -> SolverResult:
    """Minimize proxy maximum robot load; this is not a joint path-planning oracle."""
    started = time.perf_counter()
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    unit_count, robot_count = len(units), len(robots)
    if any(not any(math.isfinite(value) for value in row) for row in costs):
        return SolverResult(method_id, "infeasible", None, time.perf_counter() - started, None, None, None, ("UNIT_WITHOUT_FEASIBLE_ROBOT",))

    x_count = unit_count * robot_count
    makespan_index = x_count
    objective = np.zeros(x_count + 1, dtype=float)
    finite_costs = [value for row in costs for value in row if math.isfinite(value)]
    scale = max(finite_costs, default=1.0)
    for unit_index in range(unit_count):
        for robot_index in range(robot_count):
            value = costs[unit_index][robot_index]
            objective[unit_index * robot_count + robot_index] = 1e-6 * (value / scale if math.isfinite(value) else 0.0)
    objective[makespan_index] = 1.0

    lower = np.zeros(x_count + 1, dtype=float)
    upper = np.ones(x_count + 1, dtype=float)
    upper[makespan_index] = np.inf
    for unit_index, row in enumerate(costs):
        for robot_index, value in enumerate(row):
            if not math.isfinite(value):
                upper[unit_index * robot_count + robot_index] = 0.0
    integrality = np.zeros(x_count + 1, dtype=int)
    integrality[:x_count] = 1

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    for unit_index in range(unit_count):
        row = np.zeros(x_count + 1)
        row[unit_index * robot_count : (unit_index + 1) * robot_count] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
    for robot_index in range(robot_count):
        row = np.zeros(x_count + 1)
        for unit_index in range(unit_count):
            value = costs[unit_index][robot_index]
            if math.isfinite(value):
                row[unit_index * robot_count + robot_index] = value
        row[makespan_index] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub)),
        options={"time_limit": protocol.time_limit_s, "mip_rel_gap": protocol.relative_gap, "presolve": True},
    )
    status_names = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    raw_status = status_names.get(int(result.status), "solver_error")
    diagnostics = (
        "ASSIGNMENT_MILP_PLUS_DETERMINISTIC_SCHEDULER",
        "NOT_JOINT_MOTION_PLANNING",
        f"SCIPY_STATUS={result.status}",
        f"SCIPY_MESSAGE={result.message}",
        f"TIME_LIMIT_S={protocol.time_limit_s:g}",
        f"REQUESTED_RELATIVE_GAP={protocol.relative_gap:g}",
    )
    if result.x is None:
        return SolverResult(method_id, raw_status, None, time.perf_counter() - started, _optional_float(result.fun), _attribute_float(result, "mip_dual_bound"), _attribute_float(result, "mip_gap"), diagnostics)
    assignment = {
        unit_index: robots[int(np.argmax(result.x[unit_index * robot_count : (unit_index + 1) * robot_count]))]
        for unit_index in range(unit_count)
    }
    success_status = "optimal" if result.status == 0 else "feasible_limit"
    return finalize_assignment(
        instance,
        context,
        method_id,
        assignment,
        units,
        started,
        diagnostics,
        objective_value=_optional_float(result.fun),
        best_bound=_attribute_float(result, "mip_dual_bound"),
        mip_gap=_attribute_float(result, "mip_gap"),
        success_status=success_status,
        scheduling_policy=scheduling_policy,
        assignment_incumbent=assignment,
    )


def solve_deadline_aware_assignment_milp(
    instance: AllocationInstance,
    context: OracleContext,
    protocol: SolverProtocol,
) -> SolverResult:
    """Frozen assignment MILP followed by the A2 minimum-slack scheduler.

    The MILP remains assignment-only.  Its gap is not a joint scheduling gap.
    """
    return solve_assignment_milp(
        instance,
        context,
        protocol,
        scheduling_policy="deadline_aware_v2",
        method_id="assignment-milp-deadline-schedule-v2",
    )


def solve_hybrid_assignment_milp(
    instance: AllocationInstance,
    context: OracleContext,
    protocol: SolverProtocol,
) -> SolverResult:
    """Assignment MILP with the better fixed/deadline proxy schedule."""
    return solve_assignment_milp(
        instance,
        context,
        protocol,
        scheduling_policy="best_of_fixed_and_deadline_v2",
        method_id="assignment-milp-hybrid-schedule-v2",
    )


def _attribute_float(value: object, name: str) -> float | None:
    return _optional_float(getattr(value, name, None))


def _optional_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None
