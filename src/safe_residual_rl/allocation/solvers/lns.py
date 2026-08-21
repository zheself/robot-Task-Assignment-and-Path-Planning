"""Deterministic assignment-LNS baseline for the A2 benchmark."""

from __future__ import annotations

import math
import time
from typing import Mapping

import numpy as np

from ..oracle import OracleContext
from ..schema import AllocationInstance, AllocationPlan
from ..search_scheduling import search_fixed_assignment_schedule
from ..verifier import verify_plan
from .common import SolverResult, edge_mask_and_costs, finalize_assignment
from .heuristics import solve_load_balanced


def solve_deterministic_lns(
    instance: AllocationInstance,
    context: OracleContext,
    iterations: int = 100,
    seed: int = 0,
    objective_weights: Mapping[str, float] | None = None,
) -> SolverResult:
    """Destroy/reassign units under the A1 mask; no route or repair claim."""
    method_id = "deterministic-assignment-lns-v1"
    started = time.perf_counter()
    weights = dict(objective_weights or {"makespan": 1.0, "load_variance": 0.05, "travel_setup_time": 0.1, "priority_tardiness": 1.0})
    if iterations < 1:
        raise ValueError("iterations must be positive")
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    initial = solve_load_balanced(instance, context)
    if initial.plan is None or initial.status != "feasible":
        return SolverResult(method_id, initial.status, None, time.perf_counter() - started, None, None, None, ("NO_FEASIBLE_INITIAL_PLAN", f"INITIAL_STATUS={initial.status}"))
    assigned_by_segment = {item.segment_id: item.robot_id for item in initial.plan.schedule}
    best_assignment = {index: assigned_by_segment[unit[0]] for index, unit in enumerate(units)}
    best_plan = initial.plan
    best_score = _score(best_plan, weights)
    accepted = 0
    rng = np.random.default_rng(seed)
    destroy_count = max(1, int(math.ceil(0.2 * len(units))))
    for _ in range(iterations):
        candidate = dict(best_assignment)
        destroyed = rng.choice(len(units), size=min(destroy_count, len(units)), replace=False).tolist()
        for unit_index in destroyed:
            candidate.pop(int(unit_index), None)
        loads = {robot_id: 0.0 for robot_id in robots}
        for unit_index, robot_id in candidate.items():
            loads[robot_id] += costs[unit_index][robots.index(robot_id)]
        for unit_index in rng.permutation(destroyed).tolist():
            choices = [
                (loads[robot_id] + value, robot_id, value)
                for robot_id, value in zip(robots, costs[unit_index])
                if math.isfinite(value)
            ]
            if not choices:
                candidate = {}
                break
            _, robot_id, value = min(choices)
            candidate[unit_index] = robot_id
            loads[robot_id] += value
        if len(candidate) != len(units):
            continue
        built = finalize_assignment(instance, context, method_id, candidate, units, time.perf_counter(), ("LNS_CANDIDATE",))
        if built.status != "feasible" or built.plan is None:
            continue
        score = _score(built.plan, weights)
        if score + 1e-12 < best_score:
            best_assignment, best_plan, best_score = candidate, built.plan, score
            accepted += 1
    return SolverResult(
        method_id=method_id,
        status="feasible",
        plan=best_plan,
        runtime_s=time.perf_counter() - started,
        objective_value=best_score,
        best_bound=None,
        mip_gap=None,
        diagnostics=(
            "DETERMINISTIC_ASSIGNMENT_LNS",
            "NOT_A4_CONSTRAINT_REPAIR",
            "NOT_PATH_PLANNING",
            f"ITERATIONS={iterations}",
            f"DESTROY_COUNT={destroy_count}",
            f"ACCEPTED={accepted}",
            f"SEED={seed}",
        ),
    )


def solve_order_aware_lns(
    instance: AllocationInstance,
    context: OracleContext,
    iterations: int = 200,
    seed: int = 0,
    objective_weights: Mapping[str, float] | None = None,
) -> SolverResult:
    """Assignment neighbourhood search with deadline-aware order reconstruction.

    This A2 development baseline can start without a feasible greedy schedule.
    Every candidate assignment is independently ordered by the deterministic
    minimum-slack scheduler.  It is not CP-SAT repair, path planning or proof of
    global infeasibility when no incumbent is found.
    """
    method_id = "order-aware-assignment-lns-v2"
    started = time.perf_counter()
    weights = dict(
        objective_weights
        or {
            "makespan": 1.0,
            "load_variance": 0.05,
            "travel_setup_time": 0.1,
            "priority_tardiness": 1.0,
        }
    )
    if iterations < 1:
        raise ValueError("iterations must be positive")
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    if any(not any(math.isfinite(value) for value in row) for row in costs):
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

    rng = np.random.default_rng(seed)
    base_assignment = _load_balanced_assignment(robots, costs)
    best_assignment: dict[int, str] | None = None
    best_plan: AllocationPlan | None = None
    best_score = math.inf
    feasible_candidates = 0
    accepted = 0
    destroy_count = max(1, int(math.ceil(0.25 * len(units))))

    for iteration in range(iterations + 1):
        anchor = best_assignment if best_assignment is not None else base_assignment
        candidate = dict(anchor)
        if iteration:
            destroyed = rng.choice(
                len(units), size=min(destroy_count, len(units)), replace=False
            ).tolist()
            for unit_index in destroyed:
                candidate.pop(int(unit_index), None)
            candidate = _reinsert_units(
                candidate,
                [int(item) for item in destroyed],
                robots,
                costs,
                rng,
                diversify=best_assignment is None or iteration % 5 == 0,
            )
        if len(candidate) != len(units):
            continue
        built = finalize_assignment(
            instance,
            context,
            method_id,
            candidate,
            units,
            time.perf_counter(),
            ("ORDER_AWARE_LNS_CANDIDATE",),
            scheduling_policy="best_of_fixed_and_deadline_v2",
            schedule_weights=weights,
        )
        if built.status != "feasible" or built.plan is None:
            continue
        feasible_candidates += 1
        score = _score(built.plan, weights)
        if score + 1e-12 < best_score:
            best_assignment = candidate
            best_plan = built.plan
            best_score = score
            accepted += 1

    if best_plan is None:
        return SolverResult(
            method_id,
            "schedule_infeasible",
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            (
                "NO_FEASIBLE_INCUMBENT_AFTER_ORDER_AWARE_SEARCH",
                f"ITERATIONS={iterations}",
                f"DESTROY_COUNT={destroy_count}",
                f"SEED={seed}",
            ),
        )
    return SolverResult(
        method_id,
        "feasible",
        best_plan,
        time.perf_counter() - started,
        best_score,
        None,
        None,
        (
            "ORDER_AWARE_ASSIGNMENT_LNS_V2",
            "FIXED_AND_DEADLINE_ORDER_RECONSTRUCTION",
            "NO_FEASIBLE_INITIAL_SCHEDULE_REQUIRED",
            "NOT_GLOBAL_OPTIMALITY_OR_PATH_PLANNING",
            f"ITERATIONS={iterations}",
            f"DESTROY_COUNT={destroy_count}",
            f"FEASIBLE_CANDIDATES={feasible_candidates}",
            f"ACCEPTED={accepted}",
            f"SEED={seed}",
        ),
    )


def _load_balanced_assignment(
    robots: tuple[str, ...], costs: tuple[tuple[float, ...], ...]
) -> dict[int, str]:
    loads = {robot_id: 0.0 for robot_id in robots}
    result: dict[int, str] = {}
    for unit_index, row in enumerate(costs):
        _, robot_id, value = min(
            (loads[robot_id] + value, robot_id, value)
            for robot_id, value in zip(robots, row)
            if math.isfinite(value)
        )
        result[unit_index] = robot_id
        loads[robot_id] += value
    return result


def _reinsert_units(
    assignment: dict[int, str],
    destroyed: list[int],
    robots: tuple[str, ...],
    costs: tuple[tuple[float, ...], ...],
    rng: np.random.Generator,
    diversify: bool,
) -> dict[int, str]:
    loads = {robot_id: 0.0 for robot_id in robots}
    for unit_index, robot_id in assignment.items():
        loads[robot_id] += costs[unit_index][robots.index(robot_id)]
    for unit_index in rng.permutation(destroyed).tolist():
        choices = sorted(
            (loads[robot_id] + value, robot_id, value)
            for robot_id, value in zip(robots, costs[unit_index])
            if math.isfinite(value)
        )
        if not choices:
            return {}
        choice_index = int(rng.integers(0, min(3, len(choices)))) if diversify else 0
        _, robot_id, value = choices[choice_index]
        assignment[unit_index] = robot_id
        loads[robot_id] += value
    return assignment


def _score(plan: AllocationPlan, weights: Mapping[str, float]) -> float:
    terms = dict(plan.objective_terms)
    return sum(float(weights.get(key, 0.0)) * value for key, value in terms.items())


def solve_beam_alns(
    instance: AllocationInstance,
    context: OracleContext,
    *,
    iterations: int = 40,
    seed: int = 0,
    beam_width: int = 12,
    beam_node_limit: int = 8_000,
    objective_weights: Mapping[str, float] | None = None,
) -> SolverResult:
    """ALNS-style assignment moves plus branched sequence reconstruction."""
    method_id = "beam-sequence-alns-v1"
    started = time.perf_counter()
    weights = dict(
        objective_weights
        or {
            "makespan": 1.0,
            "load_variance": 0.05,
            "travel_setup_time": 0.1,
            "priority_tardiness": 1.0,
        }
    )
    if iterations < 1 or beam_width < 1 or beam_node_limit < 1:
        raise ValueError("ALNS and beam budgets must be positive")
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    if any(not any(math.isfinite(value) for value in row) for row in costs):
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

    rng = np.random.default_rng(seed)
    base = _load_balanced_assignment(robots, costs)
    best_assignment: dict[int, str] | None = None
    best_plan: AllocationPlan | None = None
    best_score = math.inf
    feasible_candidates = 0
    accepted = 0
    destroy_fractions = (0.10, 0.25, 0.40)

    for iteration in range(iterations + 1):
        anchor = best_assignment if best_assignment is not None else base
        candidate = dict(anchor)
        if iteration:
            fraction = destroy_fractions[(iteration - 1) % len(destroy_fractions)]
            destroy_count = max(1, int(math.ceil(fraction * len(units))))
            destroyed = [
                int(item)
                for item in rng.choice(
                    len(units), size=min(destroy_count, len(units)), replace=False
                ).tolist()
            ]
            for unit_index in destroyed:
                candidate.pop(unit_index, None)
            candidate = _reinsert_units(
                candidate,
                destroyed,
                robots,
                costs,
                rng,
                diversify=True,
            )
        if len(candidate) != len(units):
            continue
        assignment = {
            segment_id: candidate[index]
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
            beam_width=beam_width,
            node_limit=beam_node_limit,
        )
        if searched.plan is None:
            continue
        checked = verify_plan(instance, searched.plan, context)
        if not checked.feasible:
            continue
        feasible_candidates += 1
        score = _score(searched.plan, weights)
        if score + 1e-12 < best_score:
            best_assignment = candidate
            best_plan = searched.plan
            best_score = score
            accepted += 1

    diagnostics = (
        "ALNS_ASSIGNMENT_MOVES_WITH_BEAM_SEQUENCE_SEARCH",
        "BRANCHES_ROBOT_AND_SHARED_RESOURCE_ORDER",
        f"ITERATIONS={iterations}",
        f"BEAM_WIDTH={beam_width}",
        f"BEAM_NODE_LIMIT={beam_node_limit}",
        f"FEASIBLE_CANDIDATES={feasible_candidates}",
        f"ACCEPTED={accepted}",
        f"SEED={seed}",
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
