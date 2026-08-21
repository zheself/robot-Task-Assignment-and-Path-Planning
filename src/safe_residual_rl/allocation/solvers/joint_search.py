"""Small-instance joint assignment/sequencing reference for the A1 proxy."""

from __future__ import annotations

import itertools
import math
import time
from typing import Mapping

from ..oracle import OracleContext
from ..schema import AllocationInstance, AllocationPlan
from ..search_scheduling import search_fixed_assignment_schedule
from ..verifier import verify_plan
from .common import SolverResult, edge_mask_and_costs


def solve_joint_assignment_sequence_reference(
    instance: AllocationInstance,
    context: OracleContext,
    *,
    objective_weights: Mapping[str, float] | None = None,
    max_segments: int = 10,
    max_assignment_combinations: int = 20_000,
    node_limit: int = 250_000,
    time_limit_s: float = 10.0,
) -> SolverResult:
    """Enumerate admissible unit assignments and all proxy schedule branches.

    ``optimal`` means complete enumeration only within the declared A1 proxy
    model and supplied bounds. It is not path, collision or factory optimality.
    """
    method_id = "joint-assignment-sequence-reference-v1"
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
    if min(max_segments, max_assignment_combinations, node_limit) < 1 or time_limit_s <= 0:
        raise ValueError("joint reference limits must be positive")
    if len(instance.segments) > max_segments:
        return SolverResult(
            method_id,
            "unsupported_scale",
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            (f"SEGMENTS={len(instance.segments)}", f"MAX_SEGMENTS={max_segments}"),
        )

    _, units, robots, costs = edge_mask_and_costs(instance, context)
    eligible = [
        tuple(robot_id for robot_id, value in zip(robots, row) if math.isfinite(value))
        for row in costs
    ]
    if any(not row for row in eligible):
        return SolverResult(
            method_id,
            "infeasible",
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            ("UNIT_WITHOUT_FEASIBLE_ROBOT", "COMPLETE_EDGE_INFEASIBILITY"),
        )
    combination_count = math.prod(len(row) for row in eligible)
    if combination_count > max_assignment_combinations:
        return SolverResult(
            method_id,
            "unsupported_scale",
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            (
                f"ASSIGNMENT_COMBINATIONS={combination_count}",
                f"MAX_ASSIGNMENT_COMBINATIONS={max_assignment_combinations}",
            ),
        )

    best_plan: AllocationPlan | None = None
    best_score = math.inf
    total_nodes = 0
    assignments_completed = 0
    search_complete = True
    for selected in itertools.product(*eligible):
        elapsed = time.perf_counter() - started
        remaining_time = time_limit_s - elapsed
        remaining_nodes = node_limit - total_nodes
        if remaining_time <= 0 or remaining_nodes <= 0:
            search_complete = False
            break
        unit_assignment = dict(enumerate(selected))
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
            mode="exact",
            objective_weights=weights,
            node_limit=remaining_nodes,
            time_limit_s=remaining_time,
        )
        total_nodes += searched.nodes_expanded
        if not searched.search_complete:
            search_complete = False
            if searched.plan is not None:
                score = _score(searched.plan, weights)
                if score + 1e-12 < best_score:
                    best_plan, best_score = searched.plan, score
            break
        assignments_completed += 1
        if searched.plan is not None:
            score = _score(searched.plan, weights)
            if score + 1e-12 < best_score:
                best_plan, best_score = searched.plan, score

    diagnostics = (
        "JOINT_ASSIGNMENT_AND_SEQUENCE_ENUMERATION",
        f"ASSIGNMENT_COMBINATIONS={combination_count}",
        f"ASSIGNMENTS_COMPLETED={assignments_completed}",
        f"NODES_EXPANDED={total_nodes}",
        f"SEARCH_COMPLETE={search_complete}",
        f"TIME_LIMIT_S={time_limit_s:g}",
        f"NODE_LIMIT={node_limit}",
        "OPTIMAL_ONLY_WITHIN_A1_PROXY_IF_SEARCH_COMPLETE",
        "NOT_PATH_COLLISION_OR_FACTORY_OPTIMALITY",
    )
    if best_plan is None:
        status = "infeasible" if search_complete else "limit"
        return SolverResult(
            method_id,
            status,
            None,
            time.perf_counter() - started,
            None,
            None,
            None,
            diagnostics,
        )
    checked = verify_plan(instance, best_plan, context)
    if not checked.feasible:
        return SolverResult(
            method_id,
            "verification_failed",
            best_plan,
            time.perf_counter() - started,
            best_score,
            None,
            None,
            diagnostics
            + tuple(f"VERIFY={item.code}" for item in checked.violations),
        )
    return SolverResult(
        method_id,
        "optimal" if search_complete else "feasible_limit",
        best_plan,
        time.perf_counter() - started,
        best_score,
        best_score if search_complete else None,
        None,
        diagnostics + checked.diagnostics,
    )


def _score(plan: AllocationPlan, weights: Mapping[str, float]) -> float:
    return sum(float(weights.get(key, 0.0)) * value for key, value in plan.objective_terms)
